import os
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import torch.nn.functional as F
import math

# Import loss functions from allrank
from allrank.models.losses import lambdaLoss, DEFAULT_EPS
from allrank.models.losses import ndcgLoss1_scheme, ndcgLoss2_scheme, lambdaRank_scheme
from allrank.models.losses import ndcgLoss2PP_scheme, rankNet_scheme
# Import other losses as needed
from allrank.models.losses import approxNDCGLoss, listNet, listMLE, rankNet
from allrank.data.dataset_loading import PADDED_Y_VALUE

# Configuration for ranking
CONFIG = {
    "dropout_rate": 0.2,  # Increased dropout to reduce overfitting
    "cache_path": "docking_stdpose_perturbrelax_delta.pkl",
    "test_split": 0.2,
    "batch_size": 10000,
    "epochs": 100, 
    "lr": 0.00001,  # Slightly higher learning rate to start
    "start_margin": 0.1,
    "target_margin": 0.1,
    "margin_schedule_epochs": 40,
    "weight_decay": 0.0,  # Increased weight decay for better regularization
    "random_state": 42,
    "hidden_dims": {
        # wider first layers for more capacity
        "hidden_dim1": 1024,
        "hidden_dim2": 512,
        "hidden_dim3": 256,
        "hidden_dim4": 128,
        "hidden_dim5": 54,
    },
    "early_stop_patience": 15,
    "hard_neg_rel_diff": 1,
    "loss_type": "success_aware_ranking_loss",  # "ranking_loss", "lambda_loss", "approxNDCG", "listNet", "listMLE", "rankNet"
    "lambda_loss_config": {
        "weighing_scheme": "lambdaRank_scheme",  #  None, "ndcgLoss1_scheme", "ndcgLoss2_scheme", "lambdaRank_scheme", "ndcgLoss2PP_scheme", "rankNet_scheme"
        "k": 10,  # truncation parameter for DCG
        "sigma": 1.0,  # score difference weight in sigmoid
        "mu": 10.0,  # weight used in ndcgLoss2PP_scheme
        "reduction": "sum",  # reduction method: "sum" or "mean"
        "reduction_log": "binary"  # log base: "binary" or "natural"
    },
    "rmsd_threshold": 2.0,
    "top_k_weight": 2.0
}

class Colors:
    INFO = '\033[94m'; SUCCESS = '\033[92m'; WARNING = '\033[93m'; ERROR = '\033[91m'; END = '\033[0m'

def log(msg, level='INFO'):
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    col = getattr(Colors, level, Colors.INFO)
    print(f"{col}[{t}] {level} {Colors.END}{msg}")

class FCRankingNet(nn.Module):
    def __init__(self, input_size, hidden_dims, dropout_rate=0.1):
        super().__init__()
        h1, h2, h3, h4, h5 = hidden_dims['hidden_dim1'], hidden_dims['hidden_dim2'], hidden_dims['hidden_dim3'], hidden_dims['hidden_dim4'], hidden_dims['hidden_dim5']
        
        self.fc1 = nn.Linear(input_size, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        
        self.fc4 = nn.Linear(h3, h4)
        self.bn4 = nn.BatchNorm1d(h4)

        self.fc5 = nn.Linear(h4, h5)
        self.bn5 = nn.BatchNorm1d(h5)
        
        self.output = nn.Linear(h5, 1)
        
        self.act = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x = self.act(x)
        x = self.dropout(x)
        
        return self.output(x).squeeze(-1)

# Pairwise hinge ranking loss in PyTorch
def ranking_loss(y_pred: torch.Tensor, y_true: torch.Tensor, margin: float, hard_neg_rel_diff: int = None) -> torch.Tensor:
    # LambdaRank-style pairwise hinge with ΔNDCG weights and optional hard-negative sampling
    pred = y_pred.view(-1)
    true = y_true.view(-1)
    B = pred.size(0)
    # compute ranks based on current predictions
    _, idx = torch.sort(pred, descending=True)
    ranks = torch.empty_like(pred)
    ranks[idx] = torch.arange(1, B+1, device=pred.device, dtype=torch.float)
    # discount weights 1/log2(rank+1)
    discount = 1.0 / torch.log2(ranks + 1.0)
    # pairwise differences
    diff = pred.unsqueeze(1) - pred.unsqueeze(0)  # BxB
    true_diff = (true.unsqueeze(1) - true.unsqueeze(0)).float()
    # mask for true-label ordering
    mask = (true_diff > 0).float()
    # hard-negative sampling: only include pairs where relevance difference <= threshold
    if hard_neg_rel_diff is not None:
        mask = mask * ((true_diff <= hard_neg_rel_diff).float())
    # ΔDCG weight: absolute difference of discounts
    delta_ndcg = torch.abs(discount.unsqueeze(1) - discount.unsqueeze(0))
    weight = delta_ndcg * mask
    # hinge loss weighted by ΔNDCG
    loss_matrix = torch.relu(margin - diff) * weight
    # normalize
    denom = weight.sum() + 1e-8
    return loss_matrix.sum() / denom


def success_aware_ranking_loss(y_pred, y_true, margin, threshold=2.0, top_k_weight=5.0):
    # Standard ranking components
    pred = y_pred.view(-1)
    true = y_true.view(-1)

    # Add special weighting for threshold-critical pairs
    threshold_mask = (y_true < threshold).float()

    # Create masks for critical pairs (below vs above threshold)
    critical_pairs = threshold_mask.unsqueeze(1) * (1 - threshold_mask.unsqueeze(0))

    # Weight critical pairs more heavily
    weight = torch.ones_like(pred.unsqueeze(1) - pred.unsqueeze(0))
    weight = weight + (top_k_weight - 1.0) * critical_pairs

    # Apply standard ranking loss with custom weighting
    diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    true_diff = (true.unsqueeze(1) - true.unsqueeze(0)).float()
    mask = (true_diff > 0).float()

    loss_matrix = torch.relu(margin - diff) * mask * weight
    return loss_matrix.sum() / (mask.sum() + 1e-8)

def run_training(config, model_save_path=None):
    df = pd.DataFrame(pickle.load(open(config['cache_path'], 'rb')))
    print(df.head(10))
    # assign finer graded relevance labels: 5:<0.25Å, 4:<0.5Å, 3:<1Å, 2:<1.5Å, 1:<2Å, 0:>=2Å
    bins = [-np.inf, 1.0, 1.5, 2.0, 3.0, np.inf]
    labels = [4, 3, 2, 1, 0]
    df['relevance'] = pd.cut(df['rmsd_to_input'], bins=bins, labels=labels).astype(int)
    features = ['angle_constraint','atom_pair_constraint','chainbreak','coordinate_constraint',
                'dihedral_constraint','dslf_ca_dih','dslf_cs_ang','dslf_ss_dih','dslf_ss_dst',
                'fa_atr','fa_dun','fa_elec','fa_intra_rep','fa_pair','fa_rep','fa_sol',
                'hbond_bb_sc','hbond_lr_bb','hbond_sc','hbond_sr_bb','omega','p_aa_pp','pro_close','rama','ref']
    df['fa_rep'] = df['fa_rep'].clip(df['fa_rep'].quantile(0.01), df['fa_rep'].quantile(0.99))
    df['atr_rep_ratio'] = df['fa_atr']/(df['fa_rep']+1e-8)
    df['hbond_total'] = df['hbond_bb_sc']+df['hbond_sc']
    df['atr_rep_sum'] = df['fa_atr']+df['fa_rep']
    for c in ['hbond_bb_sc','hbond_sc']: features.remove(c)
    features += ['atr_rep_ratio','hbond_total','atr_rep_sum']
    X = df[features].fillna(0).values.astype(np.float32)

    print("Using features:", features)
    y = df['relevance'].values.astype(np.float32)
    groups = df['pdb_id'].values
    gss = GroupShuffleSplit(n_splits=1, test_size=config['test_split'], random_state=config['random_state'])
    train_idx, val_idx = next(gss.split(X, y, groups))
    X_train, X_val = X[train_idx], X[val_idx]
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    y_train, y_val = y[train_idx], y[val_idx]
    train_pids = groups[train_idx]
    val_pids = groups[val_idx]
    train_groups = {pid: np.where(train_pids==pid)[0] for pid in np.unique(train_pids)}
    val_groups = {pid: np.where(val_pids==pid)[0] for pid in np.unique(val_pids)}
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    net = FCRankingNet(X_train.shape[1], config['hidden_dims'], config["dropout_rate"]).to(device)
    optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    # use plateau scheduler to reduce LR on stalling validation loss
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_loss = float('inf')
    early_stop_patience = config['early_stop_patience']
    patience_counter = 0
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    best_metric = -1.0  # composite of success_t1 + success_t5
    best_success = (0.0, 0.0)

    # Loss type selection
    loss_type = config.get("loss_type", "ranking_loss")
    if loss_type == "lambda_loss":
        log(f"Using Lambda Loss with weighing scheme: {config['lambda_loss_config'].get('weighing_scheme', 'None')}", "INFO")
    elif loss_type in ["approxNDCG", "listNet", "listMLE", "rankNet"]:
        log(f"Using {loss_type} loss from allrank library", "INFO")

    start_margin = config["start_margin"]
    target_margin = config["target_margin"]
    schedule_epochs = config["margin_schedule_epochs"]

    for epoch in range(config['epochs']):
        net.train()

        if epoch < schedule_epochs:
            current_margin = start_margin + (target_margin - start_margin) * (epoch / schedule_epochs)
        else:
            current_margin = target_margin

        # per-query ranking updates
        for pid, idxs in train_groups.items():
            if len(idxs) < 2: continue
            xb = torch.from_numpy(X_train[idxs]).to(device)
            yb = torch.from_numpy(y_train[idxs]).to(device)
            optimizer.zero_grad()
            preds = net(xb)
            
            if loss_type == "success_aware_ranking_loss":
                loss = success_aware_ranking_loss(
                    preds,
                    yb,
                    current_margin,
                    threshold=config["rmsd_threshold"],
                    top_k_weight=config["top_k_weight"]
                )
            elif loss_type == "lambda_loss":
                lambda_config = config.get('lambda_loss_config', {})
                loss = lambdaLoss(
                    preds.unsqueeze(0),  # Lambda loss expects [batch_size, slate_length]
                    yb.unsqueeze(0),     # Same for ground truth
                    eps=DEFAULT_EPS,
                    weighing_scheme=lambda_config.get('weighing_scheme'),
                    k=lambda_config.get('k', 10),
                    sigma=lambda_config.get('sigma', 1.0),
                    mu=lambda_config.get('mu', 10.0),
                    reduction=lambda_config.get('reduction', 'sum'),
                    reduction_log=lambda_config.get('reduction_log', 'binary')
                )
            elif loss_type == "approxNDCG":
                loss = approxNDCGLoss(preds.unsqueeze(0), yb.unsqueeze(0))
            elif loss_type == "listNet":
                loss = listNet(preds.unsqueeze(0), yb.unsqueeze(0))
            elif loss_type == "listMLE":
                loss = listMLE(preds.unsqueeze(0), yb.unsqueeze(0))
            elif loss_type == "rankNet":
                loss = rankNet(preds.unsqueeze(0), yb.unsqueeze(0))
            else:
                loss = ranking_loss(preds, yb, current_margin, config['hard_neg_rel_diff'])
            
            loss.backward()
            # gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
        
        # validation
        net.eval()
        with torch.no_grad():
            val_losses = []
            # per-query validation loss
            for pid, idxs in val_groups.items():
                if len(idxs) < 2: continue
                xb = torch.from_numpy(X_val[idxs]).to(device)
                yb = torch.from_numpy(y_val[idxs]).to(device)
                
                if loss_type == "success_aware_ranking_loss":
                    val_loss_margin = target_margin
                    val_loss = success_aware_ranking_loss(
                        net(xb),
                        yb,
                        val_loss_margin,
                        threshold=config['rmsd_threshold'],
                        top_k_weight=config["top_k_weight"]
                    ).item()
                elif loss_type == "lambda_loss":
                    lambda_config = config.get('lambda_loss_config', {})
                    val_loss = lambdaLoss(
                        net(xb).unsqueeze(0),
                        yb.unsqueeze(0),
                        eps=DEFAULT_EPS,
                        weighing_scheme=lambda_config.get('weighing_scheme'),
                        k=lambda_config.get('k', 10),
                        sigma=lambda_config.get('sigma', 1.0),
                        mu=lambda_config.get('mu', 10.0),
                        reduction=lambda_config.get('reduction', 'sum'),
                        reduction_log=lambda_config.get('reduction_log', 'binary')
                    ).item()
                elif loss_type == "approxNDCG":
                    val_loss = approxNDCGLoss(net(xb).unsqueeze(0), yb.unsqueeze(0)).item()
                elif loss_type == "listNet":
                    val_loss = listNet(net(xb).unsqueeze(0), yb.unsqueeze(0)).item()
                elif loss_type == "listMLE":
                    val_loss = listMLE(net(xb).unsqueeze(0), yb.unsqueeze(0)).item()
                elif loss_type == "rankNet":
                    val_loss = rankNet(net(xb).unsqueeze(0), yb.unsqueeze(0)).item()
                else:
                    val_loss_margin = target_margin
                    val_loss = ranking_loss(net(xb), yb, val_loss_margin, config['hard_neg_rel_diff']).item()
                
                val_losses.append(val_loss)
        
        avg_val = np.mean(val_losses)
        with torch.no_grad():
            X_val_tensor = torch.from_numpy(X_val.astype(np.float32)).to(device)
            scores_val = net(X_val_tensor).cpu().numpy().flatten()
        val_df['score'] = scores_val
        # compute per-query success rates
        metrics = []
        for pid, group in val_df.groupby('pdb_id'):
            df_sorted = group.sort_values('score', ascending=False)
            s1 = (df_sorted.iloc[0]['rmsd_to_input'] < config.get('rmsd_threshold', 2.0))
            s5 = any(df_sorted.head(5)['rmsd_to_input'] < config.get('rmsd_threshold', 2.0))
            metrics.append((s1, s5))
        metrics = np.array(metrics, dtype=float)
        success_t1 = metrics[:,0].mean() * 100
        success_t5 = metrics[:,1].mean() * 100
        composite = success_t1 + success_t5
        improved = composite > best_metric
        if improved:
            best_metric = composite
            best_success = (success_t1, success_t5)
            torch.save(net.state_dict(), model_save_path or 'weights/best_ranker_model_raw_energies_3.pt')
            print("Saved best model.")
            patience_counter = 0
        else:
            patience_counter += 1
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, LR: {current_lr:.5f}, Margin: {current_margin:.3f}, Val Loss: {avg_val:.4f}, s@1: {success_t1:.2f}%, s@5: {success_t5:.2f}%")
        scheduler.step(avg_val)
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}; best composite metric {best_metric:.2f} (s@1: {best_success[0]:.2f}%, s@5: {best_success[1]:.2f}%)")
            break
    print(f"Training done. Best val loss: {best_loss:.4f}, best s@1: {best_success[0]:.2f}%, s@5: {best_success[1]:.2f}%.")
    return best_loss, net

if __name__ == '__main__':
    best_val_loss, net = run_training(CONFIG)
    print(f"Best validation loss: {best_val_loss:.4f}")
