import os
import pickle
import math
import json
import itertools
import copy
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Import classification metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer

from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad.tensor import Tensor
from tinygrad import nn, TinyJit
from tinygrad.helpers import GlobalCounters, Context # Added Context




class Colors:
    HEADER = '\033[95m'; INFO = '\033[94m'; SUCCESS = '\033[92m'
    WARNING = '\033[93m'; ERROR = '\033[91m'; BOLD = '\033[1m'
    UNDERLINE = '\033[4m'; END = '\033[0m'

def log(message, level='INFO'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    if level == 'INFO': color = Colors.INFO
    elif level == 'SUCCESS': color = Colors.SUCCESS
    elif level == 'WARNING': color = Colors.WARNING
    elif level == 'ERROR': color = Colors.ERROR
    else: color = Colors.INFO
    print(f"{color}[{timestamp}] {level:<8}{Colors.END} {message}")

CONFIG = {
    "cache_path": "docking_relax_perturb_delta.pkl",
    "sample_size": "inf",
    "test_split": 0.4,
    "batch_size": 700,
    "epochs": 400,
    "patience_threshold": 40,
    "loss_choice": "bce",
    "rmsd_threshold_good_pose": 2.0,
    "random_state": 42,
    "learning_rate": 0.00005,
    "dropout_rate": 0.2,  # adjust dropout
    "hidden_dims": {
        "hidden_dim1": 64,
        "hidden_dim2": 32,
        "hidden_dim3": 16,
        "hidden_dim4": 8,
    },
    "hyperparam_tuning_enabled": False,
}
#[2025-04-29 22:23:15.030] INFO     Epoch 004, LR: 1.0E-04, Train Loss: 0.2209, Val Loss: 0.185



def load_dataset(cache_path):
    with open(cache_path, 'rb') as f: log("Loading cached docking data.", "INFO"); docking_data = pickle.load(f)
    log(f"Loaded {len(docking_data)} records.", "SUCCESS"); return pd.DataFrame(docking_data)


# Validation improved to 0.1228


class ClassifierNet:
    def __init__(self, input_size, hidden_dims):
        self.l1 = nn.Linear(input_size, hidden_dims["hidden_dim1"], bias=True)
        self.bn1 = nn.BatchNorm(hidden_dims["hidden_dim1"], track_running_stats=True, eps=1e-5, momentum=0.1)

        self.l2 = nn.Linear(hidden_dims["hidden_dim1"], hidden_dims["hidden_dim2"], bias=True)
        self.bn2 = nn.BatchNorm(hidden_dims["hidden_dim2"], track_running_stats=True, eps=1e-5, momentum=0.1)

        self.l3 = nn.Linear(hidden_dims["hidden_dim2"], hidden_dims["hidden_dim3"], bias=True)
        self.bn3 = nn.BatchNorm(hidden_dims["hidden_dim3"], track_running_stats=True, eps=1e-5, momentum=0.1)

        self.l4 = nn.Linear(hidden_dims["hidden_dim3"], hidden_dims["hidden_dim4"], bias=True)
        self.bn4 = nn.BatchNorm(hidden_dims["hidden_dim4"], track_running_stats=True, eps=1e-5, momentum=0.1)


        self.l6 = nn.Linear(hidden_dims["hidden_dim4"], 1, bias=True)
        self.dropout_rate = CONFIG["dropout_rate"]

    def __call__(self, x):
        x = self.l1(x)
        x = self.bn1(x)
        x = x.relu()
        x = x.dropout(p=self.dropout_rate)

        x = self.l2(x)
        x = self.bn2(x)
        x = x.relu()
        x = x.dropout(p=self.dropout_rate)

        x = self.l3(x)
        x = self.bn3(x)
        x = x.relu()
        x = x.dropout(p=self.dropout_rate)

        x = self.l4(x)
        x = self.bn4(x)
        x = x.relu()
        x = x.dropout(p=self.dropout_rate)


        x = self.l6(x).sigmoid()
        return x


def calculate_loss(y_pred: Tensor, y_batch: Tensor, loss_choice: str):
    if loss_choice == 'bce':
        # weighted BCE using computed pos_weight
        return weighted_bce_loss(y_pred, y_batch).mean()
    elif loss_choice == 'focal':
         return focal_loss(y_pred, y_batch, gamma=2.0, alpha=0.25)
    else:
         raise ValueError(f"Unknown loss_choice: {loss_choice}")




def weighted_bce_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    y_pred = y_pred.clip(1e-7, 1 - 1e-7)
    loss = y_pred.binary_crossentropy(y_true).mean()
    return loss

def focal_loss(y_pred: Tensor, y_true: Tensor, gamma: float = 2.0, alpha: float = 0.25) -> Tensor:
    # Clip predictions to prevent log(0)
    y_pred = y_pred.clip(1e-7, 1 - 1e-7)
    # Compute the probability term (pt)
    pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    # Compute the focal loss
    loss = -alpha * ((1 - pt) ** gamma) * (y_true * y_pred.log()) \
           - (1 - alpha) * (pt ** gamma) * ((1 - y_true) * (1 - y_pred).log())
    return loss.mean()


@TinyJit
@Tensor.train()
def train_step(X_batch: Tensor, y_batch: Tensor, net, opt, loss_choice: str):
    opt.zero_grad()
    y_pred = net(X_batch) 
    loss = calculate_loss(y_pred, y_batch, loss_choice)
    loss.backward()
    opt.step()
    return loss.realize()

@TinyJit
def evaluate(x: Tensor, net):
    return net(x).realize()


def run_training(config, tuning_mode=False, model_save_path=None):
    np.random.seed(config["random_state"])
    df = load_dataset(config["cache_path"])

    RMSD_THRESHOLD_GOOD = config["rmsd_threshold_good_pose"]
    df['is_good'] = (df["rmsd_to_input"] < RMSD_THRESHOLD_GOOD).astype(int)
    y_class = df['is_good']

    feature_cols = ['angle_constraint', 'atom_pair_constraint', 'chainbreak', 'coordinate_constraint',
                    'dihedral_constraint', 'dslf_ca_dih', 'dslf_cs_ang', 'dslf_ss_dih', 'dslf_ss_dst',
                    'fa_atr', 'fa_dun', 'fa_elec', 'fa_intra_rep', 'fa_pair', 'fa_rep',
                    'fa_sol', 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb',
                    'omega', 'p_aa_pp', 'pro_close', 'rama', 'ref']

    # df['fa_rep'] = df['fa_rep'].clip(lower=df['fa_rep'].quantile(0.01),
    #                                  upper=df['fa_rep'].quantile(0.99))

    df['atr_rep_ratio'] = df['fa_atr'] / (df['fa_rep'] + 1e-8)  # Attractive/repulsive ratio
    df['hbond_total'] = df['hbond_bb_sc'] + df['hbond_sc']  # Total hydrogen bonding
    df['atr_rep_sum'] = df['fa_atr'] + df['fa_rep']
    feature_cols.extend(['atr_rep_ratio', 'atr_rep_sum', 'hbond_total'])
    feature_cols.remove('hbond_bb_sc')
    feature_cols.remove('hbond_sc')

    X = df[feature_cols]

    selector = VarianceThreshold(threshold=0.0)
    try:
        X_filtered = selector.fit_transform(X)
        kept_indices = selector.get_support(indices=True)
        if len(kept_indices) < X.shape[1]:
             feature_cols_kept = [feature_cols[i] for i in kept_indices]
             X = pd.DataFrame(X_filtered, columns=feature_cols_kept, index=X.index)
             log(f"Removed zero-variance features. Kept {len(feature_cols_kept)} features.", "INFO")
        else:
             log("No zero-variance features found.", "INFO")
             feature_cols_kept = feature_cols
    except ValueError as e:
         log(f"Error during VarianceThreshold: {e}. Skipping feature selection.", "ERROR")
         feature_cols_kept = feature_cols

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    input_size = X_scaled.shape[1]

    X_train, X_val, y_train_class, y_val_class = train_test_split(
        X_scaled, y_class, test_size=config["test_split"], stratify=y_class
    )

    log(f"Data split: Train: {len(X_train)}, Val: {len(X_val)}", "SUCCESS")


    net = ClassifierNet(input_size, config["hidden_dims"])
    opt = nn.optim.Adam(params=nn.state.get_parameters(net), lr=config["learning_rate"])


    X_train_np = np.array(X_train, dtype=np.float32)
    y_train_np = np.array(y_train_class, dtype=np.int32).reshape(-1, 1)
    X_val_tensor = Tensor(np.array(X_val, dtype=np.float32), requires_grad=False)
    #print(X_val_tensor.numpy())
    y_val_tensor = Tensor(np.array(y_val_class, dtype=np.int32).reshape(-1, 1), requires_grad=False)
    #print(y_val_tensor.numpy())

    n_samples = len(X_train_np)
    indices = np.arange(n_samples)
    batch_size = config["batch_size"]
    loss_choice = config["loss_choice"]

    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    epochs_run = 0


    for epoch in range(100):
        epochs_run += 1
        np.random.shuffle(indices)
        X_shuffled = X_train_np[indices]
        y_shuffled = y_train_np[indices]
        epoch_loss = 0.0
        batches = 0

        for i in range(0, n_samples, batch_size):
            GlobalCounters.reset()
            X_batch_np = X_shuffled[i:i + batch_size]
            y_batch_np = y_shuffled[i:i + batch_size]
            X_batch = Tensor(X_batch_np, requires_grad=False)
            y_batch = Tensor(y_batch_np, requires_grad=False)
            #print(y_batch.numpy())

            if X_batch.shape[0] == batch_size:
                loss = train_step(X_batch, y_batch, net, opt, loss_choice)
            else:
                Tensor.training = True
                opt.zero_grad()
                y_pred = net(X_batch)
                loss = calculate_loss(y_pred, y_batch, loss_choice)
                loss.backward()
                opt.step()
                Tensor.training = False

            epoch_loss += loss.numpy()
            batches += 1

        avg_train_loss = epoch_loss / batches

        Tensor.training = False
        val_preds = evaluate(X_val_tensor, net)
        val_loss = calculate_loss(val_preds, y_val_tensor, loss_choice)
        val_loss_np = val_loss.numpy()
        Tensor.training = True

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_loss_np)
        current_lr = opt.lr.numpy().item()
        log(f"Epoch {epoch:03d}, LR: {current_lr:.1E}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss_np:.4f}",
            "INFO")

        if val_loss_np < best_val_loss:
            best_val_loss = val_loss_np
            safe_save(get_state_dict(net), "weights/best_classifier_model.safetensors")
            log(f"Validation improved to {best_val_loss:.4f}", "SUCCESS")
            patience_counter = 0


    log(f"Training completed in {epochs_run} epochs.", "SUCCESS")

    return best_val_loss, net


def main():
    run_training(CONFIG, tuning_mode=False)


if __name__ == "__main__":
    main()