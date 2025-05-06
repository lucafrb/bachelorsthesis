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
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupShuffleSplit
import torch.nn.functional as F
import math
# Add these imports
from torch.utils.data import Dataset, DataLoader
import random
# Add defaultdict
from collections import defaultdict

# Configuration for ranking
CONFIG = {
    "dropout_rate": 0.1,
    "cache_path": "docking_relax_perturb_delta.pkl",
    "test_split": 0.1,
    "batch_size": 10000,
    "epochs": 100, 
    "lr": 0.01,
    "start_margin": 0.05,
    "target_margin": 0.05,
    "margin_schedule_epochs": 30, 
    "weight_decay": 1e-4,
    "random_state": 42,
    "hidden_dims": {
        "hidden_dim1": 1024,
        "hidden_dim2": 512,
        "hidden_dim3": 256,
        "hidden_dim4": 128,
        "hidden_dim5": 64,
    },
    "early_stop_patience": 20,
    "hard_neg_rel_diff": 1,
    "loss_type": "margin_ranking",}

# --- Add RankingPairDataset ---
class RankingPairDataset(Dataset):
    def __init__(self, features, relevance, groups):
        self.features = features
        self.relevance = relevance
        # groups is expected to be a dict {pdb_id: [indices...]}
        self.groups = groups
        print("RankingPairDataset: Initialized.")
        # Pre-calculate relevance groups within each PDB for faster sampling
        self.relevance_groups = defaultdict(lambda: defaultdict(list))
        print("RankingPairDataset: Pre-calculating relevance groups...")
        for pdb_id, indices in self.groups.items():
            if len(indices) < 2: continue # Skip PDBs with less than 2 poses
            for idx_global in indices: # Iterate directly over global indices for this PDB
                # Ensure index is within bounds of self.relevance
                if idx_global < len(self.relevance):
                    rel = self.relevance[idx_global]
                    self.relevance_groups[pdb_id][rel].append(idx_global)
                # else: # Optional: Log if an index is out of bounds
                #     print(f"Warning: Index {idx_global} for PDB {pdb_id} out of bounds for relevance array (len {len(self.relevance)}). Skipping.")

            # Remove PDBs that ended up with only one relevance level after grouping
            if pdb_id in self.relevance_groups and len(self.relevance_groups[pdb_id]) < 2:
                del self.relevance_groups[pdb_id]

        # Update pdb_ids to only include those with multiple relevance levels
        self.valid_pdb_ids = list(self.relevance_groups.keys())
        print(f"RankingPairDataset: Found {len(self.valid_pdb_ids)} PDBs with multiple relevance levels for training pairs.")
        if not self.valid_pdb_ids:
            raise ValueError("No PDBs found with multiple relevance levels. Cannot create ranking pairs.")

        # Add a flag to limit __getitem__ logging
        self._getitem_logged_attempts = 0
        self._getitem_max_log_attempts = 5 # Reduce logging frequency

    def __len__(self):
        # Estimate length - can be refined, but less critical now
        # A simpler estimate might be sufficient, or base it on valid PDBs
        # This length is mainly used by DataLoader, exact value isn't crucial if sampling is efficient
        estimated_pairs_per_pdb = 50 # Heuristic, adjust as needed
        length = len(self.valid_pdb_ids) * estimated_pairs_per_pdb
        # print(f"RankingPairDataset: Estimated length: {length}") # Keep commented unless debugging len
        return length

    def __getitem__(self, idx):
        # Optimized sampling:
        # 1. Pick a PDB known to have multiple relevance levels.
        # 2. Pick two different relevance levels present in that PDB.
        # 3. Pick one pose from each level.
        attempts = 0
        max_attempts = 100 # Should not be needed often now, but keep as safety
        while True:
            attempts += 1
            if attempts > max_attempts:
                 # This should be rare now
                 print(f"RankingPairDataset: __getitem__ exceeded max attempts ({max_attempts}) even with optimization. Check data integrity or relevance distribution.")
                 # Fallback to old method or raise error
                 raise RuntimeError(f"__getitem__ failed after {max_attempts} attempts. Check data.")

            # 1. Pick a valid PDB ID
            pdb_id = random.choice(self.valid_pdb_ids)
            pdb_relevance_groups = self.relevance_groups[pdb_id]

            # This check should be redundant now due to pre-filtering, but keep for safety
            if len(pdb_relevance_groups) < 2:
                if self._getitem_logged_attempts < self._getitem_max_log_attempts:
                    print(f"RankingPairDataset: Warning - PDB {pdb_id} selected despite having < 2 relevance levels. Retrying...")
                    self._getitem_logged_attempts += 1
                continue

            # 2. Pick two different relevance levels
            rel1, rel2 = random.sample(list(pdb_relevance_groups.keys()), 2)

            # 3. Pick one global index (pose) from each relevance level
            idx_global1 = random.choice(pdb_relevance_groups[rel1])
            idx_global2 = random.choice(pdb_relevance_groups[rel2])

            # Assign better/worse based on relevance (higher relevance is better)
            # Ensure indices are valid before accessing relevance and features
            if idx_global1 >= len(self.relevance) or idx_global2 >= len(self.relevance) or \
               idx_global1 >= len(self.features) or idx_global2 >= len(self.features):
                print(f"RankingPairDataset: Warning - Sampled invalid index (idx1: {idx_global1}, idx2: {idx_global2}) for PDB {pdb_id}. Max index: {len(self.features)-1}. Retrying...")
                continue # Retry sampling

            if self.relevance[idx_global1] > self.relevance[idx_global2]:
                idx_better, idx_worse = idx_global1, idx_global2
            else:
                idx_better, idx_worse = idx_global2, idx_global1

            feature_better = self.features[idx_better]
            feature_worse = self.features[idx_worse]
            # Target is 1.0, indicating the first score (worse) should be > second score (better)
            target = 1.0

            # Log success only once
            if self._getitem_logged_attempts == 0:
                 print(f"RankingPairDataset: First valid pair found (using optimized method).")
                 self._getitem_logged_attempts += 1 # Mark that we've logged success once

            return (torch.tensor(feature_worse, dtype=torch.float32),
                    torch.tensor(feature_better, dtype=torch.float32),
                    torch.tensor(target, dtype=torch.float32))
# --- End RankingPairDataset ---


# Model definition using PyTorch
class ReweightModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.weights = nn.Linear(input_size, 1, bias=False)
    
    def forward(self, x):
        return self.weights(x)

def run_training(config, model_save_path=None):
    print("run_training: Starting...")
    # Prepare data
    print("run_training: Loading data...")
    # Use the correct pickle file as specified in CONFIG
    try:
        with open(config['cache_path'], 'rb') as f:
            # Load as list of dicts first, then convert to DataFrame
            data_list = pickle.load(f)
            if isinstance(data_list, pd.DataFrame):
                df = data_list # Already a DataFrame
            elif isinstance(data_list, list):
                df = pd.DataFrame(data_list)
            else:
                raise TypeError(f"Loaded data is not a list or DataFrame, but {type(data_list)}")
        print(f"run_training: Data loaded from {config['cache_path']}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Cache file not found at {config['cache_path']}")
        return None, None
    except Exception as e:
        print(f"Error loading or processing pickle file: {e}")
        return None, None

    # Ensure 'rmsd_to_crystal' column exists for relevance calculation
    if 'rmsd_to_crystal' not in df.columns:
        print("Warning: 'rmsd_to_crystal' column not found in DataFrame.")
        # Attempt to use 'rmsd_to_input' as a fallback if it exists
        if 'rmsd_to_input' in df.columns:
            print("Using 'rmsd_to_input' as fallback for relevance calculation.")
            rmsd_col_for_relevance = 'rmsd_to_input'
        else:
            print("Error: Neither 'rmsd_to_crystal' nor 'rmsd_to_input' found for relevance.")
            return None, None
    else:
        rmsd_col_for_relevance = 'rmsd_to_crystal'

    # assign finer graded relevance labels: 5:<0.5Å, 4:<1.0Å, 3:<1.5Å, 2:<2.0Å, 1:<4.0Å, 0:>=4.0Å
    print(f"run_training: Calculating relevance based on '{rmsd_col_for_relevance}'...")
    # Adjusted bins based on common practice/potential usefulness
    bins = [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf]
    labels = [4, 3, 2, 1, 0] # Higher label is better (lower RMSD)
    # Ensure the column exists before using it
    if rmsd_col_for_relevance in df.columns:
        # Use right=False to include the left edge (e.g., 0.5 is label 5, not 4)
        df['relevance'] = pd.cut(df[rmsd_col_for_relevance], bins=bins, labels=labels, right=False).astype(int)
        print("run_training: Relevance calculated.")
        # Check distribution
        print("Relevance distribution (Train+Val):")
        print(df['relevance'].value_counts(normalize=True).sort_index())
    else:
        # This case should be caught above, but double-check
        print(f"Error: Column '{rmsd_col_for_relevance}' not found for relevance calculation.")
        return None, None

    features = ['angle_constraint','atom_pair_constraint','chainbreak','coordinate_constraint',
                'dihedral_constraint','dslf_ca_dih','dslf_cs_ang','dslf_ss_dih','dslf_ss_dst',
                'fa_atr','fa_dun','fa_elec','fa_intra_rep','fa_pair','fa_rep','fa_sol',
                'hbond_bb_sc','hbond_lr_bb','hbond_sc','hbond_sr_bb','omega','p_aa_pp','pro_close','rama','ref']
    # Keep feature engineering if desired
    # df['fa_rep'] = df['fa_rep'].clip(df['fa_rep'].quantile(0.01), df['fa_rep'].quantile(0.99))
    # df['atr_rep_ratio'] = df['fa_atr']/(df['fa_rep']+1e-8)
    # df['hbond_total'] = df['hbond_bb_sc']+df['hbond_sc']
    # df['atr_rep_sum'] = df['fa_atr']+df['fa_rep']
    # for c in ['hbond_bb_sc','hbond_sc']: features remove(c)
    # features += ['atr_rep_ratio','hbond_total','atr_rep_sum']

    # Check if all feature columns exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing feature columns: {missing_features}. Filling with 0.")
        for f in missing_features:
            df[f] = 0.0

    print("run_training: Scaling features...")
    X = df[features].fillna(0).values.astype(np.float32) # Added fillna(0) for safety
    scaler = RobustScaler(); Xs = scaler.fit_transform(X)
    # Don't update DataFrame here, keep original features if needed elsewhere
    # df[features] = Xs # Avoid this unless necessary
    print("run_training: Features scaled.")

    y = df['relevance'].values # Keep relevance for dataset creation
    groups = df['pdb_id'].values
    print("run_training: Splitting data...")
    gss = GroupShuffleSplit(n_splits=1, test_size=config['test_split'], random_state=config['random_state'])
    train_idx, val_idx = next(gss.split(Xs, y, groups))
    print(f"run_training: Data split. Train size: {len(train_idx)}, Val size: {len(val_idx)}")

    # Separate features and relevance for train/val using the scaled data (Xs)
    X_train, X_val = Xs[train_idx], Xs[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    train_pids_list = groups[train_idx]
    val_pids_list = groups[val_idx]

    # Create group dictionaries mapping PDB ID to indices *within the original DataFrame*
    # These indices correspond to the rows in the original df, needed by Dataset init
    print("run_training: Creating group dictionaries (mapping PDB ID to original DataFrame indices)...")
    train_groups_orig_idx = defaultdict(list)
    for i in train_idx:
        train_groups_orig_idx[df.iloc[i]['pdb_id']].append(i)

    val_groups_orig_idx = defaultdict(list)
    for i in val_idx:
        val_groups_orig_idx[df.iloc[i]['pdb_id']].append(i)
    print("run_training: Group dictionaries created.")

    # --- Create Datasets and DataLoaders ---
    # Pass the SCALED features (Xs) and original relevance (y) and the group dicts
    # The Dataset needs the full Xs and y arrays to index into using the global indices from groups
    print("run_training: Creating train RankingPairDataset...")
    try:
        # Pass the full scaled features (Xs) and relevance (y)
        # Pass the group dict mapping PDB ID to *original* indices
        train_dataset = RankingPairDataset(Xs, y, train_groups_orig_idx)
    except ValueError as e:
        print(f"Error creating train dataset: {e}")
        return None, None
    print("run_training: Creating train DataLoader...")
    # Use num_workers > 0 if possible, but set to 0 for MPS compatibility if needed
    num_workers = 0 if torch.backends.mps.is_available() else min(4, os.cpu_count() // 2)
    print(f"Using num_workers={num_workers} for DataLoaders.")
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True if num_workers > 0 else False)
    print("run_training: Train DataLoader created.")

    # Validation dataset (for pairwise loss calculation during validation)
    print("run_training: Creating validation RankingPairDataset...")
    try:
        # Pass the full scaled features (Xs) and relevance (y)
        val_dataset_pairs = RankingPairDataset(Xs, y, val_groups_orig_idx)
    except ValueError as e:
        print(f"Error creating validation dataset: {e}")
        return None, None # Or handle differently
    print("run_training: Creating validation DataLoader...")
    val_dataloader_pairs = DataLoader(val_dataset_pairs, batch_size=config['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True if num_workers > 0 else False)
    print("run_training: Validation DataLoader created.")
    # --- End Dataloader Creation ---

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("run_training: Initializing model...")
    net = ReweightModel(X_train.shape[1]).to(device)
    print("run_training: Initializing optimizer and scheduler...")
    optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay',0))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # --- Use nn.MarginRankingLoss ---
    margin = config.get("target_margin", 0.1) # Get margin from config
    print(f"run_training: Using margin: {margin}")
    criterion = nn.MarginRankingLoss(margin=margin)
    # ---

    best_val_loss = float('inf') # Track best validation loss
    early_stop_patience = config.get('early_stop_patience', 20)
    patience_counter = 0

    # Create val_df using original validation indices for success metric calculation
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    best_metric = -1.0  # composite of success_t1 + success_t5
    best_success = (0.0, 0.0)
    save_path = model_save_path or 'weights/best_reweighter_model_pairwise.pt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure weights directory exists
    print(f"run_training: Model will be saved to {save_path}")

    # Remove margin scheduling variables if not used
    # start_margin = config.get("start_margin", 0.05)
    # target_margin = config.get("target_margin", 0.2)
    # schedule_epochs = config.get("margin_schedule_epochs", 50)

    print("run_training: Starting training loop...") # Added log before loop
    for epoch in range(config['epochs']):
        print(f"--- Epoch {epoch+1}/{config['epochs']} ---") # Added log at start of epoch
        net.train()
        total_train_loss = 0.0
        batches_processed = 0

        # --- Training loop with DataLoader ---
        print(f"Epoch {epoch+1}: Iterating through train_dataloader...") # Added log before batch loop
        # Use tqdm for progress bar if installed
        try:
            from tqdm import tqdm
            train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Train", leave=False)
        except ImportError:
            train_iterator = train_dataloader
            print("tqdm not found, install for progress bar: pip install tqdm")

        for i, batch_data in enumerate(train_iterator):
            # Add a log inside the batch loop, maybe every N batches
            # Reduce frequency or remove batch logging if too verbose
            # if i % 500 == 0: # Log every 500 batches
            #      print(f"Epoch {epoch+1}: Processing batch {i}...")
            if batch_data is None:
                print(f"Warning: Received None batch at index {i}. Skipping.")
                continue
            try:
                feature_worse, feature_better, target = batch_data
            except ValueError as e:
                print(f"Error unpacking batch {i}: {e}. Batch data: {batch_data}")
                continue # Skip malformed batch

            feature_worse, feature_better, target = feature_worse.to(device), feature_better.to(device), target.to(device)

            optimizer.zero_grad()
            score_worse = net(feature_worse)
            score_better = net(feature_better)

            # Target is 1.0, meaning score_worse should be > score_better
            loss = criterion(score_worse, score_better, target.unsqueeze(1)) # Ensure target has correct shape

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at epoch {epoch+1}, batch {i}. Skipping batch.")
                # Optionally log inputs/outputs that caused NaN
                # print("Scores worse:", score_worse)
                # print("Scores better:", score_better)
                continue # Skip optimizer step and loss accumulation for this batch

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            batches_processed += 1
            # Update tqdm description if used
            if 'tqdm' in locals():
                train_iterator.set_postfix(loss=loss.item())
        # --- End Training Loop Update ---

        avg_train_loss = total_train_loss / batches_processed if batches_processed > 0 else 0.0

        # validation
        net.eval()
        total_val_loss = 0.0
        val_batches = 0
        # --- Validation loss using pairwise sampling ---
        print(f"Epoch {epoch+1}: Calculating validation loss...")
        with torch.no_grad():
            # Limit validation steps for speed if needed, e.g., max(1, batches_processed // 5)
            # val_steps_limit = max(1, len(val_dataloader_pairs) // 10) # Example: Use 10% of val batches
            val_steps_limit = len(val_dataloader_pairs) # Use all val batches

            try:
                from tqdm import tqdm
                val_iterator = tqdm(val_dataloader_pairs, desc=f"Epoch {epoch+1} Val Loss", leave=False, total=val_steps_limit)
            except ImportError:
                val_iterator = val_dataloader_pairs

            for i, val_batch_data in enumerate(val_iterator):
                if i >= val_steps_limit: break
                if val_batch_data is None:
                    print(f"Warning: Received None validation batch at index {i}. Skipping.")
                    continue
                try:
                    feat_worse_val, feat_better_val, target_val = val_batch_data
                except ValueError as e:
                    print(f"Error unpacking validation batch {i}: {e}. Batch data: {val_batch_data}")
                    continue

                feat_worse_val, feat_better_val, target_val = feat_worse_val.to(device), feat_better_val.to(device), target_val.to(device)
                score_worse_val = net(feat_worse_val)
                score_better_val = net(feat_better_val)
                val_loss = criterion(score_worse_val, score_better_val, target_val.unsqueeze(1))

                if not torch.isnan(val_loss):
                    total_val_loss += val_loss.item()
                    val_batches += 1
                else:
                    print(f"Warning: NaN validation loss detected at epoch {epoch+1}, batch {i}.")

                if 'tqdm' in locals():
                    val_iterator.set_postfix(loss=val_loss.item())
        # --- End Validation Loss Update ---

        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else float('inf')

        # Compute validation ranking success metrics
        print(f"Epoch {epoch+1}: Calculating validation success metrics...")
        with torch.no_grad():
            # Use the scaled validation features (X_val)
            if X_val.size == 0:
                print("Warning: X_val is empty. Skipping success metric calculation.")
                success_t1, success_t5, composite = 0.0, 0.0, 0.0
            else:
                X_val_tensor = torch.from_numpy(X_val.astype(np.float32)).to(device)
                scores_val = net(X_val_tensor).cpu().numpy().flatten()
                # Ensure scores_val has the same length as val_df
                if len(scores_val) == len(val_df):
                    val_df['score'] = scores_val
                else:
                    print(f"Warning: Length mismatch between scores ({len(scores_val)}) and val_df ({len(val_df)}). Cannot assign scores.")
                    # Handle mismatch: maybe skip metric calculation for this epoch
                    success_t1, success_t5, composite = 0.0, 0.0, 0.0
                    metrics = [] # Ensure metrics is defined

        metrics = []
        # Use the same RMSD column for success metric as used for relevance
        rmsd_key_for_success = rmsd_col_for_relevance
        # Ensure the key exists in val_df and scores were assigned
        if rmsd_key_for_success not in val_df.columns:
            print(f"Error: Success metric RMSD key '{rmsd_key_for_success}' not found in validation DataFrame.")
            success_t1, success_t5, composite = 0.0, 0.0, 0.0
        elif 'score' not in val_df.columns:
            print("Warning: 'score' column not assigned to val_df. Skipping success metric calculation.")
            success_t1, success_t5, composite = 0.0, 0.0, 0.0
        else:
            rmsd_threshold = config.get('rmsd_threshold', 2.0)
            print(f"Calculating success metrics using '{rmsd_key_for_success}' and threshold {rmsd_threshold}Å") # Log which key is used
            for pid, group in val_df.groupby('pdb_id'):
                if len(group) < 1: continue
                # Sort by score - LOWER score is better for this setup
                df_sorted = group.sort_values('score', ascending=True)
                # Check if df_sorted is empty or rmsd_key exists
                if df_sorted.empty or rmsd_key_for_success not in df_sorted.columns:
                    continue
                s1 = (df_sorted.iloc[0][rmsd_key_for_success] < rmsd_threshold)
                s5 = any(df_sorted.head(min(5, len(group)))[rmsd_key_for_success] < rmsd_threshold)
                metrics.append((s1, s5))

            if metrics: # Avoid error if metrics list is empty
                metrics = np.array(metrics, dtype=float)
                success_t1 = metrics[:,0].mean() * 100
                success_t5 = metrics[:,1].mean() * 100
                composite = success_t1 + success_t5
            else:
                print("Warning: No valid groups found for success metric calculation.")
                success_t1, success_t5, composite = 0.0, 0.0, 0.0

        # Early stopping and model saving logic
        improved = composite > best_metric
        if improved:
            best_metric = composite
            best_success = (success_t1, success_t5)
            best_val_loss = avg_val_loss # Save the corresponding validation loss
            torch.save(net.state_dict(), save_path)
            print(f"Saved best model to {save_path} (Epoch {epoch+1}, s@1: {success_t1:.2f}%, s@5: {success_t5:.2f}%)." )
            patience_counter = 0
        else:
            patience_counter += 1

        current_lr = optimizer.param_groups[0]['lr']
        # Log avg_train_loss and avg_val_loss
        print(f"Epoch {epoch+1}, LR: {current_lr:.6f}, Margin: {margin:.3f}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, s@1: {success_t1:.2f}%, s@5: {success_t5:.2f}%, Patience: {patience_counter}/{early_stop_patience}")

        scheduler.step(avg_val_loss) # Step scheduler based on validation loss

        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}; best composite metric {best_metric:.2f} (s@1: {best_success[0]:.2f}%, s@5: {best_success[1]:.2f}%)")
            break

    print(f"Training done. Best val loss: {best_val_loss:.4f}, best s@1: {best_success[0]:.2f}%, s@5: {best_success[1]:.2f}%.")
    # Load the best model before returning
    if os.path.exists(save_path):
         try:
             net.load_state_dict(torch.load(save_path, map_location=device))
             print(f"Loaded best model from {save_path} for return.")
         except Exception as e:
             print(f"Error loading best model state_dict: {e}")
             # Return the current net state instead
    else:
         print(f"Warning: Best model file not found at {save_path}.")

    return best_val_loss, net # Return best loss and the loaded best net

if __name__ == '__main__':
    # Update CONFIG if needed
    # Ensure cache_path is correct
    CONFIG['cache_path'] = "docking_relax_perturb_rmsd_to_input.pkl" # Explicitly set correct path
    print(f"Using cache path: {CONFIG['cache_path']}")

    CONFIG['loss_type'] = 'margin_ranking_pairwise' # Indicate the new method (optional)
    # Clean up unused config keys if desired
    if 'hard_neg_rel_diff' in CONFIG: del CONFIG['hard_neg_rel_diff']
    if 'start_margin' in CONFIG: del CONFIG['start_margin']
    if 'margin_schedule_epochs' in CONFIG: del CONFIG['margin_schedule_epochs']

    # Add rmsd_threshold to CONFIG if not present, used in validation metrics
    if 'rmsd_threshold' not in CONFIG:
        CONFIG['rmsd_threshold'] = 2.0
        print(f"Using default RMSD threshold for success metrics: {CONFIG['rmsd_threshold']}Å")

    best_val_loss, net = run_training(CONFIG)

    if net is not None:
        print(f"Final best validation loss returned: {best_val_loss:.4f}")

        # Print learned weights
        try:
            weights = net.weights.weight.data.cpu().numpy().flatten()
            features = ['angle_constraint','atom_pair_constraint','chainbreak','coordinate_constraint',
                        'dihedral_constraint','dslf_ca_dih','dslf_cs_ang','dslf_ss_dih','dslf_ss_dst',
                        'fa_atr','fa_dun','fa_elec','fa_intra_rep','fa_pair','fa_rep','fa_sol',
                        'hbond_bb_sc','hbond_lr_bb','hbond_sc','hbond_sr_bb','omega','p_aa_pp','pro_close','rama','ref']
            # Add engineered features if used during training
            # features += ['atr_rep_ratio','hbond_total','atr_rep_sum']
            print("\nLearned scoring weights (from best model):")
            weights_dict = {name: weight for name, weight in zip(features, weights)}
            # Sort weights by absolute value for clarity
            sorted_weights = sorted(weights_dict.items(), key=lambda item: abs(item[1]), reverse=True)
            print("Weights (sorted by absolute value):")
            for name, weight in sorted_weights:
                print(f"  {name}: {weight:.4f}")
        except AttributeError:
            print("Could not retrieve weights from the model (maybe model structure changed?).")
        except Exception as e:
            print(f"An error occurred while printing weights: {e}")
    else:
        print("Training failed or was aborted, cannot print weights.")
