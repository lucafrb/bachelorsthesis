import json
import os
import pandas as pd
import numpy as np
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from sklearn.metrics import *
import pickle
from datetime import datetime
from tinygrad.tensor import Tensor
from tinygrad import nn
from tinygrad import TinyJit
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt


class Colors:
    HEADER = '\033[95m'
    INFO = '\033[94m'  # Blue
    SUCCESS = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    ERROR = '\033[91m'  # Red
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def log(message, level='INFO'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Corrected timestamp format

    if level == 'INFO':
        color = Colors.INFO
        level_str = 'INFO'
    elif level == 'SUCCESS':
        color = Colors.SUCCESS
        level_str = 'SUCCESS'
    elif level == 'WARNING':
        color = Colors.WARNING
        level_str = 'WARNING'
    elif level == 'ERROR':
        color = Colors.ERROR
        level_str = 'ERROR'
    else:
        color = Colors.INFO  # Default to INFO
        level_str = level

    print(f"{color}[{timestamp}] {level_str:<8}{Colors.END} {message}")


# 363 test pdbs
# 16 were not successfully docked
# 347 were successfully docked
# 80% of the test set is below 2.0 RMSD

# rosetta energy func: 53.7 and 65.8 respectively

# 7 layer + bias = batch 1000, identity at last layer
# success_t1_pred_rmsd: 41.33%
# success_t5_pred_rmsd: 55.20%


def load_test_dataset():
    #cache_path = 'astex_test_set.pkl'
    cache_path = "posebuster_relax.pkl"  # 261
    # cache_path = 'docking_relax_perturb_rmsd_to_input.pkl'
    with open(cache_path, 'rb') as f:
        log(f"Loading cached docking_data.", "INFO")
        docking_data = pickle.load(f)
    log(f"Loaded {len(docking_data)} records.", "SUCCESS")
    return pd.DataFrame(docking_data)


CONFIG = {
    "hidden_dims": {
        "hidden_dim1": 1024,
        "hidden_dim2": 512,
        "hidden_dim3": 256,
        "hidden_dim4": 128
    }
}


class Net:
    def __init__(self, input_size, hidden_dims):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_dims["hidden_dim1"], bias=True)
        self.bn1 = nn.BatchNorm(hidden_dims["hidden_dim1"], track_running_stats=True, eps=1e-5, momentum=0.1)
        self.l2 = nn.Linear(hidden_dims["hidden_dim1"], hidden_dims["hidden_dim2"], bias=True)
        self.bn2 = nn.BatchNorm(hidden_dims["hidden_dim2"], track_running_stats=True, eps=1e-5, momentum=0.1)
        self.l3 = nn.Linear(hidden_dims["hidden_dim2"], hidden_dims["hidden_dim3"], bias=True)
        self.bn3 = nn.BatchNorm(hidden_dims["hidden_dim3"], track_running_stats=True, eps=1e-5, momentum=0.1)
        self.l4 = nn.Linear(hidden_dims["hidden_dim3"], hidden_dims["hidden_dim4"], bias=True)
        self.bn4 = nn.BatchNorm(hidden_dims["hidden_dim4"], track_running_stats=True, eps=1e-5, momentum=0.1)
        self.l5 = nn.Linear(hidden_dims["hidden_dim4"], 1, bias=True)

    def __call__(self, x):
        x = self.l1(x)
        x = self.bn1(x).relu()
        #x = x.dropout(0.15)
        x = self.l2(x)
        x = self.bn2(x).relu()
        #x = x.dropout(0.15)
        x = self.l3(x)
        x = self.bn3(x).relu()
        #x = x.dropout(0.10)
        x = self.l4(x)
        x = self.bn4(x).relu()
        #x = x.dropout(0.05)
        x = self.l5(x)
        return x


@TinyJit
def test_jit(x: Tensor):
    return net(x).realize()


test_df = load_test_dataset()

feature_cols = ['angle_constraint', 'atom_pair_constraint', 'chainbreak', 'coordinate_constraint',
                'dihedral_constraint', 'dslf_ca_dih', 'dslf_cs_ang', 'dslf_ss_dih', 'dslf_ss_dst',
                'fa_atr', 'fa_dun', 'fa_elec', 'fa_intra_rep', 'fa_pair', 'fa_rep',
                'fa_sol', 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb', 'omega',
                'p_aa_pp', 'pro_close', 'rama', 'ref']

test_df['fa_rep'] = test_df['fa_rep'].clip(lower=test_df['fa_rep'].quantile(0.01),
                                 upper=test_df['fa_rep'].quantile(0.99))

test_df['atr_rep_ratio'] = test_df['fa_atr'] / (test_df['fa_rep'] + 1e-8)  # Attractive/repulsive ratio
test_df['hbond_total'] = test_df['hbond_bb_sc'] + test_df['hbond_sc']  # Total hydrogen bonding
test_df['atr_rep_sum'] = test_df['fa_atr'] + test_df['fa_rep']
feature_cols.remove('hbond_bb_sc')
feature_cols.remove('hbond_sc')
feature_cols.extend(['atr_rep_ratio', 'atr_rep_sum', 'hbond_total'])
X_test = test_df[feature_cols]
#y_test_log = np.log1p(test_df["rmsd_to_input"])
y_test = test_df["rmsd_to_input"]


from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.0)
# try:
#     X_filtered = selector.fit_transform(X_test)
#     kept_indices = selector.get_support(indices=True)
#     if len(kept_indices) < X_test.shape[1]:
#         feature_cols_kept = [feature_cols[i] for i in kept_indices]
#         X = pd.DataFrame(X_filtered, columns=feature_cols_kept, index=X_test.index)
#         log(f"Removed zero-variance features. Kept {len(feature_cols_kept)} features.", "INFO")
#     else:
#         log("No zero-variance features found.", "INFO")
#         feature_cols_kept = feature_cols
# except ValueError as e:
#     log(f"Error during VarianceThreshold: {e}. Skipping feature selection.", "ERROR")
#     feature_cols_kept = feature_cols

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_test)
input_size = X_scaled.shape[1]
X_test = X_scaled

net = Net(input_size, CONFIG["hidden_dims"])


log("Loading best model for evaluation.", "INFO")
best_state_dict = safe_load("weights/best_model.safetensors")
load_state_dict(net, best_state_dict)

log("Predicting on test set.", "WARNING")
Tensor.training = False
X_test_tensor = Tensor(np.array(X_test, dtype=np.float32), requires_grad=False)
test_predictions = test_jit(X_test_tensor)
y_pred = test_predictions.numpy().flatten()

# Evaluate
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
log(f"RMSE: {rmse:.3f}", "SUCCESS")
log(f"R2: {r2:.3f}", "SUCCESS")

results_df = pd.DataFrame({
    'pdb_id': test_df.loc[test_df.index, 'pdb_id'],  # Get pdb_id using original index
    'true_rmsd': y_test,
    'predicted_rmsd': y_pred
})
print(results_df)

RMSD_THRESHOLD = 2.0

mode = "regression"
def calculate_top_n_success(group_df):
    metrics = {}
    n_poses = len(group_df)

    if mode == "regression":
        group_df = group_df.sort_values('predicted_rmsd', ascending=True)
        # print(group_df.iloc[0]["pdb_id"])
        # print(f"predicted rmsd:" + str(group_df.iloc[0]["predicted_rmsd"]))
        # print(f"true rmsd:" + str(group_df.iloc[0]['true_rmsd']))

        # Top-1 Predicted RMSD Success
        if n_poses >= 1:
            metrics['success_t1_pred_rmsd'] = group_df.iloc[0]['true_rmsd'] < RMSD_THRESHOLD
        else:
            metrics['success_t1_pred_rmsd'] = False

        # Top-5 Predicted RMSD Success
        if n_poses >= 1:
            top_5_indices = group_df.head(min(5, n_poses)).index
            metrics['success_t5_pred_rmsd'] = (group_df.loc[top_5_indices, 'true_rmsd'] < RMSD_THRESHOLD).any()
        else:
            metrics['success_t5_pred_rmsd'] = False

    elif mode == "classifier":
        group_df = group_df.sort_values('classifier_score', ascending=False)

        # Top-1 Classifier Success
        if n_poses >= 1:
            metrics['success_t1_classifier'] = group_df.iloc[0]['true_rmsd'] < RMSD_THRESHOLD
        else:
            metrics['success_t1_classifier'] = False

        # Top-5 Classifier Success
        if n_poses >= 1:
            top_5_indices = group_df.head(min(5, n_poses)).index
            metrics['success_t5_classifier'] = (group_df.loc[top_5_indices, 'true_rmsd'] < RMSD_THRESHOLD).any()
        else:
            metrics['success_t5_classifier'] = False

    if 'ranker_score' in group_df.columns:
        # Assuming lower score means better rank
        group_df = group_df.sort_values('ranker_score', ascending=True)

        # Top-1 Ranker Success
        if n_poses >= 1:
            metrics['success_t1_ranker'] = group_df.iloc[0]['true_rmsd'] < RMSD_THRESHOLD
        else:
            metrics['success_t1_ranker'] = False

        # Top-5 Ranker Success
        if n_poses >= 1:
            top_5_indices = group_df.head(min(5, n_poses)).index
            metrics['success_t5_ranker'] = (group_df.loc[top_5_indices, 'true_rmsd'] < RMSD_THRESHOLD).any()
        else:
            metrics['success_t5_ranker'] = False

    return metrics


all_complex_metrics = []
grouped_results = results_df.groupby('pdb_id')

log(f"Calculating Top-N metrics for {len(grouped_results)} complexes...", "INFO")

for name, group in grouped_results:
    complex_metrics = calculate_top_n_success(group)
    complex_metrics['pdb_id'] = name
    all_complex_metrics.append(complex_metrics)

metrics_summary_df = pd.DataFrame(all_complex_metrics)

overall_success_rates = {}
metric_cols = [col for col in metrics_summary_df.columns if col.startswith('success_')]

for col in metric_cols:
    overall_success_rates[col] = metrics_summary_df[col].mean() * 100

log("Overall Top-N Success Rates (%):", "SUCCESS")
for metric, rate in overall_success_rates.items():
    print(f"  {metric}: {rate:.2f}%")