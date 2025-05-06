import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
from ranker import run_training, CONFIG, FCRankingNet
from ranker_3 import FCRankingNet as FCRankingNet3
from ranker_3 import CONFIG as CONFIG3
import os
import argparse

# Load test dataset
#cache_path = 'astex_test_set.pkl'
cache_path = "docking_relax_test_set.pkl"
#cache_path = "docking_relax_perturb_test_set.pkl"
#cache_path = "posebuster_relax.pkl" # 261

with open(cache_path, 'rb') as f:
    test_data = pickle.load(f)
test_df = pd.DataFrame(test_data)

# Prepare features
feature_cols = ['angle_constraint','atom_pair_constraint','chainbreak','coordinate_constraint',
                'dihedral_constraint','dslf_ca_dih','dslf_cs_ang','dslf_ss_dih','dslf_ss_dst',
                'fa_atr','fa_dun','fa_elec','fa_intra_rep','fa_pair','fa_rep','fa_sol',
                'hbond_bb_sc','hbond_lr_bb','hbond_sc','hbond_sr_bb','omega','p_aa_pp','pro_close','rama','ref']
test_df['fa_rep'] = test_df['fa_rep'].clip(lower=test_df['fa_rep'].quantile(0.01),
                                           upper=test_df['fa_rep'].quantile(0.99))

test_df['atr_rep_ratio'] = test_df['fa_atr'] / (test_df['fa_rep'] + 1e-8)
test_df['hbond_total'] = test_df['hbond_bb_sc'] + test_df['hbond_sc']
test_df['atr_rep_sum'] = test_df['fa_atr'] + test_df['fa_rep']
# # Remove raw hbonds
feature_cols.remove('hbond_bb_sc'); feature_cols.remove('hbond_sc')
feature_cols += ['atr_rep_ratio','hbond_total','atr_rep_sum']
# for c in ['hbond_bb_sc', 'hbond_sc', 'angle_constraint', 'atom_pair_constraint', 'chainbreak',
#           'coordinate_constraint', 'dihedral_constraint', 'dslf_ca_dih', 'dslf_cs_ang', 'dslf_ss_dih',
#           'dslf_ss_dst', 'hbond_sr_bb', 'omega', 'p_aa_pp', 'pro_close', 'rama', 'ref', 'fa_dun', 'hbond_lr_bb',
#           'fa_intra_rep']: feature_cols.remove(c)
#
# feature_cols += ['atr_rep_ratio', 'hbond_total', 'atr_rep_sum']
# X = test_df[feature_cols].fillna(0).values.astype(np.float32)
#
# columns_to_remove = ['run_id', 'protocol', 'angle_constraint', 'atom_pair_constraint', 'chainbreak',
#                      'coordinate_constraint', 'dihedral_constraint', 'dslf_ca_dih', 'dslf_cs_ang', 'dslf_ss_dih',
#                      'dslf_ss_dst', 'hbond_sr_bb', 'omega', 'p_aa_pp', 'pro_close', 'rama', 'ref', 'fa_dun',
#                      'hbond_lr_bb', 'fa_intra_rep']
# test_df.drop(columns=columns_to_remove, inplace=True)

X_test = test_df[feature_cols].fillna(0)
y_true = test_df['rmsd_to_input'].values
pdb_ids = test_df['pdb_id'].values

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_test)
#X_scaled = X_test.values

input_size = X_scaled.shape[1]
X_test = X_scaled

parser = argparse.ArgumentParser(description='Evaluate ranking model single or ensemble')
parser.add_argument('--no-ensemble', action='store_true', help='Use single model instead of ensemble')
parser.add_argument('--model-path', default=None, help='Path to single model checkpoint')
args = parser.parse_args()
use_ensemble = not args.no_ensemble

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
nets = []
if use_ensemble:
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        model_path = f'weights/ranker_seed_{seed}.pt'
        cfg = CONFIG.copy()
        cfg['random_state'] = seed
        if not os.path.exists(model_path):
            print(f"Training ranking model with seed {seed}...")
            _, net_i = run_training(cfg, model_save_path=model_path)
        else:
            print(f"Loading model for seed {seed}...")
            net_i = FCRankingNet(input_size, cfg['hidden_dims']).to(device)
            net_i.load_state_dict(torch.load(model_path, map_location=device))
        net_i.eval()
        nets.append(net_i)
else:
    single_path = args.model_path or 'weights/best_ranker_model_raw_energies.pt'
    print(f"Loading single model from {single_path}...")
    net = FCRankingNet(input_size, CONFIG['hidden_dims']).to(device)
    net.load_state_dict(torch.load(single_path, map_location=device))
    net.eval()
    nets.append(net)
# success_t1_pred_rmsd: 54.91%  raw_energies
# success_t5_pred_rmsd: 69.36%
# Predict scores with ensemble
with torch.no_grad():
    X_tensor = torch.from_numpy(X_scaled.astype(np.float32)).to(device)
    preds = [net_i(X_tensor).cpu().numpy().flatten() for net_i in nets]
    scores = np.mean(preds, axis=0)

# Build results
results_df = pd.DataFrame({'pdb_id': pdb_ids, 'true_rmsd': y_true, 'score': scores})
print(f"Number of unique pdb_ids processed in Test Set: {results_df['pdb_id'].nunique()}")

# Compute top-N success
RMSD_THRESHOLD = 2.0

def calculate_top_n(group):
    dfg = group.sort_values('score', ascending=False)
    print(dfg.iloc[0])
    success_t1 = (dfg.iloc[0]['true_rmsd'] < RMSD_THRESHOLD)
    #print(success_t1)
    top5 = dfg.head(5)
    success_t5 = any(top5['true_rmsd'] < RMSD_THRESHOLD)
    return success_t1, success_t5

all_metrics = []
for pdb, grp in results_df.groupby('pdb_id'):
    all_metrics.append(calculate_top_n(grp))
all_metrics = np.array(all_metrics)

success_t1_rate = np.mean(all_metrics[:,0]) * 100
success_t5_rate = np.mean(all_metrics[:,1]) * 100

print(f"success_t1_pred_rmsd: {success_t1_rate:.2f}%")
print(f"success_t5_pred_rmsd: {success_t5_rate:.2f}%")

# Assert performance
# assert success_t1_rate > 57.0, f"Expected success_t1_pred_rmsd > 57%, got {success_t1_rate:.2f}%"
# assert success_t5_rate >= 70.0, f"Expected success_t5_pred_rmsd >= 70%, got {success_t5_rate:.2f}%"