import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import torch
from reweighter import ReweightModel, CONFIG, run_training
import os
import argparse

# Load test dataset
#cache_path = 'astex_test_set.pkl'
#cache_path = "docking_relax_test_set.pkl"
#cache_path = "docking_relax_perturb_test_set.pkl"
cache_path = "posebuster_relax.pkl"  # 261

with open(cache_path, 'rb') as f:
    test_data = pickle.load(f)
test_df = pd.DataFrame(test_data)

# Prepare features
feature_cols = ['angle_constraint','atom_pair_constraint','chainbreak','coordinate_constraint',
                'dihedral_constraint','dslf_ca_dih','dslf_cs_ang','dslf_ss_dih','dslf_ss_dst',
                'fa_atr','fa_dun','fa_elec','fa_intra_rep','fa_pair','fa_rep','fa_sol',
                'hbond_bb_sc','hbond_lr_bb','hbond_sc','hbond_sr_bb','omega','p_aa_pp','pro_close','rama','ref']

scaler = RobustScaler()
Xs = scaler.fit_transform(test_df[feature_cols].values.astype(np.float32))

X_test = Xs
y_true = test_df['rmsd_to_input'].values
pdb_ids = test_df['pdb_id'].values

input_size = X_test.shape[1]

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
nets = []
single_path = 'weights/best_reweighter_model.pt'
print(f"Loading single model from {single_path}...")
net = ReweightModel(input_size).to(device)
net.load_state_dict(torch.load(single_path, map_location=device))
net.eval()
nets.append(net)

with torch.no_grad():
    X_tensor = torch.from_numpy(X_test).to(device)
    preds = [net_i(X_tensor).cpu().numpy().flatten() for net_i in nets]
    scores = np.mean(preds, axis=0)

results_df = pd.DataFrame({'pdb_id': pdb_ids, 'true_rmsd': y_true, 'score': scores})
print(f"Number of unique pdb_ids processed in Test Set: {results_df['pdb_id'].nunique()}")

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