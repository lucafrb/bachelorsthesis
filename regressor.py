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
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, RobustScaler

from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad.tensor import Tensor
from tinygrad import nn, TinyJit
from tinygrad.helpers import GlobalCounters

class Colors:
    HEADER = '\033[95m'
    INFO = '\033[94m'    # Blue
    SUCCESS = '\033[92m' # Green
    WARNING = '\033[93m' # Yellow
    ERROR = '\033[91m'   # Red
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def log(message, level='INFO'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    if level == 'INFO':
        color = Colors.INFO
    elif level == 'SUCCESS':
        color = Colors.SUCCESS
    elif level == 'WARNING':
        color = Colors.WARNING
    elif level == 'ERROR':
        color = Colors.ERROR
    else:
        color = Colors.INFO
    print(f"{color}[{timestamp}] {level:<8}{Colors.END} {message}")


CONFIG = {
    "cache_path": "docking_relax_perturb_delta.pkl",
    "sample_size": "inf",
    "test_split": 0.1,
    "batch_size": 256,
    "epochs": 100,
    "patience_threshold": 10,
    "loss_choice": "mse",
    "lr_schedule": {
        0: 0.0001,
        50: 0.00005,
        100: 0.00001
    },
    "random_state": 42,
    "hidden_dims": {
        "hidden_dim1": 1024,
        "hidden_dim2": 512,
        "hidden_dim3": 256,
        "hidden_dim4": 128,
        "hidden_dim5": 64
    },
    "hyperparam_tuning_enabled": False,
    "tuning_epochs": 50,
    "tuning_patience_threshold": 5,
    "tuning_search_space": {}
}



def load_dataset(cache_path):
    with open(cache_path, 'rb') as f:
        log("Loading cached docking data.", "INFO")
        docking_data = pickle.load(f)
    log(f"Loaded {len(docking_data)} records.", "SUCCESS")
    return pd.DataFrame(docking_data)



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
        x = x.dropout(0.1)
        x = self.l2(x)
        x = self.bn2(x).relu()
        x = x.dropout(0.1)
        x = self.l3(x)
        x = self.bn3(x).relu()
        x = x.dropout(0.1)
        x = self.l4(x)
        x = self.bn4(x).relu()
        x = x.dropout(0.1)
        x = self.l5(x)
        return x





def weighted_mse(y_pred, y_true, epsilon=0.1):
    weights = 1 / (y_true + epsilon)
    return (weights * (y_pred - y_true) ** 2).mean()


def huber_loss(preds: Tensor, targets: Tensor, delta: float = 1.0) -> Tensor:
    residual = preds - targets
    abs_residual = residual.abs()
    mask = abs_residual <= delta
    mse_loss = 0.5 * (residual ** 2)
    mae_loss = delta * (abs_residual - 0.5 * delta)
    loss = mask * mse_loss + (~mask) * mae_loss
    return loss.mean()


def calculate_loss(y_pred: Tensor, y_batch: Tensor, loss_choice: str):
    if loss_choice == 'mse':
        return ((y_pred - y_batch) ** 2).mean()
    elif loss_choice == 'mae':
        return (y_pred - y_batch).abs().mean()
    elif loss_choice == 'huber':
        return huber_loss(y_pred, y_batch)
    elif loss_choice == 'weighted_mse':
        return weighted_mse(y_pred, y_batch)
    else:
        raise ValueError(f"Unknown loss_choice: {loss_choice}")



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



def plot_loss_curve(train_losses, val_losses, num_epochs, loss_choice):
    epochs_axis = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, train_losses, label='Training Loss', linewidth=2)
    plt.plot(epochs_axis, val_losses, label='Validation Loss', linewidth=2)
    plt.title(f'Training and Validation Loss Curve ({loss_choice.upper()})', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_filename = f'loss_epoch_viz/loss_curve_{loss_choice}.png'
    if not os.path.exists("loss_epoch_viz"):
        os.makedirs("loss_epoch_viz")
    plt.savefig(plot_filename)
    log(f"Loss curve plot saved as '{plot_filename}'", "INFO")
    plt.show()



def update_learning_rate(epoch, opt, lr_schedule, initial_lr):
    new_lr = initial_lr
    for milestone in sorted(lr_schedule.keys()):
        if epoch >= milestone:
            new_lr = lr_schedule[milestone]
        else:
            break

    current_lr_val = opt.lr.numpy().item()
    if abs(current_lr_val - new_lr) > 1e-9:
        log(f"Epoch {epoch}: Updating LR from {current_lr_val:.1E} to {new_lr:.1E}", "WARNING")
        opt.lr.assign(Tensor([new_lr], dtype=opt.lr.dtype, requires_grad=False)).realize()



def run_training(config, tuning_mode=False, model_save_path=None):
    epochs_to_run = config.get("tuning_epochs", config["epochs"]) if tuning_mode else config["epochs"]
    patience_threshold = config.get("tuning_patience_threshold", config["patience_threshold"]) if tuning_mode else config["patience_threshold"]

    np.random.seed(config["random_state"])

    df = load_dataset(config["cache_path"])
    # if config["sample_size"] != "inf":
    #     log(f"Sampling {config['sample_size']} records out of {len(df)}", "INFO")
    #     df = df.sample(n=config["sample_size"], random_state=config["random_state"])
    # log("Extracted features", "SUCCESS")
    #
    # MAX_REASONABLE_RMSD = 20.0  # Adjust as needed
    # initial_count = len(df)
    # df = df[df['rmsd_to_input'] < MAX_REASONABLE_RMSD].copy()  # Filter out extreme outliers
    # log(f"Filtered out extreme RMSD outliers > {MAX_REASONABLE_RMSD}Å. Kept {len(df)} / {initial_count} records.", "WARNING")

    y = df["rmsd_to_input"]
    log("Applied log1p transformation to target variable 'y'.", "WARNING")

    # Define features and target
    feature_cols = ['angle_constraint', 'atom_pair_constraint', 'chainbreak', 'coordinate_constraint',
                    'dihedral_constraint', 'dslf_ca_dih', 'dslf_cs_ang', 'dslf_ss_dih', 'dslf_ss_dst',
                    'fa_atr', 'fa_dun', 'fa_elec', 'fa_intra_rep', 'fa_pair', 'fa_rep',
                    'fa_sol', 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb',
                    'omega', 'p_aa_pp', 'pro_close', 'rama', 'ref']

    df['fa_rep'] = df['fa_rep'].clip(lower=df['fa_rep'].quantile(0.01),
                                     upper=df['fa_rep'].quantile(0.99))

    df['atr_rep_ratio'] = df['fa_atr'] / (df['fa_rep'] + 1e-8)  # Attractive/repulsive ratio
    df['hbond_total'] = df['hbond_bb_sc'] + df['hbond_sc']  # Total hydrogen bonding
    df['atr_rep_sum'] = df['fa_atr'] + df['fa_rep']
    feature_cols.extend(['atr_rep_ratio', 'atr_rep_sum', 'hbond_total'])
    feature_cols.remove('hbond_bb_sc')
    feature_cols.remove('hbond_sc')
    X = df[feature_cols]

    # Remove zero-variance features and scale the remaining ones
    # selector = VarianceThreshold(threshold=0.0)
    # X_filtered = selector.fit_transform(X)
    # kept_indices = selector.get_support(indices=True)
    # feature_cols_kept = [feature_cols[i] for i in kept_indices]
    # X = pd.DataFrame(X_filtered, columns=feature_cols_kept, index=X.index)
    # log(f"Removed zero-variance features. Kept {len(feature_cols_kept)} features.", "INFO")

    scaler = RobustScaler()

    scaler.fit(X)
    X_scaled = scaler.transform(X)
    input_size = X_scaled.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=config["test_split"], random_state=config["random_state"]
    )
    log(f"Data split: Train: {len(X_train)}, Val: {len(X_val)}", "SUCCESS")


    net = Net(input_size, config["hidden_dims"])
    initial_lr = config["lr_schedule"][0]  # initial learning rate from schedule
    opt = nn.optim.AdamW(params=nn.state.get_parameters(net), lr=initial_lr)

    X_train_np = np.array(X_train, dtype=np.float32)


    y_train_np = np.array(y_train, dtype=np.float32).reshape(-1, 1)
    X_val_tensor = Tensor(np.array(X_val, dtype=np.float32), requires_grad=False)
    y_val_tensor = Tensor(np.array(y_val, dtype=np.float32).reshape(-1, 1), requires_grad=False)

    n_samples = len(X_train_np)
    indices = np.arange(n_samples)
    batch_size = config["batch_size"]
    loss_choice = config["loss_choice"]
    lr_schedule = config["lr_schedule"]

    best_val_loss = float('inf')
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []
    epochs_run = 0

    log(f"Starting training with {loss_choice.upper()} loss.", "WARNING")

    for epoch in range(epochs_to_run):
        epochs_run += 1
        # Update learning rate as per schedule
        update_learning_rate(epoch, opt, lr_schedule, initial_lr)
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
        log(f"Epoch {epoch:03d}, LR: {current_lr:.1E}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss_np:.4f}", "INFO")

        # Save best model (if improvement observed)
        if val_loss_np < best_val_loss:
            best_val_loss = val_loss_np
            if not tuning_mode:
                if not os.path.exists("weights"):
                    os.makedirs("weights")
                safe_save(get_state_dict(net), "weights/best_model.safetensors")
            log(f"Validation improved to {best_val_loss:.4f}", "SUCCESS")
            patience_counter = 0
        else:
            patience_counter += 1
            log(f"No improvement in validation loss. Patience: {patience_counter}/{patience_threshold}", "WARNING")
            if patience_counter >= patience_threshold:
                log(f"Early stopping triggered at epoch {epoch+1}.", "ERROR")
                break

    log(f"Training completed in {epochs_run} epochs.", "SUCCESS")

    # Plot loss curves if not in tuning mode
    if not tuning_mode and train_loss_history:
        plot_loss_curve(train_loss_history, val_loss_history, epochs_run, loss_choice)

    # Save final model if a save path is provided
    if model_save_path:
        if not os.path.exists("weights"):
            os.makedirs("weights")
        safe_save(get_state_dict(net), model_save_path)
        log(f"Model saved to {model_save_path}", "INFO")
        
    return best_val_loss, net



def hyperparameter_tuner(config):

    search_space = config.get("tuning_search_space", {})
    keys = list(search_space.keys())
    best_config = None
    best_loss = float('inf')

    for values in itertools.product(*(search_space[key] for key in keys)):
        trial_config = copy.deepcopy(config)
        trial_name = []
        # Update tunable hyperparameters (do not include loss_choice)
        for key, value in zip(keys, values):
            if key == "initial_lr":
                trial_config["lr_schedule"][0] = value
                trial_name.append(f"initial_lr={value}")
            elif key.startswith("lr_"):
                # e.g., key "lr_20" corresponds to milestone 20
                milestone = int(key.split("_")[1])
                trial_config["lr_schedule"][milestone] = value
                trial_name.append(f"lr_{milestone}={value}")
            elif key == "batch_size":
                trial_config["batch_size"] = value
                trial_name.append(f"batch_size={value}")
            else:
                trial_config[key] = value
                trial_name.append(f"{key}={value}")

        trial_name_str = ", ".join(trial_name)
        sanitized_name = trial_name_str.replace(" ", "").replace(",", "_").replace("=", "-")
        model_save_path = os.path.join("weights", f"trial_{sanitized_name}.safetensors")

        log(f"Starting hyperparameter tuning trial with {trial_name_str}", "INFO")
        val_loss, _ = run_training(trial_config, tuning_mode=True, model_save_path=model_save_path)
        log(f"Trial {trial_name_str} completed with validation loss {val_loss:.4f}", "INFO")

        if val_loss < best_loss:
            best_loss = val_loss
            best_config = trial_config
            log(f"New best configuration found: {trial_name_str} with loss {val_loss:.4f}", "SUCCESS")

    return best_config, best_loss



def main():
    if CONFIG.get("hyperparam_tuning_enabled", False):
        best_config, best_loss = hyperparameter_tuner(CONFIG)
        log(f"Hyperparameter tuning complete. Best validation loss: {best_loss:.4f}", "SUCCESS")
        log(f"Best configuration: initial_lr={best_config['lr_schedule'][0]}, hidden_dim1={best_config['hidden_dims']['hidden_dim1']}, batch_size={best_config['batch_size']}", "SUCCESS")
        CONFIG.update(best_config)
        run_training(CONFIG, tuning_mode=False, model_save_path="weights/final_model.safetensors")
    else:
        run_training(CONFIG, tuning_mode=False, model_save_path="weights/final_model.safetensors")

if __name__ == "__main__":
    main()
