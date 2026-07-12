"""
Unified script for training and evaluating GRU and LSTM models
for time-series prediction of respiratory cases.

It performs grid search over:
  - features (x1, x2, ...)
  - direction (up/down)
  - time steps (1,3,5,7,14)
  - hyperparameters (hidden_size, num_layers, lr, dropout)

Early stopping with validation split (80/20 from training period) is used.
All results (predictions and metrics) are saved under ./results/RNN/
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---------------------------- configuration ----------------------------
# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data paths (adjust if needed)
DATA_PATH = './data/data.xlsx'
HOLIDAY_PATH = './data/holiday.xlsx'

# Models to run
MODEL_TYPES = ['GRU', 'LSTM']

# Feature and direction settings
X_FEATURES = ['x1', 'x2', 'x5', 'x10', 'x20', 'x26', 'x36', 'x43']
DIRECTIONS = ['up', 'down']
TIME_STEPS = [1, 3, 5, 7]

# Hyperparameter search spaces
GRID_GRU = {
    'hidden_size': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'learning_rate': [0.0005, 0.0001, 0.005, 0.001, 0.05, 0.01],
    'dropout': [0.2, 0.25, 0.3, 0.35, 0.4]   # only effective when num_layers > 1
    
}
GRID_LSTM = {
    'hidden_size': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'learning_rate': [0.0005, 0.0001, 0.005, 0.001, 0.05, 0.01],
    'dropout': [0.2, 0.25, 0.3, 0.35, 0.4]
}

# Training parameters (fixed)
BATCH_SIZE = 4
NUM_EPOCHS = 300
PATIENCE = 100
SPLIT_DATE = pd.Timestamp('2024-09-01')   # train / test split

# Output root
OUTPUT_ROOT = './results/DL'
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ---------------------------- data loading ----------------------------
def load_and_preprocess(data_path, holiday_path):
    data = pd.read_excel(data_path)
    data['rq'] = pd.to_datetime(data['rq'])

    holiday = pd.read_excel(holiday_path)
    holiday['rq'] = pd.to_datetime(holiday['date'])
    holidays = holiday[['rq', 'is_holiday']]

    # 7-day moving average (skip first column 'rq')
    smodata = data.copy()
    smodata.iloc[:, 2:] = data.iloc[:, 2:].rolling(window=7, min_periods=1).mean()

    selected = ['rq'] + X_FEATURES + DIRECTIONS + ['sum']
    df = smodata[selected]
    df = pd.merge(df, holidays, on='rq', how='left')
    return df

# ---------------------------- model definitions ----------------------------
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.0):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# ---------------------------- sequence creation ----------------------------
def create_sequences(features, targets, time_step):
    xs, ys = [], []
    for i in range(len(features) - time_step):
        xs.append(features[i:i+time_step])
        ys.append(targets[i+time_step])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

# ---------------------------- training function with early stopping ---------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience):
    best_val_loss = float('inf')
    no_improve = 0
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            best_state = model.state_dict().copy()
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        if (epoch+1) % 100 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}")

    model.load_state_dict(best_state)
    return model, train_losses, val_losses

# ---------------------------- evaluation and saving -----------------------
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return {'R2': r2, 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def save_predictions_and_metrics(df, model_name, direction, feature, time_step, params,
                                  train_dates, y_train_true, y_train_pred,
                                  test_dates, y_test_true, y_test_pred, metrics):
    """Save both train and test predictions."""
    base = os.path.join(OUTPUT_ROOT, model_name, direction, feature, f'ts_{time_step}')
    os.makedirs(base, exist_ok=True)
    train_df = pd.DataFrame({
        'date': train_dates,
        'actual': y_train_true,
        'predicted': y_train_pred,
        'set': 'train'
    })
    test_df = pd.DataFrame({
        'date': test_dates,
        'actual': y_test_true,
        'predicted': y_test_pred,
        'set': 'test'
    })
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined.to_csv(os.path.join(base, 'predictions.csv'), index=False)

    # Save metrics (append to global summary)
    summary_path = os.path.join(OUTPUT_ROOT, model_name, f'{model_name}_summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    row = {
        'direction': direction,
        'feature': feature,
        'time_step': time_step,
        **params,
        **metrics
    }
    new_row = pd.DataFrame([row])
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(summary_path, index=False)

    # Also save a plot (optional)
    try:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
    # plot training and test
        train_mask = combined['set'] == 'train'
        test_mask = combined['set'] == 'test'
        if train_mask.any():
            ax.plot(combined.loc[train_mask, 'date'], combined.loc[train_mask, 'actual'],
                    label='Train actual', color='#0A2472', linewidth=1.0)
            ax.plot(combined.loc[train_mask, 'date'], combined.loc[train_mask, 'predicted'],
                    label='Train predicted', color='#1C6DD0', linestyle='--', linewidth=1.0)
        if test_mask.any():
            ax.plot(combined.loc[test_mask, 'date'], combined.loc[test_mask, 'actual'],
                    label='Test actual', color='#5D8BF4', linewidth=1.2)
            ax.plot(combined.loc[test_mask, 'date'], combined.loc[test_mask, 'predicted'],
                    label='Test predicted', color='#9AC5F4', linestyle='--', linewidth=1.2)
        ax.set_xlabel('Date')
        ax.set_ylabel('ARIDs cases')
        ax.legend()
        ax.set_title(f'{model_name} {feature} {direction} ts={time_step}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(base, 'prediction_plot.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"    Warning: Plot generation failed: {e}")

# ---------------------------- main loop ---------------------------------
def main():
    df = load_and_preprocess(DATA_PATH, HOLIDAY_PATH)
    df['rq'] = pd.to_datetime(df['rq'])

    # For each model type
    for model_name in MODEL_TYPES:
        print(f"\n===== Training {model_name} =====")
        # Select grid
        if model_name == 'GRU':
            grid = GRID_GRU
            model_class = GRUModel
        else:
            grid = GRID_LSTM
            model_class = LSTMModel

        # Build parameter combinations
        import itertools
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        param_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

        # For each direction
        for direction in DIRECTIONS:
            print(f"  Direction: {direction}")
            for feature in X_FEATURES:
                print(f"    Feature: {feature}")
                # Prepare data: X = [feature, direction, is_holiday]
                X_df = df[[feature, direction, 'is_holiday']]
                y_df = df['sum']

                train_mask = df['rq'] < SPLIT_DATE
                test_mask = df['rq'] >= SPLIT_DATE

                X_train_raw = X_df[train_mask].values.astype(np.float32)
                y_train_raw = y_df[train_mask].values.astype(np.float32)
                X_test_raw = X_df[test_mask].values.astype(np.float32)
                y_test_raw = y_df[test_mask].values.astype(np.float32)

                # Normalize features and target (use training stats)
                feature_scaler = MinMaxScaler()
                X_train_scaled = feature_scaler.fit_transform(X_train_raw)
                X_test_scaled = feature_scaler.transform(X_test_raw)

                target_scaler = MinMaxScaler()
                y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1,1)).flatten()
                y_test_scaled = target_scaler.transform(y_test_raw.reshape(-1,1)).flatten()

                all_dates = df['rq']
                train_dates_original = all_dates[train_mask].values
                test_dates_original = all_dates[test_mask].values

                # For each time step
                for time_step in TIME_STEPS:
                    print(f"      Time step: {time_step}")

                    # Create sequences from the scaled training set (for later validation split)
                    X_seq, y_seq = create_sequences(X_train_scaled, y_train_scaled, time_step)
                    # Split sequences into train/val (80/20) preserving temporal order
                    n_seq = len(X_seq)
                    val_size = int(0.2 * n_seq)
                    if val_size < 1:
                        val_size = 1
                    train_X_seq = X_seq[:-val_size]
                    train_y_seq = y_seq[:-val_size]
                    val_X_seq = X_seq[-val_size:]
                    val_y_seq = y_seq[-val_size:]

                    # Test sequences
                    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, time_step)

                    train_dates = train_dates_original[time_step:]
                    test_dates = test_dates_original[time_step:]
                    if len(train_dates) > len(train_y_seq):
                        train_dates = train_dates[:len(train_y_seq)]
                    elif len(train_dates) < len(train_y_seq):
                        train_y_seq = train_y_seq[:len(train_dates)]
                        train_X_seq = train_X_seq[:len(train_dates)]
                    if len(test_dates) > len(y_test_seq):
                        test_dates = test_dates[:len(y_test_seq)]
                    elif len(test_dates) < len(y_test_seq):
                        y_test_seq = y_test_seq[:len(test_dates)]
                        X_test_seq = X_test_seq[:len(test_dates)]

                    # Convert to tensors and loaders
                    train_X_t = torch.FloatTensor(train_X_seq)
                    train_y_t = torch.FloatTensor(train_y_seq).view(-1,1)
                    val_X_t = torch.FloatTensor(val_X_seq)
                    val_y_t = torch.FloatTensor(val_y_seq).view(-1,1)
                    test_X_t = torch.FloatTensor(X_test_seq)
                    test_y_t = torch.FloatTensor(y_test_seq).view(-1,1)

                    train_ds = TensorDataset(train_X_t, train_y_t)
                    val_ds = TensorDataset(val_X_t, val_y_t)
                    test_ds = TensorDataset(test_X_t, test_y_t)

                    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
                    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
                    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

                    input_size = train_X_t.shape[2]

                    for params in param_combos:
                        hidden_size = params['hidden_size']
                        num_layers = params['num_layers']
                        lr = params['learning_rate']
                        dropout = params['dropout']

                        print(f"        Params: {params}")

                        model = model_class(input_size, hidden_size, num_layers,
                                            output_size=1, dropout=dropout).to(device)
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                        # Train with early stopping
                        model, _, _ = train_model(model, train_loader, val_loader,
                                                  criterion, optimizer, NUM_EPOCHS, PATIENCE)

                        # Predict on test set
                        model.eval()
                        train_preds = []
                        with torch.no_grad():
                            for xb, _ in train_loader:
                                xb = xb.to(device)
                                out = model(xb)
                                train_preds.append(out.cpu().numpy())
                        train_preds = np.concatenate(train_preds, axis=0)
                        train_pred_actual = target_scaler.inverse_transform(train_preds).flatten()
                        y_train_true = target_scaler.inverse_transform(train_y_t.numpy()).flatten()

                        min_len_train = min(len(train_dates), len(y_train_true), len(train_pred_actual))
                        if min_len_train < len(train_dates):
                            train_dates = train_dates[:min_len_train]
                            y_train_true = y_train_true[:min_len_train]
                            train_pred_actual = train_pred_actual[:min_len_train]

                        test_preds = []
                        with torch.no_grad():
                            for xb, _ in test_loader:
                                xb = xb.to(device)
                                out = model(xb)
                                test_preds.append(out.cpu().numpy())
                        test_preds = np.concatenate(test_preds, axis=0)
                        test_pred_actual = target_scaler.inverse_transform(test_preds).flatten()
                        y_test_true = target_scaler.inverse_transform(test_y_t.numpy()).flatten()
            
                        min_len_test = min(len(test_dates), len(y_test_true), len(test_pred_actual))
                        if min_len_test < len(test_dates):
                            test_dates = test_dates[:min_len_test]
                            y_test_true = y_test_true[:min_len_test]
                            test_pred_actual = test_pred_actual[:min_len_test]

                
                        train_pred_actual = np.clip(train_pred_actual, 0, None)
                        test_pred_actual = np.clip(test_pred_actual, 0, None)

                        metrics = evaluate(y_test_true, test_pred_actual)

                        # save predictions and metrics
                        save_predictions_and_metrics(
                            df, model_name, direction, feature, time_step, params,
                            train_dates, y_train_true, train_pred_actual,
                            test_dates, y_test_true, test_pred_actual, metrics
                        )

                        print(f"          Metrics: R2={metrics['R2']:.3f}, MAPE={metrics['MAPE']:.2f}%")

    print("\nAll models finished. Results saved under ./results/DL/")

if __name__ == "__main__":
    main()