"""
Unified script for training and evaluating multiple ML models
for respiratory disease prediction using time-series features.
All results are saved under './results/' with model-specific subfolders.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 1. Global settings & paths
# ==============================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Define base output directory (will be created)
BASE_DIR = './results/ML'
os.makedirs(BASE_DIR, exist_ok=True)

# Plot style (Nature-like)
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.title_fontsize': 8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (7.2, 4.8),
    'figure.constrained_layout.use': True,
})

# Colour palette (blue gradient)
def modern_blue_gradient():
    return {
        'train_actual': '#0A2472',
        'train_pred': '#1C6DD0',
        'test_actual': '#5D8BF4',
        'test_pred': '#9AC5F4',
        'split_line': '#607D8B',
        'background': '#F5F9FF',
    }
COLORS = modern_blue_gradient()

# ==============================
# 2. Data loading and preprocessing
# ==============================
def load_and_preprocess(data_path, holiday_path):
    """Load raw data, apply 7-day moving average, merge holidays."""
    data = pd.read_excel(data_path, engine='openpyxl')  
    data['rq'] = pd.to_datetime(data['rq'])
    
    holiday = pd.read_excel(holiday_path, engine='openpyxl')
    holiday['rq'] = pd.to_datetime(holiday['date'])
    holidays = holiday[['rq', 'is_holiday']]
    
    smodata = data.copy()
    smodata.iloc[:, 2:] = data.iloc[:, 2:].rolling(window=7, min_periods=1).mean()
    
    x_cols = [col for col in smodata.columns if col.startswith('x')]
    selected_cols = ['rq'] + x_cols + ['up', 'down', 'sum']
    data1 = smodata[selected_cols]
    data1 = pd.merge(data1, holidays, on='rq', how='left')
    return data1
# Feature selection
def load_selected_features(selection_path='./results/feature_selection/spearman_selected_features.csv',
                           default_features=None):
    if default_features is None:
        default_features = ['x1', 'x2', 'x5', 'x10', 'x20', 'x26', 'x36', 'x43']
    if os.path.exists(selection_path):
        try:
            df = pd.read_csv(selection_path)
            if 'feature' in df.columns:
                features = df['feature'].tolist()
                print(f"Loaded {len(features)} selected features from {selection_path}")
                return features
            else:
                print(f"Warning: '{selection_path}' missing 'feature' column. Using default list.")
        except Exception as e:
            print(f"Error reading {selection_path}: {e}. Using default list.")
    else:
        print(f"Feature selection file not found at {selection_path}. Using default feature list.")
    return default_features

# Define feature columns (auto-load)
X_FEATURES = load_selected_features()
SPLIT_DATE = pd.Timestamp('2024-09-01')

# ==============================
# 3. Helper functions
# ==============================
def split_data(df, split_date):
    """Split by date into train and test sets."""
    train_mask = df['rq'] < split_date
    test_mask = df['rq'] >= split_date
    return train_mask, test_mask

def evaluate(y_true, y_pred):
    """Compute regression metrics."""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return {'R2': r2, 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def save_predictions(df, x_col, direction, model_name, 
                     train_dates, y_train, train_pred,
                     test_dates, y_test, test_pred):
    """Save both train and test predictions to a single CSV."""
    out_dir = os.path.join(BASE_DIR, model_name, direction)
    os.makedirs(out_dir, exist_ok=True)
    fname = f'predictions_{x_col}_{direction}_holiday.csv'
    train_df = pd.DataFrame({
        'date': train_dates,
        'actual': y_train,
        'predicted': train_pred,
        'set': 'train'
    })
    test_df = pd.DataFrame({
        'date': test_dates,
        'actual': y_test,
        'predicted': test_pred,
        'set': 'test'
    })
    
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined.to_csv(os.path.join(out_dir, fname), index=False)

def save_metrics(metrics_df, model_name, direction):
    """Append or save overall metrics."""
    out_dir = os.path.join(BASE_DIR, model_name, direction)
    os.makedirs(out_dir, exist_ok=True)
    fname = 'model_performance_metrics.csv'
    # If file exists, append; else create
    filepath = os.path.join(out_dir, fname)
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        combined = pd.concat([existing, metrics_df], ignore_index=True)
    else:
        combined = metrics_df
    combined.to_csv(filepath, index=False)

def plot_results(train_dates, y_train, train_pred, test_dates, y_test, test_pred,
                 x_col, direction, model_name, split_date):
    """Generate and save the prediction plot."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    # Training period background
    ax.axvspan(train_dates.min(), split_date, alpha=0.05,
               color=COLORS.get('background', '#F5F9FF'), label='Training period')
    # Training actual
    ax.plot(train_dates, y_train, label='Training (actual)',
            color=COLORS['train_actual'], linewidth=1.4, alpha=0.95, zorder=5)
    # Training predicted
    ax.plot(train_dates, train_pred, label='Training (predicted)',
            color=COLORS['train_pred'], linestyle='--', linewidth=1.4, alpha=0.9, zorder=4)
    # Test actual
    ax.plot(test_dates, y_test, label='Test (actual)',
            color=COLORS['test_actual'], marker='o', markersize=2, linewidth=1.6,
            markeredgewidth=0.8, markeredgecolor=COLORS['train_actual'],
            markerfacecolor=COLORS['test_actual'], alpha=0.95, zorder=6)
    # Test predicted
    ax.plot(test_dates, test_pred, label='Test (predicted)',
            color=COLORS['test_pred'], linestyle='--', marker='s', markersize=2,
            linewidth=1.6, markeredgewidth=0.8,
            markeredgecolor=COLORS['train_pred'], markerfacecolor=COLORS['test_pred'],
            alpha=0.9, zorder=5)
    # Split line
    ax.axvline(x=split_date, color=COLORS['split_line'], linestyle=':',
               linewidth=1.2, alpha=0.8, zorder=3,
               label=f'Split: {split_date.strftime("%Y-%m-%d")}')
    ax.set_xlabel('Date', fontweight='medium')
    ax.set_ylabel('ARIDs cases', fontweight='medium')
    ax.grid(True, which='major', axis='both', linestyle='-', linewidth=0.3,
            alpha=0.2, color='#1E88E5')
    ax.tick_params(axis='both', which='both', length=3, width=0.5)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(bottom=0 if y_min > 0 else y_min, top=y_max)
    # Date formatting
    all_dates = pd.concat([train_dates, test_dates])
    date_range = all_dates.max() - all_dates.min()
    if date_range.days > 180:
        date_fmt = '%Y-%m'
        locator = mdates.MonthLocator(interval=max(1, int(date_range.days/30/6)))
    else:
        date_fmt = '%Y-%m-%d'
        locator = mdates.DayLocator(interval=max(1, int(date_range.days/10)))
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    plt.setp(ax.get_xticklabels(), rotation=0, ha='right')
    ax.legend(frameon=True, framealpha=0.95, edgecolor=COLORS['split_line'],
              facecolor='white', loc='best', borderaxespad=0.5)
    ax.set_title(f'{model_name} model: {x_col}', fontsize=9, fontweight='medium',
                 pad=12, color=COLORS['train_actual'])
    plt.tight_layout()
    # Save
    out_dir = os.path.join(BASE_DIR, model_name, direction, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    fname = f'nature_blue_style_{x_col}_{direction}_holiday.png'
    plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight',
                pad_inches=0.05, facecolor='white', edgecolor='none')
    plt.close()

# ==============================
# 4. Model training functions
# ==============================
def train_linear_regression(X_train, y_train):
    """Linear Regression (no hyperparameters)."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, None

def train_decision_tree(X_train, y_train):
    """Decision Tree with GridSearchCV."""
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'ccp_alpha': [0.0, 0.01, 0.1]
    }
    base = DecisionTreeRegressor(random_state=RANDOM_STATE)
    gs = GridSearchCV(base, param_grid, cv=5, scoring='neg_mean_squared_error',
                      n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_

def train_random_forest(X_train, y_train):
    """Random Forest with GridSearchCV."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    base = RandomForestRegressor(random_state=RANDOM_STATE)
    gs = GridSearchCV(base, param_grid, cv=5, scoring='neg_mean_squared_error',
                      n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_

def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    """
    XGBoost with GridSearchCV + early stopping.
    First, GridSearchCV finds best hyperparameters (without early stopping).
    Then, we retrain with the best params on the full training set,
    using a validation set (split from training) for early stopping.
    """
    # Default param grid (simplified for speed)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    base = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
    gs = GridSearchCV(base, param_grid, cv=5, scoring='neg_mean_squared_error',
                      n_jobs=-1)
    gs.fit(X_train, y_train)
    best_params = gs.best_params_
    
    # Prepare validation set for early stopping (if not provided)
    if X_val is None or y_val is None:
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, shuffle=False)  # keep temporal order? better shuffle for generalisation
        # Actually, for time series we should not shuffle? But here we use a holdout from train.
        # We'll use the last 15% of training dates as validation.
        # Since data is time-ordered, we can split by index.
        split_idx = int(0.85 * len(X_train))
        X_train_split = X_train[:split_idx]
        y_train_split = y_train[:split_idx]
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
    else:
        X_train_split, y_train_split = X_train, y_train
    
    # Retrain with early stopping
    model = XGBRegressor(**best_params, random_state=RANDOM_STATE,
                         early_stopping_rounds=10, eval_metric='rmse',
                         verbosity=0)
    model.fit(X_train_split, y_train_split,
              eval_set=[(X_val, y_val)],
              verbose=False)
    return model, best_params

def train_svm(X_train, y_train):
    """SVM with GridSearchCV (requires scaling)."""
    param_grid = {
        'C': [1.0, 10.0, 100.0],
        'epsilon': [0.1, 0.01, 0.001],
        'gamma': [0.1, 1, 'scale'],
        'kernel': ['rbf', 'poly', 'linear']
    }
    base = SVR()
    gs = GridSearchCV(base, param_grid, cv=5, scoring='neg_mean_squared_error',
                      n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_

# ==============================
# 5. Main execution loop
# ==============================
def main():
    # Load data (adjust file paths as needed)
    data_path = './data/data.xlsx'   # adjust
    holiday_path = './data/holiday.xlsx' # adjust
    df = load_and_preprocess(data_path, holiday_path)
    
    # For each direction (up/down) – which represents a separate feature set?
    # Actually 'up' and 'down' are separate features, and we combine with each x_col and holiday
    directions = ['up', 'down']
    # We'll create a combined feature list: each x_col + direction + is_holiday
    # The original code loops over x_col and direction inside.
    # We'll define a dictionary mapping model names to their training functions
    model_registry = {
        'LR': train_linear_regression,
        'DT': train_decision_tree,
        'RF': train_random_forest,
        'XGBoost': train_xgboost,
        'SVM': train_svm
    }
    
    # For each model
    for model_name, train_func in model_registry.items():
        print(f"\n=== Training {model_name} ===")
        for direction in directions:
            print(f"  Direction: {direction}")
            # Prepare feature column (will be concatenated with direction and holiday)
            for x_col in X_FEATURES:
                print(f"    Feature: {x_col}")
                # Prepare X and y
                # Original code uses: X = df[[x_col, direction, 'is_holiday']]
                X = df[[x_col, direction, 'is_holiday']]
                y = df['sum']
                train_mask, test_mask = split_data(df, SPLIT_DATE)
                train_dates = df.loc[train_mask, 'rq']
                test_dates = df.loc[test_mask, 'rq']
                X_train = X[train_mask]
                y_train = y[train_mask]
                X_test = X[test_mask]
                y_test = y[test_mask]
        
                # Special handling for models that need scaling (SVM, XGBoost)
                # We'll apply scaling inside their training functions or here? 
                # To keep original logic, we scale for SVM and XGBoost only.
                # We'll create copies.
                if model_name in ['SVM', 'XGBoost']:
                    scaler = MinMaxScaler()
                    # Separate numeric and binary columns (binary = is_holiday)
                    num_cols = [0, 1]  # x_col and direction
                    bin_col = [2]      # is_holiday
                    X_train_num = scaler.fit_transform(X_train.iloc[:, num_cols])
                    X_test_num = scaler.transform(X_test.iloc[:, num_cols])
                    X_train_bin = X_train.iloc[:, bin_col].values.reshape(-1, 1)
                    X_test_bin = X_test.iloc[:, bin_col].values.reshape(-1, 1)
                    X_train_scaled = np.concatenate([X_train_num, X_train_bin], axis=1)
                    X_test_scaled = np.concatenate([X_test_num, X_test_bin], axis=1)
                    # Use these for training
                    X_train_use = X_train_scaled
                    X_test_use = X_test_scaled
                else:
                    # For others, use raw values (no scaling)
                    X_train_use = X_train.values
                    X_test_use = X_test.values
                
                # Train model (with grid search)
                if model_name == 'XGBoost':
                    # For XGBoost, we need to split training into train+val for early stopping
                    # We'll pass the entire X_train, y_train and let the function split
                    model, best_params = train_func(X_train_use, y_train.values)
                else:
                    model, best_params = train_func(X_train_use, y_train.values)
                
                # Predictions
                train_pred = model.predict(X_train_use)
                test_pred = model.predict(X_test_use)
                train_pred = np.clip(train_pred, 0, None)
                test_pred = np.clip(test_pred, 0, None)
                
                # Evaluate
                train_metrics = evaluate(y_train.values, train_pred)
                test_metrics = evaluate(y_test.values, test_pred)
                
                # Save metrics
                metrics_row = pd.DataFrame({
                    'Feature': [f"{x_col}_{direction}_holiday"],
                    'MAPE': [test_metrics['MAPE']],
                    'test_MSE': [test_metrics['MSE']],
                    'test_MAE': [test_metrics['MAE']],
                    'RMSE': [test_metrics['RMSE']],
                    'test_R2': [test_metrics['R2']]
                })
                save_metrics(metrics_row, model_name, direction)
                
                # Save predictions
                save_predictions(
                        df, x_col, direction, model_name,
                        train_dates, y_train.values, train_pred,
                        test_dates, y_test.values, test_pred
                    )
                
                # Plot results
                train_dates = df.loc[train_mask, 'rq']
                test_dates = df.loc[test_mask, 'rq']
                plot_results(train_dates, y_train.values, train_pred,
                             test_dates, y_test.values, test_pred,
                             x_col, direction, model_name, SPLIT_DATE)
                
                # (Optional) print best parameters
                if best_params:
                    print(f"      Best params: {best_params}")
                
    print("\nAll models finished. Results saved under ./results/")

if __name__ == "__main__":
    main()