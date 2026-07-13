#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Outbreak detection using Z-score thresholding on model predictions.

This script evaluates the ability of a trained prediction model (e.g., LSTM, XGBoost)
to detect epidemic outbreaks. It computes daily Z-scores based on weekday-specific
baselines (mean and std) derived from the training set, then applies a threshold
to flag predicted outbreaks. Performance metrics (precision, recall, F1, lead days)
are calculated against actual outbreak days.

Usage:
    python z_score_detection.py --predictions ./results/DL/LSTM/up/x1/ts_5/predictions.csv \\
                                --output ./results/outbreak \\
                                --split_date 2024-09-01 \\
                                --thresholds 2.0 2.5 3.0

Input CSV must contain columns: 'date', 'actual', 'predicted' (or custom names).
The script will automatically split train/test based on split_date.
"""

from html import parser
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error
)
import warnings
warnings.filterwarnings('ignore')

# ============================
# 1. Configuration & defaults
# ============================
DEFAULT_THRESHOLDS = [2.0, 2.5, 3.0]
CASE_BINS = [0, 100, 300, 500, np.inf]
CASE_LABELS = ['0-100', '101-300', '301-500', '>500']

# Plotting style (consistent with other scripts)
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'legend.title_fontsize': 7,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Times New Roman'],
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'figure.dpi': 150,
    'savefig.dpi': 150,
})

# Colour palette for strata
STRATUM_COLORS = {
    '0-100': '#1f77b4',
    '101-300': '#ff7f0e',
    '301-500': '#2ca02c',
    '>500': '#d62728'
}

# ============================
# 2. Helper functions
# ============================
def parse_date_col(df, col='date'):
    """Ensure date column is datetime."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
    return df

def compute_weekday_baseline(train_df, target_col='actual'):
    """Calculate mean and std of target per weekday from training set."""
    train_copy = train_df.copy()
    train_copy['weekday'] = train_copy['date'].dt.dayofweek
    baseline = {}
    for wd in range(7):
        subset = train_copy[train_copy['weekday'] == wd][target_col].dropna()
        if len(subset) > 0:
            baseline[wd] = {'mean': subset.mean(), 'std': subset.std(ddof=1)}
        else:
            # fallback to global stats
            global_mean = train_copy[target_col].mean()
            global_std = train_copy[target_col].std(ddof=1)
            baseline[wd] = {'mean': global_mean, 'std': global_std if global_std > 0 else 1e-6}
    return baseline

def find_best_prediction_file(base_dir='./results', model_type='DL', model_name='LSTM',
                              direction='up', metric='R2', ascending=False):
    """
    Find the best prediction CSV based on a given metric.
    For DL: summary file is {model_name}_summary.csv under ./results/DL/{model_name}/
    For ML: summary file is model_performance_metrics.csv under ./results/ML/{model_name}/{direction}/
    Returns path to the corresponding predictions.csv.
    """
    if model_type == 'DL':
        summary_dir = os.path.join(base_dir, 'DL', model_name)
        summary_file = os.path.join(summary_dir, f'{model_name}_summary.csv')
        pred_base = summary_dir
    else:  # ML
        summary_dir = os.path.join(base_dir, 'ML', model_name, direction)
        summary_file = os.path.join(summary_dir, 'model_performance_metrics.csv')
        pred_base = summary_dir

    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    df = pd.read_csv(summary_file)
    # Determine column name for metric
    metric_col = f'test_{metric}' if metric not in df.columns else metric
    if metric_col not in df.columns:
        # Try alternative
        metric_col = 'test_R2' if metric == 'R2' else 'test_MAPE'
    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in summary file.")

    # Find best row
    df = df.sort_values(metric_col, ascending=ascending)
    best_row = df.iloc[0]
    print(f"Best model based on {metric_col}: {best_row.to_dict()}")

    # Construct prediction file path
    if model_type == 'DL':
        direction = best_row['direction']
        feature = best_row['feature']
        time_step = best_row['time_step']
        pred_file = os.path.join(pred_base, direction, feature, f'ts_{time_step}', 'predictions.csv')
    else:  # ML
        # For ML, the feature is in 'Feature' column, e.g., 'x1_up_holiday'
        feature_str = best_row['Feature']
        # Extract x_col and direction from feature string (assumes format x{num}_{direction}_holiday)
        parts = feature_str.split('_')
        if len(parts) >= 2:
            x_col = parts[0]
            dir_part = parts[1]
            # But we already have direction from loop, ensure consistency
        # For ML, predictions are saved as predictions_{x_col}_{direction}_holiday.csv
        pred_file = os.path.join(pred_base, f'predictions_{x_col}_{dir_part}_holiday.csv')
        # If not found, try alternative
        if not os.path.exists(pred_file):
            # Search for file matching pattern
            import glob
            pattern = os.path.join(pred_base, f'predictions_*_{direction}_holiday.csv')
            files = glob.glob(pattern)
            if files:
                pred_file = files[0]  # take first match

    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    return pred_file

def calculate_z_scores(test_df, baseline, target_col='actual'):
    """Add Z-score column to test_df for given target_col."""
    test_copy = test_df.copy()
    test_copy['weekday'] = test_copy['date'].dt.dayofweek
    z_scores = []
    for _, row in test_copy.iterrows():
        wd = row['weekday']
        mean_val = baseline[wd]['mean']
        std_val = baseline[wd]['std']
        if std_val == 0:
            std_val = 1e-6
        z = (row[target_col] - mean_val) / std_val
        z_scores.append(z)
    test_copy['z_score'] = z_scores
    return test_copy

def evaluate_outbreak_at_threshold(test_df, threshold, actual_col='actual', pred_col='predicted'):
    """
    For a given Z threshold, compute outbreak flags and performance metrics.
    Returns: precision, recall, f1, mean_lead_days, n_actual, n_pred, actual_flags, pred_flags
    """
    actual_outbreak = (test_df['z_score_actual'] > threshold).astype(int).values
    pred_outbreak = (test_df['z_score_pred'] > threshold).astype(int).values
    
    n_actual = np.sum(actual_outbreak)
    n_pred = np.sum(pred_outbreak)
    
    if n_actual > 0 and n_pred > 0:
        precision = precision_score(actual_outbreak, pred_outbreak, zero_division=0)
        recall = recall_score(actual_outbreak, pred_outbreak, zero_division=0)
        f1 = f1_score(actual_outbreak, pred_outbreak, zero_division=0)
    else:
        precision = recall = f1 = 0.0
    
    # Lead days (only for actual outbreaks that were predicted on or before the actual day)
    lead_days_list = []
    if n_actual > 0 and n_pred > 0:
        actual_dates = test_df.loc[actual_outbreak == 1, 'date']
        pred_dates = test_df.loc[pred_outbreak == 1, 'date']
        for ad in actual_dates:
            candidates = pred_dates[pred_dates <= ad]
            if len(candidates) > 0:
                closest = candidates.max()
                lead = (ad - closest).days
                if lead >= 0:
                    lead_days_list.append(lead)
    mean_lead_days = np.mean(lead_days_list) if lead_days_list else np.nan
    
    return precision, recall, f1, mean_lead_days, n_actual, n_pred, actual_outbreak, pred_outbreak

def stratified_confusion_table(test_df, actual_outbreak, pred_outbreak, threshold, output_dir):
    """Generate and save stratified confusion matrix by case stratum."""
    strata = CASE_LABELS
    rows = []
    for s in strata:
        mask = test_df['case_stratum'] == s
        if mask.sum() == 0:
            continue
        act = actual_outbreak[mask]
        prd = pred_outbreak[mask]
        tn, fp, fn, tp = confusion_matrix(act, prd, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        rows.append({
            'Stratum': s,
            'Total_Days': int(mask.sum()),
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn),
            'FPR': round(fpr, 3),
            'Actual_Outbreaks': int(np.sum(act)),
            'Pred_Outbreaks': int(np.sum(prd))
        })
    # Overall row
    tn, fp, fn, tp = confusion_matrix(actual_outbreak, pred_outbreak, labels=[0, 1]).ravel()
    fpr_all = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    rows.append({
        'Stratum': 'Overall',
        'Total_Days': len(actual_outbreak),
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
        'FPR': round(fpr_all, 3),
        'Actual_Outbreaks': int(np.sum(actual_outbreak)),
        'Pred_Outbreaks': int(np.sum(pred_outbreak))
    })
    conf_df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, f'confusion_by_stratum_Z{threshold:.1f}.xlsx')
    conf_df.to_excel(out_path, index=False)
    print(f"  - Stratified confusion table (Z={threshold:.1f}) saved to {out_path}")
    return conf_df

# ============================
# 3. Plotting functions
# ============================
def plot_timeseries_with_outbreak(train_df, test_df, threshold, output_dir, metrics_dict):
    """
    Generate Figure with two panels:
    A: full time series (train + test) with predictions
    B: test period with outbreak markers (actual/pred)
    """
    # Use unified color palette
    colors = {
        'train_actual': '#5D8BF4',
        'train_pred': '#9AC5F4',
        'test_actual': '#0A2472',
        'test_pred': '#E2656D',
        'split_line': '#607D8B',
        'background': '#F5F9FF',
        'outbreak_actual': '#D62728',
        'outbreak_pred': '#2CA02C',
    }
    split_date = train_df['date'].max()  # or use provided split_date
    test_dates = test_df['date'].values
    train_dates = train_df['date'].values
    train_actual = train_df['actual'].values
    train_pred = train_df['predicted'].values
    test_actual = test_df['actual'].values
    test_pred = test_df['predicted'].values
    actual_outbreak = test_df['outbreak_actual'].values
    pred_outbreak = test_df['outbreak_pred'].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.5),
                                   gridspec_kw={'height_ratios': [1.2, 1]})

    # Panel A: Full series
    ax1.axvspan(train_dates.min(), split_date, alpha=0.05, color=colors['background'])
    ax1.plot(train_dates, train_actual, label='Training (actual)',
             color=colors['train_actual'], linewidth=1.0, alpha=0.95)
    ax1.plot(train_dates, train_pred, label='Training (predicted)',
             color=colors['train_pred'], linewidth=1.0, linestyle='--', alpha=0.9)
    ax1.plot(test_dates, test_actual, label='Test (actual)',
             color=colors['test_actual'], linewidth=1.0, alpha=0.9)
    ax1.plot(test_dates, test_pred, label='Test (predicted)',
             color=colors['test_pred'], linewidth=1.2, linestyle='-', marker='o',
             markersize=1, alpha=0.9)
    ax1.axvline(x=split_date, color=colors['split_line'],
                linestyle=':', linewidth=1.2, alpha=0.8,
                label=f'Split: {split_date.strftime("%Y-%m-%d")}')
    ax1.set_xlabel('')
    ax1.set_ylabel('ARIDs cases', fontweight='medium')
    ax1.grid(True, linestyle='-', linewidth=0.3, alpha=0.2, color='#1E88E5')
    ax1.legend(loc='upper left', framealpha=0.9, facecolor='white')
    ax1.set_title('A. Model prediction', fontsize=8, fontweight='medium', loc='left')
    # Metrics box
    r2 = metrics_dict.get('R2', np.nan)
    mape = metrics_dict.get('MAPE', np.nan)
    rmse = metrics_dict.get('RMSE', np.nan)
    text = f'  R² = {r2:.2f}\n  MAPE = {mape:.1f}%\n  RMSE = {rmse:.1f}'
    ax1.text(0.98, 0.95, text, transform=ax1.transAxes,
             fontsize=7.5, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#778899'))

    # Panel B: Test period with outbreak markers
    ax2.plot(test_dates, test_actual, label='Test (actual)',
             color=colors['test_actual'], linewidth=1.0, alpha=0.9)
    ax2.plot(test_dates, test_pred, label='Test (predicted)',
             color=colors['test_pred'], linewidth=1.2, linestyle='-', marker='o',
             markersize=1, alpha=0.9)

    # Color-coded strata (background scatter)
    for s in CASE_LABELS:
        mask = test_df['case_stratum'] == s
        if mask.sum() > 0:
            ax2.scatter(test_df.loc[mask, 'date'], test_df.loc[mask, 'actual'],
                        color=STRATUM_COLORS[s], s=5, alpha=0.3, label=f'Stratum {s}')

    # Outbreak points
    actual_out_dates = test_df.loc[actual_outbreak == 1, 'date']
    actual_out_vals = test_df.loc[actual_outbreak == 1, 'actual']
    if len(actual_out_dates) > 0:
        ax2.scatter(actual_out_dates, actual_out_vals, marker='o', s=20,
                    color=colors['outbreak_actual'], edgecolor='black', linewidth=0.8,
                    label='Actual outbreak', zorder=10)
    pred_out_dates = test_df.loc[pred_outbreak == 1, 'date']
    pred_out_vals = test_df.loc[pred_outbreak == 1, 'predicted']
    if len(pred_out_dates) > 0:
        ax2.scatter(pred_out_dates, pred_out_vals, marker='*', s=30,
                    color=colors['outbreak_pred'], edgecolor='black', linewidth=0.8,
                    label='Predicted outbreak', zorder=10)

    ax2.set_xlabel('Date', fontweight='medium')
    ax2.set_ylabel('ARIDs cases', fontweight='medium')
    ax2.grid(True, linestyle='-', linewidth=0.3, alpha=0.2, color='#1E88E5')
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=6, ncol=2)
    ax2.set_title(f'B. Outbreak detection (Z={threshold:.1f})', fontsize=8,
                  fontweight='medium', loc='left')

    # Metrics box for outbreak detection
    prec = metrics_dict.get('Precision', np.nan)
    rec = metrics_dict.get('Recall', np.nan)
    f1 = metrics_dict.get('F1', np.nan)
    text2 = f'  Precision = {prec:.2f}\n  Recall = {rec:.2f}\n  F1-score = {f1:.2f}'
    ax2.text(0.98, 0.95, text2, transform=ax2.transAxes,
             fontsize=7.5, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#778899'))

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'Figure_outbreak_Z{threshold:.1f}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  - Time-series plot saved to {out_path}")

def plot_scatter_with_outbreak(test_df, threshold, output_dir):
    """Scatter plot of actual vs predicted with TP and FP highlighted."""
    fig, ax = plt.subplots(figsize=(7, 6))
    actual = test_df['actual'].values
    pred = test_df['predicted'].values
    actual_out = test_df['outbreak_actual'].values
    pred_out = test_df['outbreak_pred'].values

    tp_mask = (actual_out == 1) & (pred_out == 1)
    fp_mask = (actual_out == 0) & (pred_out == 1)
    others = ~(tp_mask | fp_mask)

    ax.scatter(actual[others], pred[others], color='gray', s=12, alpha=0.3, label='Others')
    if tp_mask.sum() > 0:
        ax.scatter(actual[tp_mask], pred[tp_mask], color='green', s=50,
                   edgecolors='black', linewidth=0.5, label=f'TP (n={tp_mask.sum()})')
    if fp_mask.sum() > 0:
        ax.scatter(actual[fp_mask], pred[fp_mask], color='red', s=50,
                   edgecolors='black', linewidth=0.5, label=f'FP (n={fp_mask.sum()})')

    max_val = max(actual.max(), pred.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.8, label='y = x')
    ax.set_xlabel('Actual cases')
    ax.set_ylabel('Predicted cases')
    ax.set_title(f'Scatter plot (Z={threshold:.1f})')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'scatter_outbreak_Z{threshold:.1f}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  - Scatter plot saved to {out_path}")

# ============================
# 4. Main analysis function
# ============================
def run_outbreak_analysis(pred_file, output_dir, split_date=None,
                          date_col='date', actual_col='actual', pred_col='predicted',
                          thresholds=DEFAULT_THRESHOLDS):
    """
    Main pipeline:
    1. Load predictions CSV
    2. Split train/test by date
    3. Compute weekday baselines from training set
    4. For each threshold:
       - Compute Z-scores for actual and predicted
       - Evaluate outbreak detection
       - Generate plots and tables
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(pred_file)
    df = parse_date_col(df, date_col)
    df.sort_values(date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Determine split date
    if split_date is None:
        # Use date from test set? We assume split date is provided.
        # If not, use the last date in training (if train/test already separated)
        # Here we assume the file contains both train and test, and we split by a fixed date.
        # If split_date is None, we set a default (e.g., 2024-09-01)
        split_date = pd.Timestamp('2024-09-01')
        print(f"Warning: split_date not provided, using default {split_date.strftime('%Y-%m-%d')}")
    else:
        split_date = pd.Timestamp(split_date)

    train_df = df[df[date_col] < split_date].copy()
    test_df = df[df[date_col] >= split_date].copy()
    print(f"Train set: {len(train_df)} days, Test set: {len(test_df)} days")

    # Compute baseline from training set (using actual values)
    baseline = compute_weekday_baseline(train_df, target_col=actual_col)

    # Add Z-score columns for actual and predicted in test set
    test_df = calculate_z_scores(test_df, baseline, target_col=actual_col)
    test_df.rename(columns={'z_score': 'z_score_actual'}, inplace=True)
    test_df = calculate_z_scores(test_df, baseline, target_col=pred_col)
    test_df.rename(columns={'z_score': 'z_score_pred'}, inplace=True)

    # Add case stratum for stratification
    test_df['case_stratum'] = pd.cut(test_df[actual_col],
                                     bins=CASE_BINS, labels=CASE_LABELS,
                                     right=True, include_lowest=True)

    # Overall test set metrics (R2, MAPE, RMSE)
    act_clean = test_df[actual_col].dropna()
    pred_clean = test_df[pred_col].dropna()
    # Align lengths (they should be the same)
    if len(act_clean) == len(pred_clean):
        r2 = r2_score(act_clean, pred_clean) if len(act_clean) > 1 else np.nan
        mse = mean_squared_error(act_clean, pred_clean)
        mae = mean_absolute_error(act_clean, pred_clean)
        rmse = np.sqrt(mse)
        eps = 1e-10
        mape = np.mean(np.abs((act_clean - pred_clean) / (act_clean + eps))) * 100
    else:
        r2 = mape = rmse = np.nan

    overall_metrics = {'R2': r2, 'MAPE': mape, 'RMSE': rmse}

    # Store results for all thresholds
    results = []

    for thr in thresholds:
        print(f"\n--- Evaluating threshold Z = {thr:.1f} ---")
        prec, rec, f1, lead, n_act, n_pred, act_flags, pred_flags = evaluate_outbreak_at_threshold(
            test_df, thr, actual_col, pred_col
        )
        # Store flags in test_df for plotting
        test_df['outbreak_actual'] = act_flags
        test_df['outbreak_pred'] = pred_flags

        # Collect metrics
        row = {
            'Threshold': thr,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'Mean_Lead_Days': lead,
            'N_Actual_Outbreaks': n_act,
            'N_Predicted_Outbreaks': n_pred
        }
        results.append(row)

        # Generate plots
        plot_metrics = {
            'R2': overall_metrics['R2'],
            'MAPE': overall_metrics['MAPE'],
            'RMSE': overall_metrics['RMSE'],
            'Precision': prec,
            'Recall': rec,
            'F1': f1
        }
        plot_timeseries_with_outbreak(train_df, test_df, thr, output_dir, plot_metrics)
        plot_scatter_with_outbreak(test_df, thr, output_dir)

        # Stratified confusion table
        stratified_confusion_table(test_df, act_flags, pred_flags, thr, output_dir)

    # Save overall outbreak metrics summary
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(output_dir, 'outbreak_metrics_summary.xlsx')
    summary_df.to_excel(summary_path, index=False)
    print(f"\nOutbreak metrics summary saved to {summary_path}")
    print(summary_df.to_string(index=False))

    # Also save the test_df with all columns for further inspection
    test_df.to_excel(os.path.join(output_dir, 'test_data_with_outbreak_flags.xlsx'), index=False)

    return summary_df, test_df

# ============================
# 5. Command-line interface
# ============================
def main():
    parser = argparse.ArgumentParser(
        description="Outbreak detection using Z-score on model predictions."
    )
    parser.add_argument('--auto_best', action='store_true',
                    help='Automatically select best model from summary files')
    parser.add_argument('--model_type', type=str, choices=['ML', 'DL'], default='DL',
                        help='Type of model: ML or DL (used with --auto_best)')
    parser.add_argument('--model_name', type=str, default='LSTM',
                        help='Model name, e.g., LSTM, XGBoost (used with --auto_best)')
    parser.add_argument('--direction', type=str, default='up',
                        help='Direction (up/down) for ML models (used with --auto_best)')
    parser.add_argument('--metric', type=str, default='R2',
                        help='Metric to optimize (R2, MAPE, RMSE) (used with --auto_best)')
    parser.add_argument('--ascending', action='store_true', default=False,
                        help='Sort ascending (e.g., for MAPE lower is better)')
    parser.add_argument('--predictions', type=str, required=False,
                    help='Path to CSV with predictions (columns: date, actual, predicted)')
    parser.add_argument('--output', type=str, default='./results/outbreak',
                        help='Output directory for results (default: ./results/outbreak)')
    parser.add_argument('--split_date', type=str, default='2024-09-01',
                        help='Date to split train/test (format: YYYY-MM-DD)')
    parser.add_argument('--date_col', type=str, default='date',
                        help='Name of date column in CSV')
    parser.add_argument('--actual_col', type=str, default='actual',
                        help='Name of actual cases column')
    parser.add_argument('--pred_col', type=str, default='predicted',
                        help='Name of predicted cases column')
    parser.add_argument('--thresholds', type=float, nargs='+', default=DEFAULT_THRESHOLDS,
                        help='Z-score thresholds to evaluate (default: 2.0 2.5 3.0)')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    results_base = os.path.join(project_root, 'results')

    # Determine prediction file
    if args.auto_best:
        pred_file = find_best_prediction_file(
            base_dir=results_base,
            model_type=args.model_type,
            model_name=args.model_name,
            direction=args.direction,
            metric=args.metric,
            ascending=args.ascending
        )
        print(f"Auto-selected prediction file: {pred_file}")
    else:
        if not args.predictions:
            parser.error("Either provide --predictions or use --auto_best")
        pred_file = args.predictions

    if args.output:
        output_dir = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output
    else:
        output_dir = os.path.join(results_base, 'outbreak')

    run_outbreak_analysis(
        pred_file=pred_file,
        output_dir=output_dir,
        split_date=args.split_date,
        date_col=args.date_col,
        actual_col=args.actual_col,
        pred_col=args.pred_col,
        thresholds=args.thresholds
    )
if __name__ == "__main__":
    main()