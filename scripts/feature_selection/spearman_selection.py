"""
Spearman correlation analysis with linear interpolation for missing values.
Only processes data (symptom combination frequencies) against target variable 'sum'.
Produces:
  1) Full correlation table (all features)
  2) Filtered feature list with |correlation| > 0.4 (for predictive modeling input)

Usage:
    python spearman_feature_selection.py --data ./data/data.xlsx --target sum --output ./results/
"""

import os
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import List, Tuple

# ============================== Configuration ==============================
DEFAULT_DATA_PATH = './data/data.xlsx'
DEFAULT_OUTPUT_DIR = './results/feature_selection'
DEFAULT_TARGET = 'sum'

# ============================== Functions =================================
def load_data(file_path: str) -> pd.DataFrame:
    """Load Excel file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_excel(file_path)
    print(f"Loaded data shape: {df.shape}")
    return df

def identify_feature_columns(df: pd.DataFrame, target_col: str, exclude_cols: List[str] = None) -> List[str]:
    """
    Identify feature columns by excluding date columns and the target column.
    """
    if exclude_cols is None:
        exclude_cols = ['rq', target_col]   # exclude date and target
    # data contains symptom combinations (x1~x43) plus up/down/normal etc.
    # We only want the symptom combination columns (x1, x2, ...).
    # But to be safe, we exclude the known aggregated/non-symptom columns.
    extra_exclude = ['up', 'down', 'normal']
    exclude_cols = exclude_cols + extra_exclude
    
    all_cols = df.columns.tolist()
    feature_cols = [col for col in all_cols if col not in exclude_cols]
    print(f"Identified {len(feature_cols)} feature columns (symptom combinations).")
    return feature_cols

def spearman_correlation_with_interpolation(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> pd.DataFrame:
    """
    Compute Spearman correlation after linear interpolation of missing values.
    Interpolation is performed in time order (if 'rq' column exists) to preserve time-series continuity.
    """
    # Sort by date to respect time-series order for interpolation
    if 'rq' in df.columns:
        df_sorted = df.sort_values('rq').copy()
        print("Data sorted by 'rq' for linear interpolation.")
    else:
        df_sorted = df.copy()
        print("Warning: 'rq' column not found, interpolation uses current row order (assume time-ordered).")

    # Interpolate missing values for the target and all features
    cols_to_interpolate = [target_col] + feature_cols
    # Only keep columns that actually exist in the dataframe
    existing_cols = [col for col in cols_to_interpolate if col in df_sorted.columns]
    
    # Linear interpolation with forward and backward filling (to handle edge NaNs)
    # 'limit_direction='both'' ensures leading/trailing NaNs are also filled using nearest neighbor if linear fails
    df_sorted[existing_cols] = df_sorted[existing_cols].interpolate(
        method='linear',
        limit_direction='both',   # fills missing values at the beginning and end as well
        axis=0
    )
    
    # After interpolation, there might still be columns that are completely NaN (all missing).
    # We'll drop them from correlation calculation.
    results = []
    for col in feature_cols:
        if col not in df_sorted.columns:
            continue
        
        # If entire column is NaN after interpolation, skip
        if df_sorted[col].isna().all():
            print(f"Warning: {col} is completely missing (all NaN), skipping.")
            continue
        
        # Extract valid pairs (should be nearly all after interpolation, but keep this for safety)
        valid = df_sorted[[col, target_col]].dropna()
        if len(valid) < 3:
            print(f"Warning: {col} has insufficient valid pairs (n={len(valid)}) after interpolation, skipping.")
            continue
        
        corr, p = spearmanr(valid[col], valid[target_col])
        results.append({
            'feature': col,
            'correlation': round(corr, 4),
            'p_value': round(p, 4)
        })
    
    return pd.DataFrame(results)

def save_results(all_results: pd.DataFrame, output_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    """Save full results and filtered features (|correlation| > 0.4). Return filtered list."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save all results
    all_path = os.path.join(output_dir, 'spearman_all_results.csv')
    all_results.to_csv(all_path, index=False)
    print(f"Full results saved to: {all_path}")
    
    # 2. Filter |correlation| > 0.4
    filtered = all_results[abs(all_results['correlation']) > 0.4].copy()
    filtered = filtered.sort_values('correlation', ascending=False)  # descending by strength
    
    if not filtered.empty:
        selected_features = filtered['feature'].tolist()
        pd.DataFrame({'feature': selected_features}).to_csv(os.path.join(output_dir, 'selected_feature_names.csv'), index=False)
        selected_path = os.path.join(output_dir, 'spearman_selected_features.csv')
        filtered.to_csv(selected_path, index=False)
        print(f"Filtered features (|corr|>0.4) saved to: {selected_path}")
        print(f"Number of selected features: {len(selected_features)}")
        print("Selected features:", selected_features)
    else:
        print("No features with |correlation| > 0.4 found.")
        selected_features = []
    
    return filtered, selected_features

def main(args):
    # Load data
    df = load_data(args.data)
    print(f"Columns: {df.columns.tolist()}")
    
    # Identify feature columns (symptom combination columns only)
    feature_cols = identify_feature_columns(df, args.target)
    
    # Compute Spearman correlation with linear interpolation
    results_df = spearman_correlation_with_interpolation(df, feature_cols, args.target)
    print(f"Computed correlations for {len(results_df)} features.")
    
    # Save and filter
    filtered_df, selected_features = save_results(results_df, args.output_dir)
    
    # Print summary
    print("\n===== Summary =====")
    print(f"Total features evaluated: {len(results_df)}")
    print(f"Features with |corr| > 0.4: {len(selected_features)}")
    if len(selected_features) > 0:
        print("Top 5 strongest correlations (absolute value):")
        print(filtered_df.head(5).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spearman correlation feature selection with linear interpolation for data.")
    parser.add_argument('--data', type=str, default=DEFAULT_DATA_PATH,
                        help='Path to data Excel file (symptom combinations)')
    parser.add_argument('--target', type=str, default=DEFAULT_TARGET,
                        help='Target column name (default: sum)')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save output CSV files')
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()