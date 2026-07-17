"""
Run this script to reproduce results and figures for the LLM performance section.

Input files (place under ./data/):
    - train1000mark.csv (gold standard)
    - LLM predictions
    - train_zz.csv (Regex / regex predictions)

Output (saved under ./results/LLM/):
    - symeva.xlsx (merged performance table)
    - statistical_analysis_results_detailed.xlsx
    - Figure4.png/tif/pdf/svg
    - Figure3.png/tif/pdf/svg
    - model_confusion_matrices.png/tif/pdf/svg
    - symptom_*_analysis_advanced.png/svg
    - model_evaluation.csv, fourfold_table_*.csv, zz_table_*.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
from scikit_posthocs import posthoc_dunn
from itertools import combinations
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sympy import re
warnings.filterwarnings('ignore')

# ============================================================================
# 0. Configuration
# ============================================================================
DATA_DIR = './data'
OUTPUT_DIR = './results/LLM'
os.makedirs(OUTPUT_DIR, exist_ok=True)

GOLD_FILE = os.path.join(DATA_DIR, 'train1000mark.csv')
REGEX_FILE = os.path.join(DATA_DIR, 'train_zz.csv')

# LLM model files (name -> filename)
def find_model_files(data_dir):
    model_files = {}
    for fname in os.listdir(data_dir):
        if not fname.endswith('.csv'):
            continue
        if fname in ['train1000mark.csv', 'train_zz.csv']:
            continue
        model_key = os.path.splitext(fname)[0]
        model_files[model_key] = fname
    return model_files

MODEL_FILES = find_model_files(DATA_DIR)
print(f"MODEL FILES: {list(MODEL_FILES.keys())}")

# Base processing time (seconds) for each model
# Extracted from symeva.xlsx pattern: total = symptom_idx*3600 + base
def get_model_total_time(model_key):
    """Return total processing time in seconds for the given model."""
    # Fixed value for Regex (0.1 seconds)
    if model_key == 'Regex':
        return 0.1

    time_file = os.path.join(DATA_DIR, f"{model_key}_time.txt")
    if os.path.exists(time_file):
        with open(time_file, 'r') as f:
            lines = f.read().strip().splitlines()
            if len(lines) >= 1:
                first_line = lines[0].strip()
                try:
                    return float(first_line)
                except ValueError:
                    try:
                        parts = first_line.split(':')
                        if len(parts) == 3:
                            h = int(parts[0])
                            m = int(parts[1])
                            s = float(parts[2])
                            return h * 3600 + m * 60 + s
                    except:
                        pass
                    try:
                        parts = first_line.split(':')
                        if len(parts) == 2:
                            m = int(parts[0])
                            s = float(parts[1])
                            return m * 60 + s
                    except:
                        pass
                    print(f"Warning: Could not parse time '{first_line}' for {model_key}")
                    return 60.0  

def format_total_time(total_seconds):
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', '.').ljust(15, '0')[:15]

# Fixed display order (must match the order in independent script)
MODEL_ORDER = [
    'Gemma-3-1B', 'Gemma-3-4B', 
    'Qwen-3-1.7B', 'Qwen-3-8B',
    'Deepseek-r1-7B', 
    'Llama-3.1-8B', 'Llama-3.2-3B', 
    'Regex'
]

# Symptom order and mapping
SYMPTOM_LIST = ['fever', 'cough', 'sore throat', 'chest pain', 'myalgia', 'dyspnea']
# Map from gold column msym{i} and pred column sym{i} to display name
SYMPTOM_MAP = {
    1: 'fever',
    2: 'cough',
    3: 'sore throat',
    4: 'chest pain',
    5: 'myalgia',
    6: 'dyspnea'
}
# Reverse for ID lookup
SYMPTOM_TO_IDX = {v: i for i, v in enumerate(SYMPTOM_LIST)}

MODEL_NAME_MAP = {
    'gemma3-1b': 'Gemma-3-1B',
    'gemma3-4b': 'Gemma-3-4B',
    'llama3.2-3b': 'Llama-3.2-3B',
    'llama3.1-8b': 'Llama-3.1-8B',
    'deepseek-r1-7b': 'Deepseek-r1-7B',
    'qwen3-1.7b': 'Qwen-3-1.7B',
    'qwen3-8b': 'Qwen-3-8B'
}

# ============================================================================
# 1. Helper Functions (improved time parsing)
# ============================================================================
def get_model_total_time(model_key):
    """Return total processing time in seconds for the given model."""
    if model_key == 'Regex':
        return 0.1

    # Try explicit time file
    time_file = os.path.join(DATA_DIR, f"{model_key}_time.txt")
    if os.path.exists(time_file):
        with open(time_file, 'r') as f:
            lines = f.read().strip().splitlines()
            if lines:
                first_line = lines[0].strip()
                # Try float
                try:
                    return float(first_line)
                except ValueError:
                    pass
                # Try H:MM:SS.ffffff
                try:
                    parts = first_line.split(':')
                    if len(parts) == 3:
                        h = int(parts[0]); m = int(parts[1]); s = float(parts[2])
                        return h*3600 + m*60 + s
                except:
                    pass
                # Try MM:SS.s
                try:
                    parts = first_line.split(':')
                    if len(parts) == 2:
                        m = int(parts[0]); s = float(parts[1])
                        return m*60 + s
                except:
                    pass
                print(f"Warning: Could not parse time '{first_line}' for {model_key}")

    # Fallback: read from log file
    log_file = os.path.join(DATA_DIR, f"{model_key}.log")
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        match = re.search(r'Total time:\s*([0-9]+):([0-9]{2}):([0-9]{2}\.[0-9]+)', content)
        if match:
            h = int(match.group(1)); m = int(match.group(2)); s = float(match.group(3))
            return h*3600 + m*60 + s
        match = re.search(r'Total time:\s*([0-9.]+)\s*seconds', content)
        if match:
            return float(match.group(1))
        match = re.search(r'Total time:\s*([0-9]+)s', content)
        if match:
            return float(match.group(1))

    print(f"Warning: No time info for {model_key}, using default 60s")
    return 60.0

def format_total_time(total_seconds):
    """Convert seconds to H:MM:SS.ffffff format."""
    if total_seconds is None:
        total_seconds = 60.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:06.6f}"

def compute_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return tp, fp, tn, fn, acc, prec, rec, f1

# ============================================================================
# 2. Evaluate LLM Models (unchanged)
# ============================================================================
def evaluate_llm_models():
    gold = pd.read_csv(GOLD_FILE)
    rows = []

    for model_name, filename in MODEL_FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping {model_name}")
            continue
        pred = pd.read_csv(path)
        if len(pred) != len(gold):
            print(f"Warning: {model_name} length mismatch, skipping")
            continue

        for i in range(1, 7):
            symptom_display = SYMPTOM_MAP[i]
            y_true = gold[f'msym{i}']
            y_pred = pred[f'sym{i}']
            tp, fp, tn, fn, acc, prec, rec, f1 = compute_metrics(y_true, y_pred)

            total_sec = get_model_total_time(model_name)
            time_str = format_total_time(total_sec) 
            display_name = MODEL_NAME_MAP.get(model_name, model_name)

            rows.append({
                'Symptom': symptom_display,
                'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
                'Accuracy': acc, 'Precision': prec,
                'Recall': rec, 'F1-score': f1,
                'Time': time_str,
                'Model': display_name
            })
    return rows

# ============================================================================
# 3. Evaluate Regex 
# ============================================================================
def evaluate_keyword_matching():
    train_standard = pd.read_csv(GOLD_FILE)
    model_files = {'keyword matching': REGEX_FILE}
    id_cols = ['jzlsh', 'yljgdm']
    
    # ---- Align records by composite key ----
    model_df = pd.read_csv(REGEX_FILE)
    # Check columns
    for col in id_cols:
        if col not in train_standard.columns or col not in model_df.columns:
            raise ValueError(f"Column '{col}' missing")

    std_key = train_standard[id_cols[0]].astype(str) + "_" + train_standard[id_cols[1]].astype(str)
    mod_key = model_df[id_cols[0]].astype(str) + "_" + model_df[id_cols[1]].astype(str)

    std_indexed = train_standard.copy()
    std_indexed['_key'] = std_key
    std_indexed = std_indexed.set_index('_key').sort_index()

    mod_indexed = model_df.copy()
    mod_indexed['_key'] = mod_key
    mod_indexed = mod_indexed.set_index('_key').sort_index()

    common_keys = std_indexed.index.intersection(mod_indexed.index)
    print(f"Regex alignment: {len(common_keys)} common records")
    if len(common_keys) == 0:
        raise ValueError("No common (jzlsh, yljgdm) keys found")

    std_aligned = std_indexed.loc[common_keys].reset_index(drop=True)
    mod_aligned = mod_indexed.loc[common_keys].reset_index(drop=True)

    # ---- Compute metrics per symptom ----
    rows = []
    for i in range(1, 7):
        symptom_display = SYMPTOM_MAP[i]
        y_true = std_aligned[f'msym{i}']
        y_pred = mod_aligned[f'sym{i}']

        tp, fp, tn, fn, acc, prec, rec, f1 = compute_metrics(y_true, y_pred)

        total_sec = get_model_total_time('Regex')
        time_str = format_total_time(total_sec)

        rows.append({
            'Symptom': symptom_display,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1-score': f1,
            'Time': time_str,
            'Model': 'Regex'
        })
    return rows

# ============================================================================
# 4. Generate symeva.xlsx
# ============================================================================
def generate_symeva():
    all_rows = evaluate_llm_models() + evaluate_keyword_matching()
    df = pd.DataFrame(all_rows)
    cols = ['Symptom', 'TP', 'FP', 'TN', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Time', 'Model']
    df = df[cols]
    df['Symptom_order'] = df['Symptom'].map(SYMPTOM_TO_IDX)
    numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    df[numeric_cols] = df[numeric_cols].round(2)

    # Fix model order
    existing_models = [m for m in MODEL_ORDER if m in df['Model'].unique()]
    other_models = [m for m in df['Model'].unique() if m not in MODEL_ORDER]
    ordered_models = existing_models + other_models

    df['Model'] = pd.Categorical(df['Model'], categories=ordered_models, ordered=True)
    df = df.sort_values(['Model', 'Symptom_order']).drop(columns=['Symptom_order'])
    df.to_excel(os.path.join(OUTPUT_DIR, 'symeva.xlsx'), index=False)
    print(f"symeva.xlsx generated at {OUTPUT_DIR}")
    return df

# ============================================================================
# 5. Statistical Analysis 
# ============================================================================
def run_statistical_analysis():
    print("\n" + "="*60)
    print("Running Statistical Analysis on symeva.xlsx")
    print("="*60)

    df = pd.read_excel(os.path.join(OUTPUT_DIR, 'symeva.xlsx'))

    # Use fixed model order
    existing_models = [m for m in MODEL_ORDER if m in df['Model'].unique()]
    other_models = [m for m in df['Model'].unique() if m not in MODEL_ORDER]
    models = existing_models + other_models

    # ---- Time conversion ----
    def time_to_seconds(tstr):
        if isinstance(tstr, str):
            tstr = tstr.strip()
            # H:MM:SS.ffffff
            try:
                t = datetime.strptime(tstr, "%H:%M:%S.%f")
                return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
            except:
                pass
            # MM:SS.s
            try:
                parts = tstr.split(':')
                if len(parts) == 2:
                    m = int(parts[0]); s = float(parts[1])
                    return m * 60 + s
            except:
                pass
            try:
                return float(tstr)
            except:
                return np.nan
        else:
            # if already numeric
            return tstr
    df['Time_seconds'] = df['Time'].apply(time_to_seconds)

    # Derived metrics
    df['Specificity'] = df['TN'] / (df['TN'] + df['FP'] + 1e-10)
    df['Balanced_Accuracy'] = (df['Recall'] + df['Specificity']) / 2

    # ---- Colors ----
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#996E2E', '#B2D3A4', '#56B4E9', '#E377C2']
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(models)}

    # ---- 5a. Descriptive stats ----
    numeric_cols = ['TP', 'FP', 'TN', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Balanced_Accuracy']
    model_stats = df.groupby('Model')[numeric_cols].agg(['mean', 'std', 'min', 'max', 'median'])
    symptom_stats = df.groupby('Symptom')[numeric_cols].agg(['mean', 'std', 'min', 'max', 'median'])

    # ---- 5b. Per-symptom Friedman + pairwise comparisons (based on improvement >5%) ----
    symptom_results = {}
    for symptom in df['Symptom'].unique():
        symptom_data = df[df['Symptom'] == symptom]
        symptom_results[symptom] = {}
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
            values = []
            valid_models = []
            for m in models:
                v = symptom_data[symptom_data['Model'] == m][metric].values
                if len(v) > 0 and not np.isnan(v[0]):
                    values.append(v[0])
                    valid_models.append(m)
            if len(values) >= 3:
                f_stat, f_p = stats.friedmanchisquare(*values)
                pairwise = []
                for m1, m2 in combinations(valid_models, 2):
                    v1 = symptom_data[symptom_data['Model'] == m1][metric].values[0]
                    v2 = symptom_data[symptom_data['Model'] == m2][metric].values[0]
                    diff = v1 - v2
                    imp = (diff / v2 * 100) if v2 != 0 else np.nan
                    sig = abs(imp) > 5.0 if not np.isnan(imp) else False
                    pairwise.append({'model1': m1, 'model2': m2,
                                     'mean_diff': diff, 'improvement_pct': imp,
                                     'significant': sig})
                symptom_results[symptom][metric] = {
                    'friedman_stat': f_stat,
                    'friedman_p': f_p,
                    'pairwise': pairwise
                }
            else:
                symptom_results[symptom][metric] = {'friedman_stat': np.nan, 'friedman_p': np.nan, 'pairwise': []}

    # ---- 5c. Global analysis (Friedman + Mann-Whitney pairwise) ----
    global_results = {}
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-score', 'Balanced_Accuracy']:
        data_dict = {m: df[df['Model'] == m][metric].values for m in models}
        mat = [data_dict[m] for m in models]
        max_len = max(len(a) for a in mat)
        mat = [np.pad(a, (0, max_len - len(a)), constant_values=np.nan) for a in mat]
        f_stat, f_p = stats.friedmanchisquare(*mat) if len(models)>=3 else (np.nan, np.nan)
        pairwise = []
        for m1, m2 in combinations(models, 2):
            d1, d2 = data_dict[m1], data_dict[m2]
            if len(d1) and len(d2):
                u, p = stats.mannwhitneyu(d1, d2, alternative='two-sided',method='asymptotic')
                pairwise.append({'model1': m1, 'model2': m2,
                                 'mean_diff': np.mean(d1)-np.mean(d2),
                                 'p_value': p, 'significant': p<0.05})
        global_results[metric] = {'friedman_stat': f_stat, 'friedman_p': f_p, 'pairwise': pairwise}

    # ---- 5d. Time analysis (Kruskal-Wallis + pairwise Mann-Whitney) ----
    time_groups = [df[df['Model'] == m]['Time_seconds'].values for m in models if m != 'Regex']
    if len(time_groups) >= 2:
        k_stat, k_p = stats.kruskal(*time_groups)
        time_pairwise = []
        for m1, m2 in combinations([m for m in models if m != 'Regex'], 2):
            t1 = df[df['Model'] == m1]['Time_seconds'].values
            t2 = df[df['Model'] == m2]['Time_seconds'].values
            if len(t1) and len(t2):
                u, p = stats.mannwhitneyu(t1, t2, alternative='two-sided')
                time_pairwise.append({'model1': m1, 'model2': m2,
                                      'mean_diff': np.mean(t1)-np.mean(t2),
                                      'p_value': p, 'significant': p<0.05})
    else:
        k_stat, k_p, time_pairwise = np.nan, np.nan, []

    # ---- 5e. Ranking (Composite = 0.7*F1_norm + 0.3*Time_norm) ----
    perf = df.groupby('Model').agg({'F1-score': 'mean', 'Time_seconds': 'mean'}).reset_index()
    perf = perf[perf['Model'] != 'Regex']  
    perf['F1_norm'] = perf['F1-score'] / perf['F1-score'].max()
    perf['Time_inv'] = 1 / (perf['Time_seconds'] + 1e-6)
    perf['Time_norm'] = perf['Time_inv'] / perf['Time_inv'].max()
    perf['Composite'] = 0.7 * perf['F1_norm'] + 0.3 * perf['Time_norm']
    ranking = perf.sort_values('Composite', ascending=False)

    # ========================================================================
    # 6. Visualization 
    # ========================================================================
    print("\nGenerating figures...")
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 8

    ref_model = 'Llama-3.1-8B'
    if ref_model not in models:
        print(f"Warning: '{ref_model}' not in models. Using first model as reference.")
        ref_model = models[0]

    # ---- 6a. Main 4-subplot figure4 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.3)

    # Subplot A: Balanced Accuracy (boxplot with significance)
    ax = axes[0,0]
    ba_data = [df[df['Model'] == m]['Balanced_Accuracy'].values for m in models]
    bp = ax.boxplot(ba_data, positions=range(len(models)), widths=0.6, patch_artist=True, labels=models)
    for patch, m in zip(bp['boxes'], models):
        patch.set_facecolor(model_colors[m]); patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black'); median.set_linewidth(2)

    # Friedman p-value
    valid_ba = [arr for arr in ba_data if not np.all(np.isnan(arr)) and len(arr)>=2]
    if len(valid_ba) >= 3:
        try:
            f_stat, f_p = stats.friedmanchisquare(*valid_ba)
            ax.text(0.5, 0.95, f'Friedman P = {f_p:.3f}' + ('*' if f_p<0.05 else ''),
                    transform=ax.transAxes, ha='center', fontsize=9, fontstyle='italic',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        except:
            pass

    # Significance vs reference (Mann-Whitney)
    ref_idx = list(models).index(ref_model)
    ref_vals = ba_data[ref_idx]
    sig_indices = []
    p_vals = []
    for i, m in enumerate(models):
        if i == ref_idx:
            continue
        other_vals = ba_data[i]
        if len(ref_vals)>1 and len(other_vals)>1 and not np.all(np.isnan(ref_vals)) and not np.all(np.isnan(other_vals)):
            u, p = stats.mannwhitneyu(ref_vals, other_vals, alternative='two-sided')
            if p < 0.05:
                sig_indices.append(i)
                p_vals.append(p)
    if sig_indices:
        max_vals = [np.nanpercentile(ba_data[idx], 95) for idx in sig_indices] + [np.nanpercentile(ref_vals, 95)]
        base_y = np.nanmax(max_vals) + 0.05
        for j, i in enumerate(sig_indices):
            p = p_vals[j]
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*'
            y = base_y + j * 0.04
            ax.plot([ref_idx, i], [y, y], 'k-', lw=1)
            ax.plot([ref_idx, ref_idx], [y, y-0.01], 'k-', lw=1)
            ax.plot([i, i], [y, y-0.01], 'k-', lw=1)
            ax.text((ref_idx+i)/2, y+0.01, stars, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='red')

    ax.set_ylabel('Balanced accuracy')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.text(0, 1.02, 'A ', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')

    # Subplot B: F1-score (boxplot with significance)
    ax = axes[0,1]
    f1_data = [df[df['Model'] == m]['F1-score'].dropna().values for m in models]
    f1_data = [arr if len(arr)>0 else np.array([np.nan]) for arr in f1_data]
    bp = ax.boxplot(f1_data, positions=range(len(models)), widths=0.6, patch_artist=True, labels=models)
    for patch, m in zip(bp['boxes'], models):
        patch.set_facecolor(model_colors[m]); patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black'); median.set_linewidth(2)

    valid_f1 = [arr for arr in f1_data if not np.all(np.isnan(arr)) and len(arr)>=2]
    if len(valid_f1) >= 3:
        try:
            f_stat, f_p = stats.friedmanchisquare(*valid_f1)
            ax.text(0.5, 0.95, f'Friedman P = {f_p:.3f}' + ('*' if f_p<0.05 else ''),
                    transform=ax.transAxes, ha='center', fontsize=9, fontstyle='italic',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        except:
            pass

    ref_idx = list(models).index(ref_model)
    ref_vals = f1_data[ref_idx]
    sig_indices = []
    p_vals = []
    for i, m in enumerate(models):
        if i == ref_idx:
            continue
        other_vals = f1_data[i]
        if len(ref_vals)>1 and len(other_vals)>1 and not np.all(np.isnan(ref_vals)) and not np.all(np.isnan(other_vals)):
            u, p = stats.mannwhitneyu(ref_vals, other_vals, alternative='two-sided')
            if p < 0.05:
                sig_indices.append(i)
                p_vals.append(p)
    if sig_indices:
        max_vals = [np.nanpercentile(f1_data[idx], 95) for idx in sig_indices] + [np.nanpercentile(ref_vals, 95)]
        base_y = np.nanmax(max_vals) + 0.05
        for j, i in enumerate(sig_indices):
            p = p_vals[j]
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*'
            y = base_y + j * 0.04
            ax.plot([ref_idx, i], [y, y], 'k-', lw=1)
            ax.plot([ref_idx, ref_idx], [y, y-0.01], 'k-', lw=1)
            ax.plot([i, i], [y, y-0.01], 'k-', lw=1)
            ax.text((ref_idx+i)/2, y+0.01, stars, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='red')

    ax.set_ylabel('F1-score')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.text(0, 1.02, 'B ', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')

    # Subplot C: Time (bar chart with error bars)
    ax = axes[1,0]
    df_time_plot = df[df['Model'] != 'Regex'] if 'Regex' in df['Model'].values else df
    time_means = df_time_plot.groupby('Model')['Time_seconds'].mean().dropna()
    time_stds = df_time_plot.groupby('Model')['Time_seconds'].std().fillna(0)
    if time_means.empty:
        ax.text(0.5, 0.5, 'No valid time data', transform=ax.transAxes, ha='center', va='center')
    else:
        sorted_models = time_means.sort_values().index
        x_pos = np.arange(len(sorted_models))
        means = [time_means[m] for m in sorted_models]
        stds = [time_stds[m] for m in sorted_models]
        colors_sorted = [model_colors[m] for m in sorted_models]

        bars = ax.bar(x_pos, means, yerr=stds, capsize=4, color=colors_sorted, alpha=0.8, error_kw={'ecolor': 'black'})
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_models, rotation=0)
        ax.set_xlabel('Models', fontsize=9)
        ax.set_ylabel('Time (secs)', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Kruskal-Wallis p-value
        groups = [df_time_plot[df_time_plot['Model'] == m]['Time_seconds'].dropna().values for m in sorted_models]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            try:
                k_stat, k_p = stats.kruskal(*groups)
                p_text = f'Kruskal-Wallis P = {k_p:.3f}' + ('***' if k_p < 0.001 else '')
                ax.text(0.5, 0.95, p_text, transform=ax.transAxes, fontstyle='italic', ha='center', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            except:
                pass
    ax.text(0, 1.02, 'C ', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')

    # Subplot D: Composite ranking (bar chart)
    ax = axes[1,1]
    rank_sorted = ranking[ranking['Model'] != 'Regex'].sort_values('Composite')
    bars = ax.bar(rank_sorted['Model'], rank_sorted['Composite'],
                  color=[model_colors[m] for m in rank_sorted['Model']])
    ax.set_xlabel('Models'); ax.set_ylabel('Composite score')
    ax.grid(True, linestyle='--', alpha=0.7)
    for bar, val in zip(bars, rank_sorted['Composite']):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax.text(0, 1.02, 'D ', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')

    # Legend
    legendelements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=model_colors[model], markersize=8, label=model) 
                      for model in models]
    fig.legend(handles=legendelements, loc='lower center', bbox_to_anchor=(0.5, 0.05), 
               ncol=len(models), title='Models', frameon=True)

    for ext in ['png', 'tif', 'pdf', 'svg']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'Figure4.{ext}'), dpi=80, bbox_inches='tight')
    plt.close()

    # ---- 6b. Confusion matrices ----
    fig, axes = plt.subplots(len(models), len(SYMPTOM_LIST), figsize=(len(SYMPTOM_LIST)*2.5+1, len(models)*2.5))
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.25)
    if len(models) == 1: axes = axes.reshape(1, -1)
    if len(SYMPTOM_LIST) == 1: axes = axes.reshape(-1, 1)
    for i, model in enumerate(models):
        for j, symptom in enumerate(SYMPTOM_LIST):
            ax = axes[i, j]
            subset = df[(df['Model'] == model) & (df['Symptom'] == symptom)]
            if subset.empty:
                ax.axis('off'); continue
            tp = subset['TP'].mean(); fp = subset['FP'].mean()
            tn = subset['TN'].mean(); fn = subset['FN'].mean()
            total_true = tp + fn; total_false = fp + tn
            cm = np.array([[tp, fn], [fp, tn]])
            ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max()*1.1, alpha=0.8)
            for row in range(2):
                for col in range(2):
                    count = cm[row, col]
                    pct = count / (total_true if row==0 else total_false) if (total_true if row==0 else total_false) > 0 else 0
                    text = f'{int(round(count))}\n({pct:.2f})'
                    color = 'white' if count > cm.max()*0.6 else 'black'
                    ax.text(col, row, text, ha='center', va='center', color=color, fontsize=12)
            if i == 0:
                ax.set_title(symptom.capitalize(), fontsize=12, fontweight='bold')
            ax.set_xticks([0,1]); ax.set_xticklabels(['Pos','Neg'], fontsize=12)
            ax.set_yticks([0,1]); ax.set_yticklabels(['Pos','Neg'], fontsize=12)
            if j == 0:
                ax.set_ylabel(model, rotation=90, fontsize=12, fontweight='bold', labelpad=15)
            else:
                ax.set_ylabel(''); ax.tick_params(axis='y', labelleft=False)
    fig.supxlabel('Predicted label', fontsize=12); fig.supylabel('True label', fontsize=12)
    for ext in ['png', 'tif', 'pdf', 'svg']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'model_confusion_matrices.{ext}'), dpi=600, bbox_inches='tight')
    plt.close()

    # ---- 6c. Heatmaps (Figure3) ----
    metrics_heat = {'Accuracy': 'Accuracy', 'Precision': 'Precision',
                    'Recall': 'Recall', 'F1-score': 'F1-score'}
    grouped = df.groupby(['Model', 'Symptom'])[list(metrics_heat.keys())].mean().reset_index()
    pivots = {}
    for met in metrics_heat:
        pivots[met] = grouped.pivot(index='Model', columns='Symptom', values=met)
        pivots[met] = pivots[met].reindex(index=models, columns=SYMPTOM_LIST).fillna(0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.6))
    plt.subplots_adjust(wspace=0.2, hspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.15)
    for idx, (met, display) in enumerate(metrics_heat.items()):
        ax = axes.flat[idx]
        data = pivots[met]
        sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm',
                    ax=ax, cbar=True, cbar_kws={'shrink':0.8},
                    linewidths=0.5, linecolor='gray', annot_kws={'size':8})
        ax.set_title('')
        ax.text(0, 1.02, f"{chr(65+idx)} ", transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='bottom', ha='left')
        ax.text(0.05, 1.02, display, transform=ax.transAxes,
                fontsize=9, fontweight='normal', va='bottom', ha='left')
        if idx >= 2: ax.set_xlabel('Symptom', fontsize=9)
        else: ax.set_xlabel('')
        if idx % 2 == 0: ax.set_ylabel('Model', fontsize=9)
        else: ax.set_ylabel('')
        ax.set_xticklabels([s.capitalize() for s in data.columns], rotation=0, ha='center', fontsize=8)
        ax.set_yticklabels(data.index, rotation=0, fontsize=8)
    for ext in ['png', 'tif', 'pdf', 'svg']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'Figure3.{ext}'), dpi=100, bbox_inches='tight')
    plt.close()

    # ---- 6d. Detailed per-symptom plots----
    for symptom in SYMPTOM_LIST:
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle(f'Symptom: {symptom}', fontsize=10, fontweight='bold')
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0])
        symptom_data = df[df['Symptom'] == symptom]
        metrics_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score']
        x = np.arange(len(metrics_plot))
        width = 0.8 / len(models)
        for i, model in enumerate(models):
            mdata = symptom_data[symptom_data['Model'] == model]
            vals = [mdata[met].values[0] if len(mdata[met].values) else 0 for met in metrics_plot]
            x_pos = x + i * (width + 0.02)
            bars = ax1.bar(x_pos, vals, width, label=model, color=model_colors[model], alpha=0.8)
            for bar, val in zip(bars, vals):
                ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                         f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        ax1.set_xlabel('Metrics'); ax1.set_ylabel('Score')
        ax1.set_xticks(x + (len(models)-1)*(width+0.02)/2)
        ax1.set_xticklabels(metrics_plot)
        ax1.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=len(models)//2)
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2 = fig.add_subplot(gs[1]); ax2.axis('off')
        table_data = []
        if symptom in symptom_results:
            for met in metrics_plot:
                if met in symptom_results[symptom]:
                    for comp in symptom_results[symptom][met].get('pairwise', []):
                        if comp['significant']:
                            imp = comp['improvement_pct']
                            imp_str = f"{imp:+.1f}%" if not np.isnan(imp) else "N/A"
                            table_data.append([f"{comp['model1']} vs {comp['model2']}", met, imp_str])
        if table_data:
            table = ax2.table(cellText=table_data[:15],
                              colLabels=['Model Pair', 'Metric', 'Improvement'],
                              cellLoc='center', loc='center', bbox=[0.02,0.1,0.96,0.85])
            table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.5,1.5)
            for i in range(3): table[(0,i)].set_facecolor('#4C72B0'); table[(0,i)].set_text_props(color='white')
            ax2.set_title('Notable Differences (>5% improvement)', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'No notable differences', ha='center', va='center', fontsize=10)
        plt.tight_layout(); plt.subplots_adjust(top=0.9, bottom=0.15)
        for ext in ['png', 'svg']:
            plt.savefig(os.path.join(OUTPUT_DIR, f'symptom_{symptom}_analysis_advanced.{ext}'),
                        dpi=100, bbox_inches='tight')
        plt.close()

    # ========================================================================
    # 7. Save all statistical results to Excel
    # ========================================================================
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, 'statistical_analysis_results_detailed.xlsx')) as writer:
        # Descriptive
        model_stats.to_excel(writer, sheet_name='Model_Descriptive')
        symptom_stats.to_excel(writer, sheet_name='Symptom_Descriptive')

        # Per-symptom Friedman summary
        symptom_summary = []
        for sym, res in symptom_results.items():
            for met, data in res.items():
                symptom_summary.append({
                    'Symptom': sym,
                    'Metric': met,
                    'Friedman_Statistic': data['friedman_stat'],
                    'Friedman_P': data['friedman_p'],
                    'Significant': data['friedman_p'] < 0.05 if not np.isnan(data['friedman_p']) else False
                })
        pd.DataFrame(symptom_summary).to_excel(writer, sheet_name='Symptom_Friedman', index=False)

        # Per-symptom pairwise comparisons (significant only)
        sym_pairwise = []
        for sym, res in symptom_results.items():
            for met, data in res.items():
                for comp in data.get('pairwise', []):
                    if comp['significant']:
                        row = {'Symptom': sym, 'Metric': met, 'Model1': comp['model1'],
                               'Model2': comp['model2'], 'MeanDiff': comp['mean_diff'],
                               'Improvement%': comp['improvement_pct']}
                        sym_pairwise.append(row)
        pd.DataFrame(sym_pairwise).to_excel(writer, sheet_name='Symptom_Pairwise_Sig', index=False)

        # Global tests
        global_df = pd.DataFrame([{
            'Metric': met,
            'FriedmanStatistic': res['friedman_stat'],
            'Friedman_P': res['friedman_p'],
            'Significant': res['friedman_p'] < 0.05 if not np.isnan(res['friedman_p']) else False
        } for met, res in global_results.items()])
        global_df.to_excel(writer, sheet_name='Global_Friedman', index=False)

        # Global pairwise (Mann-Whitney)
        global_pairwise = []
        for met, res in global_results.items():
            for comp in res.get('pairwise', []):
                global_pairwise.append({
                    'Metric': met,
                    'Model1': comp['model1'],
                    'Model2': comp['model2'],
                    'MeanDiff': comp['mean_diff'],
                    'P_value': comp['p_value'],
                    'Significant': comp['significant']
                })
        pd.DataFrame(global_pairwise).to_excel(writer, sheet_name='Global_Pairwise', index=False)

        # Time analysis
        time_summary = pd.DataFrame([{
            'Kruskal_Wallis_Stat': k_stat,
            'P_value': k_p,
            'Significant': k_p < 0.05 if not np.isnan(k_p) else False
        }])
        time_summary.to_excel(writer, sheet_name='Time_Summary', index=False)
        if time_pairwise:
            pd.DataFrame(time_pairwise).to_excel(writer, sheet_name='Time_Pairwise', index=False)

        # Ranking
        ranking.to_excel(writer, sheet_name='Ranking', index=False)

    print(f"\nAll statistical results and figures saved to {OUTPUT_DIR}")

# ============================================================================
# 8. Sensitivity Analysis 
# ============================================================================
def run_sensitivity_analysis():
    print("\n" + "="*60)
    print("Running Sensitivity Analysis")
    print("="*60)

    df = pd.read_excel(os.path.join(OUTPUT_DIR, 'symeva.xlsx'))
    # Time conversion
    def time_to_seconds(tstr):
        if isinstance(tstr, str):
            tstr = tstr.strip()
            try:
                t = datetime.strptime(tstr, "%H:%M:%S.%f")
                return t.hour*3600 + t.minute*60 + t.second + t.microsecond/1e6
            except:
                pass
            try:
                parts = tstr.split(':')
                if len(parts) == 2:
                    m = int(parts[0]); s = float(parts[1])
                    return m*60 + s
            except:
                pass
            try:
                return float(tstr)
            except:
                return np.nan
        else:
            return tstr
    df['Time_seconds'] = df['Time'].apply(time_to_seconds)

    # Exclude Regex
    if 'Regex' in df['Model'].unique():
        df = df[df['Model'] != 'Regex']
    model_avg = df.groupby('Model').agg({'F1-score': 'mean', 'Time_seconds': 'mean'}).reset_index()
    models = sorted(model_avg['Model'].unique())

    f1_values = model_avg['F1-score'].values
    time_values = model_avg['Time_seconds'].values
    f1_norm = f1_values / np.max(f1_values)
    time_inv = 1 / (time_values + 1e-6)
    time_norm = time_inv / np.max(time_inv)

    performance_weights = np.arange(0.60, 0.91, 0.05)
    results = []
    for w_perf in performance_weights:
        w_time = 1 - w_perf
        composite = w_perf * f1_norm + w_time * time_norm
        sorted_indices = np.argsort(composite)[::-1]
        rank_list = [models[i] for i in sorted_indices]
        score_list = [composite[i] for i in sorted_indices]
        row = {'Performance_Weight': f'{w_perf*100:.0f}%', 'Time_Weight': f'{w_time*100:.0f}%'}
        for rank_pos in range(min(7, len(rank_list))):
            row[f'Rank_{rank_pos+1}'] = rank_list[rank_pos]
            row[f'Score_{rank_pos+1}'] = score_list[rank_pos]
        results.append(row)
    sensitivity_df = pd.DataFrame(results)
    sensitivity_df.to_excel(os.path.join(OUTPUT_DIR, 'sensitivity_analysis_ranks.xlsx'), index=False)

    # ---- Ranking stability plot ----
    # Build rank matrix
    n_models = len(models)
    n_weights = len(performance_weights)
    rank_matrix = np.zeros((n_models, n_weights), dtype=int)
    for j, w in enumerate(performance_weights):
        w_perf = w
        w_time = 1 - w_perf
        composite = w_perf * f1_norm + w_time * time_norm
        sorted_indices = np.argsort(composite)[::-1]
        for rank_pos, model_idx in enumerate(sorted_indices):
            model_name = models[model_idx]
            row_idx = models.index(model_name)
            rank_matrix[row_idx, j] = rank_pos + 1

    x_labels = [f'{int(w*100)}%' for w in performance_weights]
    plt.figure(figsize=(10,6))
    colors_plot = sns.color_palette("tab10", n_models)
    for i, model in enumerate(models):
        plt.plot(x_labels, rank_matrix[i, :], marker='o', label=model, color=colors_plot[i], linewidth=2)
    plt.xlabel('Performance Weight', fontsize=12)
    plt.ylabel('Rank (1 = highest)', fontsize=12)
    plt.title('Model Ranking Stability under Different Weighting Schemes', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'sensitivity_ranking_plot.{ext}'), dpi=100, bbox_inches='tight')
    plt.close()

    # ---- Heatmap ----
    plt.figure(figsize=(10,6))
    sns.heatmap(rank_matrix, annot=True, fmt='d', cmap='RdYlGn_r',
                xticklabels=x_labels, yticklabels=models,
                cbar_kws={'label': 'Rank'})
    plt.xlabel('Performance Weight', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('Model Ranks under Different Weighting Schemes (1=best)', fontsize=14)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'sensitivity_ranking_heatmap.{ext}'), dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Sensitivity analysis results saved to {OUTPUT_DIR}")

# ============================================================================
# 9. Main Entry Point
# ============================================================================
if __name__ == "__main__":
    print("="*60)
    print("LLM Evaluation Full Pipeline")
    print("="*60)

    # Step 1: Generate symeva.xlsx
    df_symeva = generate_symeva()
    print("\nSample of symeva.xlsx:")
    print(df_symeva.head(10))

    # Step 2: Run statistical analysis
    run_statistical_analysis()

    # Step 3: Run sensitivity analysis
    run_sensitivity_analysis()

    print("\nPipeline completed successfully!")
    print(f"All outputs are in: {OUTPUT_DIR}")
