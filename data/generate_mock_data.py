"""
Synthetic Data Generator for the Respiratory Disease Prediction Project.
This script generates a mock dataset with the exact same column structure as the original,
allowing reviewers and users to run the main analysis code without accessing real EMR data.
The values are purely fictional and do not represent real epidemiological trends.

The time-series data simulates three major epidemic waves:
    - Wave 1: around April 2023
    - Wave 2: around December 2023 to February 2024
    - Wave 3: around January 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random
import base64

# ====================== Configuration ======================
START_DATE = '2023-01-01'
END_DATE = '2025-06-30'                    # Extended to match original
OUTPUT_PATH = './data/data.xlsx'

# Define the feature names exactly as in your original table
feature_cols_x = [f'x{i}' for i in range(1, 58)]  # x1 to x57
other_cols = ['up', 'down', 'normal', 'sum']

# ====================== Generate Time Series ======================
def generate_time_series():
    """
    Generate mock daily time-series data for nowcasting.
    The base trend is constructed by superimposing three Gaussian peaks
    to mimic the observed epidemic waves.
    """
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    n_days = len(date_range)
    np.random.seed(42)

    # Create a time index in days from start
    time_idx = np.arange(n_days)

    # Baseline level (constant background)
    baseline = 200

    # Define three peaks: (center_date, amplitude, sigma_days)
    peaks = [
        (datetime(2023, 4, 15), 3000, 30),   # Wave 1
        (datetime(2023, 12, 15), 4000, 40),  # Wave 2 (broad)
        (datetime(2025, 1, 15), 5000, 30)    # Wave 3
    ]

    # Convert center dates to day indices
    start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
    peak_indices = []
    for dt, amp, sigma in peaks:
        delta = (dt - start_dt).days
        peak_indices.append((delta, amp, sigma))

    # Construct base trend as sum of Gaussians
    base_trend = np.zeros(n_days)
    for center, amp, sigma in peak_indices:
        gauss = amp * np.exp(-((time_idx - center) ** 2) / (2 * sigma ** 2))
        base_trend += gauss

    # Add a small yearly sinusoidal component (to simulate seasonal variation)
    seasonal = 100 * np.sin(2 * np.pi * (time_idx / 365.25) + np.pi/2)   # peak around winter
    base_trend += seasonal

    # Add baseline and ensure non-negative
    base_trend = np.maximum(0, base_trend + baseline)

    # Add random noise (daily fluctuations)
    noise = np.random.normal(0, 80, n_days)
    daily_base = np.maximum(0, base_trend + noise).astype(int) + 100

    # Generate per-symptom columns (x1~x57)
    data_dict = {'rq': date_range}
    for i, col in enumerate(feature_cols_x):
        proportion = np.random.uniform(0.1, 0.9)
        # Weekly pattern (7-day cycle)
        weekly_effect = 50 * np.sin(np.linspace(0, 2 * np.pi, 7))
        col_data = (daily_base * proportion +
                    np.random.normal(0, 30, n_days) +
                    np.tile(weekly_effect, int(np.ceil(n_days / 7)))[:n_days])
        data_dict[col] = np.maximum(0, col_data).astype(int)

    # Aggregated columns (up, down, normal, sum)
    base_sum = daily_base * 3   # total cases roughly 3x the base
    data_dict['up'] = (base_sum * np.random.uniform(0.4, 0.6, n_days) +
                       np.random.normal(0, 50, n_days)).astype(int)
    data_dict['down'] = (base_sum * np.random.uniform(0.2, 0.3, n_days) +
                         np.random.normal(0, 30, n_days)).astype(int)
    data_dict['normal'] = (base_sum * np.random.uniform(0.1, 0.2, n_days) +
                           np.random.normal(0, 20, n_days)).astype(int)
    data_dict['sum'] = np.maximum(0,
        data_dict['up'] + data_dict['down'] + data_dict['normal'] +
        np.random.normal(0, 10, n_days).astype(int)
    )

    # Ensure non-negative for all columns
    for col in ['up', 'down', 'normal', 'sum']:
        data_dict[col] = np.maximum(0, data_dict[col])

    # Create DataFrame and save
    df_mock = pd.DataFrame(data_dict)
    os.makedirs('./data', exist_ok=True)
    df_mock.to_excel(OUTPUT_PATH, index=False)
    print(f"Mock time-series data saved to {OUTPUT_PATH}, shape: {df_mock.shape}")
    print(f"Date range: {df_mock['rq'].min()} to {df_mock['rq'].max()}")
    print("Peaks are designed around 2023-04, 2023-12, and 2025-01.")


# ====================== Generate LLM Evaluation Data ======================
def generate_llm_data(n=1000, symptom_rates=None):
    """
    Generate mock chief complaints and gold-standard labels for LLM evaluation.
    symptom_rates: dict with keys matching the six symptoms in order:
                   fever, cough, sore_throat, chest_pain, myalgia, dyspnea.
    The output files:
      - train1000.csv: contains 'yljgdm', 'jzlsh', 'jzksrq', 'zs' (chief complaint)
      - train1000mark.csv: adds six label columns 'msym1'..'msym6' in the fixed order.
    """
    if symptom_rates is None:
        symptom_rates = {
            'fever': 0.167,
            'cough': 0.228,
            'sore_throat': 0.077,
            'dyspnea': 0.031,
            'chest_pain': 0.028,
            'myalgia': 0.003
        }

    # Fixed order: msym1=fever, msym2=cough, msym3=sore_throat,
    # msym4=chest_pain, msym5=myalgia, msym6=dyspnea
    symptom_keys = ['fever', 'cough', 'sore_throat', 'chest_pain', 'myalgia', 'dyspnea']
    # Chinese synonyms for each symptom (to diversify chief complaint text)
    symptom_synonyms = {
        'fever': ['发热', '发烧', '体温升高'],
        'cough': ['咳嗽', '干咳', '咳痰'],
        'sore_throat': ['咽痛', '咽部不适', '咽干', '喉咙痛'],
        'chest_pain': ['胸痛', '胸部不适'],
        'myalgia': ['肌肉酸痛', '全身酸痛', '肌痛'],
        'dyspnea': ['呼吸困难', '气短', '喘息','胸闷']
    }
    durations = ['1天', '2天', '3天', '1周']

    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    data_mark = []  # for train1000mark.csv (with labels)
    data_raw = []   # for train1000.csv (without labels)

    for _ in range(n):
        # 1. Generate symptom labels
        labels = {}
        for sym in symptom_keys:
            labels[sym] = 1 if np.random.random() < symptom_rates[sym] else 0

        # 2. Build chief complaint text using synonyms
        symptom_parts = []
        for sym in symptom_keys:
            if labels[sym] == 1:
                # Randomly pick one synonym for this symptom
                synonym = random.choice(symptom_synonyms[sym])
                symptom_parts.append(synonym)
        if symptom_parts:
            text = '、'.join(symptom_parts) + random.choice(durations)
        else:
            text = '无明显不适'

        # 3. Generate metadata columns
        yljgdm = str(np.random.randint(100000, 999999))
        # jzlsh: 12 random bytes -> 16 base64 characters
        rand_bytes = random.randbytes(12)
        jzlsh = base64.b64encode(rand_bytes).decode('utf-8')[:16]
        # Random visit date between 2023-01-01 and 2025-06-30
        start_dt = datetime(2023, 1, 1)
        end_dt = datetime(2025, 6, 30)
        days_diff = (end_dt - start_dt).days
        random_days = random.randint(0, days_diff)
        jzksrq = (start_dt + timedelta(days=random_days)).strftime('%Y/%m/%d')

        row_common = {
            'yljgdm': yljgdm,
            'jzlsh': jzlsh,
            'jzksrq': jzksrq,
            'zs': text
        }
        data_raw.append(row_common)

        # Create labeled row with msym1..msym6 (order as defined)
        row_mark = row_common.copy()
        # msym1=fever, msym2=cough, msym3=sore_throat, msym4=chest_pain, msym5=myalgia, msym6=dyspnea
        row_mark['msym1'] = labels['fever']
        row_mark['msym2'] = labels['cough']
        row_mark['msym3'] = labels['sore_throat']
        row_mark['msym4'] = labels['chest_pain']
        row_mark['msym5'] = labels['myalgia']
        row_mark['msym6'] = labels['dyspnea']
        data_mark.append(row_mark)

    # Convert to DataFrames
    df_raw = pd.DataFrame(data_raw)
    df_mark = pd.DataFrame(data_mark)

    # Save to CSV
    os.makedirs('./data', exist_ok=True)
    raw_path = './data/train1000.csv'
    mark_path = './data/train1000mark.csv'
    df_raw.to_csv(raw_path, index=False, encoding='utf-8-sig')
    df_mark.to_csv(mark_path, index=False, encoding='utf-8-sig')
    print(f"Mock LLM evaluation data generated:")
    print(f"  - {raw_path} (chief complaints + metadata)")
    print(f"  - {mark_path} (with gold-standard labels msym1..msym6)")
    print(f"  Shape: {len(df_raw)} rows, {len(df_raw.columns)} columns (raw); {len(df_mark.columns)} columns (marked)")
    return df_raw, df_mark


# ====================== Main ======================
if __name__ == "__main__":
    # 1. Generate time-series data
    generate_time_series()
    # 2. Generate LLM evaluation mock data
    generate_llm_data(n=1000)
    print("All synthetic data generation completed.")