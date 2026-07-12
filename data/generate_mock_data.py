
"""
Synthetic Data Generator for the Respiratory Disease Prediction Project.
This script generates a mock dataset with the exact same column structure as the original,
allowing reviewers and users to run the main analysis code without accessing real EMR data.
The values are purely fictional and do not represent real epidemiological trends.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ====================== Configuration ======================
START_DATE = '2023-01-01'
END_DATE = '2025-06-01'
OUTPUT_PATH = './data/data.xlsx'

# Define the feature names exactly as in your original table
# (Based on the header you provided: rq, x1~x57, up, down, normal, xg, lg, brk, sum)
feature_cols_x = [f'x{i}' for i in range(1, 58)]  # x1 to x57
other_cols = ['up', 'down', 'normal', 'sum']

# ====================== Generate Time Series ======================
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
n_days = len(date_range)

# Set random seed for reproducibility
np.random.seed(42)

# ====================== Generate Mock Data ======================
# 1. Generate base patterns: a sinusoidal yearly trend + random noise
# This mimics the seasonality of respiratory diseases (e.g., influenza)
base_trend = 500 + 300 * np.sin(np.linspace(0, 4 * np.pi, n_days))  # 2-year cycle
trend_noise = np.random.normal(0, 150, n_days)
daily_base = np.maximum(0, base_trend + trend_noise).astype(int) + 100

# 2. Generate individual symptom columns (x1~x57)
# They should correlate with the daily_base but with varying proportions
data_dict = {'rq': date_range}

for i, col in enumerate(feature_cols_x):
    # Each symptom has a different baseline proportion (0.05 to 0.95) plus noise
    proportion = np.random.uniform(0.1, 0.9)
    # Add some column-specific weekly periodicity
    weekly_effect = 50 * np.sin(np.linspace(0, 2 * np.pi, 7))  # 7-day cycle
    
    # Generate values and ensure non-negative integers
    col_data = (daily_base * proportion + np.random.normal(0, 30, n_days) + 
                np.tile(weekly_effect, int(np.ceil(n_days/7)))[:n_days])
    data_dict[col] = np.maximum(0, col_data).astype(int)

# 3. Generate aggregated columns (up, down, normal, xg, lg, brk, sum)
# Make sure sum = up + down + normal (or at least logical)
base_sum = daily_base * 3  # total cases roughly 3x the base

data_dict['up'] = (base_sum * np.random.uniform(0.4, 0.6, n_days) + np.random.normal(0, 50, n_days)).astype(int)
data_dict['down'] = (base_sum * np.random.uniform(0.2, 0.3, n_days) + np.random.normal(0, 30, n_days)).astype(int)
data_dict['normal'] = (base_sum * np.random.uniform(0.1, 0.2, n_days) + np.random.normal(0, 20, n_days)).astype(int)

# Ensure 'sum' is the sum of ups, downs, and normals for logical consistency
data_dict['sum'] = data_dict['up'] + data_dict['down'] + data_dict['normal'] + np.random.normal(0, 10, n_days).astype(int)
data_dict['sum'] = np.maximum(0, data_dict['sum'])

# ====================== Save to Excel ======================
df_mock = pd.DataFrame(data_dict)

# Ensure the data directory exists
import os
os.makedirs('./data', exist_ok=True)

df_mock.to_excel(OUTPUT_PATH, index=False)
print(f"Mock data successfully generated at: {OUTPUT_PATH}")
print(f"Shape: {df_mock.shape}, Date range: {df_mock['rq'].min()} to {df_mock['rq'].max()}")
print("Column names (first 10):", df_mock.columns[:10].tolist())