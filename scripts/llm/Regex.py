"""
Regex-based symptom extraction from chief complaints.
Reads a CSV file containing 'jzksrq', 'yljgdm', 'jzlsh', 'zs' columns,
applies pattern matching for six symptoms, and saves the result as 'train_zz.csv'
with columns 'yljgdm', 'jzlsh', 'sym1'...'sym6' (aligned with LLM evaluation format).
"""

import pandas as pd
import re
import datetime
import os
import argparse

# ====================== Configuration ======================
# Default paths (can be overridden by command-line arguments)
DEFAULT_INPUT = './data/train1000.csv'          # Contains chief complaints
DEFAULT_OUTPUT = './data/train_zz.csv'          # Output regex predictions

# Symptom mapping: column name in output -> (positive_pattern, negative_pattern)
SYMPTOM_PATTERNS = {
    'sym1': {  # fever
        'positive': re.compile(r"发热|发烧|体温升高"),
        'negative': re.compile(r"无发热|无发烧|无体温升高|不伴发热|不伴发烧|不发热|不发烧|不体温升高")
    },
    'sym2': {  # cough
        'positive': re.compile(r"咳嗽|干咳|轻咳"),
        'negative': re.compile(r"无咳嗽|不咳嗽|无干咳|不干咳|无轻咳|不轻咳")
    },
    'sym3': {  # sore throat
        'positive': re.compile(r"咽痛|喉咙痛|咽喉疼痛|喉咙疼痛|咽部不适"),
        'negative': re.compile(r"无咽痛|无喉咙痛|无咽喉疼痛|无喉咙疼痛|无咽部不适|不伴咽痛|不伴喉咙痛|不伴咽喉疼痛|不咽痛|不喉咙痛|不咽喉疼痛|不喉咙疼痛|不咽部不适")
    },
    'sym4': {  # chest pain
        'positive': re.compile(r"胸痛|胸部疼痛"),
        'negative': re.compile(r"无胸痛|无胸部疼痛|不伴胸痛|不伴胸部疼痛|不胸痛|不胸部疼痛")
    },
    'sym5': {  # myalgia
        'positive': re.compile(r"全身肌肉疼痛|肌痛|肌肉疼痛|肌肉痛|浑身痛|全身痛"),
        'negative': re.compile(r"无全身肌肉疼痛|无肌痛|无肌肉疼痛|无肌肉痛|无浑身痛|无全身痛|不伴肌痛|不伴肌肉疼痛|不伴肌肉痛|不伴浑身痛|不伴全身痛|不肌痛|不肌肉疼痛|不肌肉痛|不浑身痛|不全身痛")
    },
    'sym6': {  # dyspnea
        'positive': re.compile(r"喘息|呼吸困难|呼吸急促|呼吸不畅|气短|气促|气喘"),
        'negative': re.compile(r"无喘息|无呼吸困难|无呼吸急促|无呼吸不畅|无气短|无气促|无气喘|不伴呼吸困难|不伴呼吸急促|不伴呼吸不畅|不伴气短|不伴气促|不伴气喘|不呼吸困难|不呼吸急促|不呼吸不畅|不气短|不气促|不气喘")
    }
}

# ====================== Extraction Function ======================
def check_symptom(text, positive_pattern, negative_pattern):
    """Return 1 if symptom is present, 0 otherwise."""
    return 1 if positive_pattern.search(text) and not negative_pattern.search(text) else 0

def extract_symptoms(input_path, output_path, verbose=False):
    """
    Read chief complaints from CSV, extract symptom indicators, and save to CSV.
    The input CSV must contain columns: 'jzksrq', 'yljgdm', 'jzlsh', 'zs'.
    The output CSV will contain 'yljgdm', 'jzlsh', and 'sym1'..'sym6'.
    """
    # Read data
    df = pd.read_csv(input_path, usecols=['jzksrq', 'yljgdm', 'jzlsh', 'zs'])
    print(f"Loaded {len(df)} records from {input_path}")

    # Ensure sorting (optional)
    df['jzdate'] = pd.to_datetime(df['jzksrq'].str[:10])
    df.sort_values('jzdate', inplace=True)
    df = df[['jzksrq', 'yljgdm', 'jzlsh', 'zs']]

    # Initialize symptom columns
    for sym in SYMPTOM_PATTERNS.keys():
        df[sym] = 0

    # Apply patterns
    total = len(df)
    for idx, row in df.iterrows():
        if verbose and (idx + 1) % 100 == 0:
            print(f"Processing {idx + 1}/{total}")
        text = row['zs']
        for sym, patterns in SYMPTOM_PATTERNS.items():
            df.at[idx, sym] = check_symptom(text, patterns['positive'], patterns['negative'])

    # Ensure integer type
    for sym in SYMPTOM_PATTERNS.keys():
        df[sym] = df[sym].astype(int)

    # Keep only required columns: yljgdm, jzlsh, sym1..sym6
    output_cols = ['yljgdm', 'jzlsh'] + list(SYMPTOM_PATTERNS.keys())
    df_out = df[output_cols]

    # Save to CSV
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Regex predictions saved to {output_path}")

    return df_out

# ====================== Main ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regex symptom extraction from chief complaints.")
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT,
                        help=f"Input CSV file (default: {DEFAULT_INPUT})")
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help=f"Output CSV file (default: {DEFAULT_OUTPUT})")
    parser.add_argument('--verbose', action='store_true', help="Print progress")
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    extract_symptoms(args.input, args.output, args.verbose)
    end_time = datetime.datetime.now()
    print(f"Elapsed time: {end_time - start_time}")