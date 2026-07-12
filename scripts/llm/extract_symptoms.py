#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Symptom extraction from chief complaints using Ollama.
Reads database records, filters invalid chief complaints, applies keyword screening,
and calls an Ollama model to label 6 target symptoms.

Usage:
    python extract_symptoms.py --config config.json

The config file must contain:
    - database: host, port, user, password, database
    - table: name of the source table
    - date_column: column name for visit date
    - text_column: column name for chief complaint text
    - id_columns: list of ID columns (optional)
    - output_dir: where to save results
    - result_file_prefix: prefix for output CSV files
    - instruction_file: path to the system prompt file
    - model_name: Ollama model name (e.g., 'llama3.1:8b')
    - temperature, top_p, top_k: inference parameters
    - min_records_per_day: minimum number of records after filtering to process a day
    - specific_keywords: list of Chinese keywords for pre‑filtering
    - target_symptoms: list of symptom names (6 in order)
"""

import os
import sys
import json
import re
import glob
import argparse
from datetime import datetime, timedelta

import pymysql
import pandas as pd
from ollama import Client


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_db_connection(cfg):
    """Create and return a MySQL connection."""
    db = cfg['database']
    try:
        conn = pymysql.connect(
            host=db['host'],
            port=db['port'],
            user=db['user'],
            password=db['password'],
            database=db['database'],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn
    except Exception as e:
        print(f"Database connection failed: {str(e)}")
        return None


def get_days_in_month(year, month):
    """Return a list of all days in a given month as 'YYYY-MM-DD' strings."""
    first = datetime(year, month, 1)
    if month == 12:
        next_first = datetime(year + 1, 1, 1)
    else:
        next_first = datetime(year, month + 1, 1)
    days = []
    cur = first
    while cur < next_first:
        days.append(cur.strftime('%Y-%m-%d'))
        cur += timedelta(days=1)
    return days


def get_processed_dates(cfg):
    """
    Scan the output directory for existing result files.
    Supports both single‑day files (hxdzzYYMMDD.csv) and monthly files (hxdzzYYMM.csv).
    Returns a sorted list of processed dates (YYYY-MM-DD).
    """
    output_dir = cfg['output_dir']
    prefix = cfg.get('result_file_prefix', 'hxdzz')
    pattern = os.path.join(output_dir, f"{prefix}*.csv")
    files = glob.glob(pattern)
    processed = set()

    if not files:
        print(f"No result files found in {output_dir}")
        return sorted(processed)

    print(f"Found {len(files)} result files:")
    for fpath in files:
        fname = os.path.basename(fpath)
        date_part = fname.replace(prefix, '').replace('.csv', '')

        # Monthly file: 4 digits (YYMM)
        if len(date_part) == 4 and date_part.isdigit():
            yy = int(date_part[:2])
            mm = int(date_part[2:4])
            if not (1 <= mm <= 12):
                print(f"  Skipping invalid monthly file: {fname}")
                continue
            year = 2000 + yy
            try:
                df = pd.read_csv(fpath)
                date_col = cfg.get('date_column', 'jzdate')
                if date_col not in df.columns:
                    print(f"  Monthly file {fname} lacks date column, treating as full month")
                    days = get_days_in_month(year, mm)
                else:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
                    df = df.dropna(subset=[date_col])
                    month_start = f"{year}-{mm:02d}-01"
                    # Last day of month
                    if mm == 12:
                        next_month = 1
                        next_year = year + 1
                    else:
                        next_month = mm + 1
                        next_year = year
                    last_day = (datetime(next_year, next_month, 1) - timedelta(days=1)).strftime('%Y-%m-%d')
                    file_dates = df[(df[date_col] >= month_start) & (df[date_col] <= last_day)][date_col].unique()
                    days = [str(d) for d in file_dates]
                processed.update(days)
                print(f"  Monthly file: {fname} -> {len(days)} days ({year}-{mm:02d})")
            except Exception as e:
                print(f"  Warning: reading {fname} failed, using full month: {e}")
                processed.update(get_days_in_month(year, mm))

        # Single‑day file: 6 digits (YYMMDD)
        elif len(date_part) == 6 and date_part.isdigit():
            yy = int(date_part[:2])
            mm = int(date_part[2:4])
            dd = int(date_part[4:6])
            try:
                full_date = datetime(2000 + yy, mm, dd).strftime('%Y-%m-%d')
                processed.add(full_date)
                print(f"  Single-day file: {fname} -> {full_date}")
            except ValueError:
                print(f"  Skipping invalid date in {fname}")
                continue
        else:
            print(f"  Skipping unknown file pattern: {fname}")

    processed = sorted(processed)
    print(f"Total processed dates: {len(processed)} (first: {processed[0] if processed else 'none'}, last: {processed[-1] if processed else 'none'})")
    return processed


def get_available_dates(cfg):
    """
    Query the database for all distinct dates that have any records.
    Filter to dates between start_date (default 2023-01-01) and today.
    """
    conn = get_db_connection(cfg)
    if not conn:
        return []

    date_col = cfg['date_column']
    table = cfg['table']
    text_col = cfg['text_column']
    try:
        with conn.cursor() as cursor:
            query = f"""
            SELECT DISTINCT DATE({date_col}) AS date
            FROM {table}
            WHERE {date_col} IS NOT NULL
              AND {text_col} IS NOT NULL
            ORDER BY date ASC
            """
            cursor.execute(query)
            results = cursor.fetchall()
            all_dates = [str(row['date']) for row in results]

            start_date = cfg.get('start_date', '2023-01-01')
            today = datetime.today().strftime('%Y-%m-%d')
            filtered = [d for d in all_dates if start_date <= d <= today]

            print(f"Database query: {len(all_dates)} raw dates, {len(filtered)} within {start_date} - {today}")
            if filtered:
                print(f"  Earliest: {filtered[0]}, latest: {filtered[-1]}")
            return filtered
    except Exception as e:
        print(f"Database query failed: {str(e)}")
        return []
    finally:
        if conn:
            conn.close()


def get_dates_to_process(cfg):
    """Compare available dates with already processed ones and return the difference."""
    available = get_available_dates(cfg)
    processed = get_processed_dates(cfg)
    if not available:
        print("No available dates found.")
        return []

    to_process = [d for d in available if d not in processed]
    print("\n" + "="*50)
    print(f"Date comparison:")
    print(f"  Available dates: {len(available)}")
    print(f"  Already processed: {len(processed)}")
    print(f"  To process: {len(to_process)}")
    if to_process:
        print(f"  First 10: {to_process[:10]}")
    print("="*50 + "\n")
    return to_process


def get_output_file_path(target_date, cfg):
    """Generate file path for a single day's results."""
    date_part = target_date.replace('-', '')[2:]   # YYYY-MM-DD -> YYMMDD
    prefix = cfg.get('result_file_prefix', 'hxdzz')
    fname = f"{prefix}{date_part}.csv"
    return os.path.join(cfg['output_dir'], fname)


def process_single_date(target_date, cfg, ollama_client, instruction_text):
    """
    Fetch data for one day, filter invalid chief complaints, apply keyword filter,
    and run Ollama to extract 6 symptoms.
    """
    print("\n" + "="*60)
    print(f"Processing date: {target_date}")
    print("="*60)

    conn = get_db_connection(cfg)
    if not conn:
        return

    date_col = cfg['date_column']
    table = cfg['table']
    text_col = cfg['text_column']
    id_cols = cfg.get('id_columns', [])

    try:
        with conn.cursor() as cursor:
            query = f"""
            SELECT *
            FROM {table}
            WHERE DATE({date_col}) = %s
              AND {text_col} IS NOT NULL
            """
            cursor.execute(query, (target_date,))
            results = cursor.fetchall()
            if not results:
                print(f"No records for {target_date}")
                return
            df = pd.DataFrame(results)
            print(f"Fetched {len(df)} raw records.")
    except Exception as e:
        print(f"Query failed: {str(e)}")
        return
    finally:
        if conn:
            conn.close()

    # Step 1: Filter invalid chief complaints
    invalid_values = ["-", "--", "/", "无", "诉", "主诉", "1", "111", ".", "=", "。", ":-", ":、",
                      "未填", "未说明", "未填写", "现", "不详", "未写", "主诉不详", "·", " "]
    pattern_invalid = '^(' + '|'.join(re.escape(v) for v in invalid_values) + ')$'
    df[text_col] = df[text_col].astype(str)
    df_valid = df[~df[text_col].str.fullmatch(pattern_invalid, case=False, na=False)]

    print(f"After removing invalid chief complaints: {len(df_valid)} records (removed {len(df)-len(df_valid)})")

    min_records = cfg.get('min_records_per_day', 1000)
    if len(df_valid) < min_records:
        print(f"Less than {min_records} valid records, skipping.")
        return

    # Step 2: Keyword filtering
    keywords = cfg.get('specific_keywords', ['咳','痰','痛','呼吸','咽','热','鼻','涕','喘','寒','胸','气','慌','烧','喉','咯'])
    pattern = '|'.join(keywords)
    df_filtered = df_valid[df_valid[text_col].str.contains(pattern, case=False, na=False)]
    print(f"After keyword filter: {len(df_filtered)} records (kept {len(df_filtered)/len(df_valid):.2%})")

    if len(df_filtered) == 0:
        print("No records matched keywords, skipping.")
        return

    # Prepare output file
    output_file = get_output_file_path(target_date, cfg)
    if os.path.exists(output_file):
        print(f"Output file {os.path.basename(output_file)} already exists, skipping.")
        return

    os.makedirs(cfg['output_dir'], exist_ok=True)
    header = True
    print(f"Writing results to: {output_file}")

    # Map symptom names to output columns
    symptoms = cfg.get('target_symptoms', ['fever', 'cough', 'sore_throat', 'chest_pain', 'myalgia', 'dyspnea'])
    # Build the system instruction: we read it from file
    # The instruction should contain the numbering (1..6) and the rules
    # We'll keep the user prompt dynamic: it uses the text content.

    # Process each row
    total = len(df_filtered)
    print(f"Starting model inference for {total} records...")
    for idx, (_, row) in enumerate(df_filtered.iterrows()):
        text = row[text_col]
        # Build user prompt (same as before, but now it includes the list of 6 symptoms)
        user_prompt = f"""
        根据患者主诉识别症状,目标症状共计6种:1发热,2咳嗽,3咽部不适、咽干或咽痛,4胸痛,5肌肉酸痛,6呼吸困难。
        需特别注意以下事项:
        严格按照主诉内容来识别症状,只有当主诉中准确出现了目标症状本身的表述,或者如'发烧'（对应'发热'）、'喉痛'（对应'咽干或咽痛'）、'全身酸痛、肌痛'（对应'肌肉酸痛'）等明确规定的标准同义词时,才可确定识别出某个症状。
        绝对不允许依据模糊的表述、推测、联想或疾病名称去推断症状。以下是一些示例情况,供参考以明确界限：
        1.如'腰痛、腰腹痛'等均不可识别为'胸痛'。只有'急性支气管炎,急性上呼吸道感染,鼻炎'等诊断性的表述，而没有具体症状的表述也不可推测症状，应按照无目标症状处理。
        2.手痛、肩痛、臀部疼痛、大腿疼痛、关节疼痛、腋下疼痛、伤处疼痛均不可识别为“肌肉酸痛”。
        3.胸闷、喘息、气短、端坐呼吸、张口呼吸等为呼吸困难的常用表达,均识别为呼吸困难。
        4.'热退、不伴发热'等否定表述不应该识别为'发热'。
        5.若主诉仅为'病史同前、检查'等这类未提及任何目标症状及其标准同义词的表述,即便从医学常理推测可能存在相关症状,也绝不能识别出任何目标症状。
        6.若主诉提到了一些与目标症状看似相关但并非标准表述或同义词的内容,比如'感觉身体有点异样,但没具体说哪里不舒服',同样不能识别出任何目标症状。 

        患者主诉文字如下：{text}
        """

        try:
            response = ollama_client.chat(
                model=cfg['model_name'],
                messages=[
                    {"role": "system", "content": instruction_text},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": cfg.get('temperature', 0.2),
                    "top_p": cfg.get('top_p', 0.7),
                    "top_k": cfg.get('top_k', 10),
                }
            )
            result = response['message']['content'].strip()
            # Extract numbers from result
            numbers = re.findall(r'\d+', result)
            # For 6 symptoms, we expect numbers 1-6
            symp_flags = {s: 0 for s in symptoms}  # initialize all to 0
            # We'll map number to symptom name: 1->fever, 2->cough, 3->sore_throat, 4->chest_pain, 5->myalgia, 6->dyspnea
            # We assume the order in symptoms list matches the numbering.
            for num in numbers:
                num_int = int(num)
                if 1 <= num_int <= len(symptoms):
                    symp_flags[symptoms[num_int-1]] = 1

            # Build output row
            out_row = {
                cfg.get('date_column', 'jzdate'): target_date,
                text_col: text,
                'model_result': result
            }
            # Add symptom columns
            for s in symptoms:
                out_row[s] = symp_flags[s]

            row_df = pd.DataFrame([out_row])
            row_df.to_csv(output_file, mode='a', index=False, header=header)
            header = False

            # Progress every 10 rows
            if (idx+1) % 10 == 0 or (idx+1) == total:
                print(f"Progress: {idx+1}/{total} | Last output: {result}")

        except Exception as e:
            print(f"Error on record {idx+1}: {str(e)}")
            continue

    print(f"Finished processing {target_date}. File size: {os.path.getsize(output_file)/1024:.2f} KB")
    print("="*60)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract symptoms from chief complaints using Ollama.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to JSON configuration file.')
    parser.add_argument('--dates', type=str, nargs='*',
                        help='Optional list of specific dates (YYYY-MM-DD) to process; if omitted, process all new dates.')
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Read instruction file
    instr_file = cfg.get('instruction_file', 'instruction.txt')
    if not os.path.exists(instr_file):
        print(f"Instruction file {instr_file} not found.")
        sys.exit(1)
    with open(instr_file, 'r', encoding='utf-8') as f:
        instruction_text = f.read()

    # Initialize Ollama client
    ollama_host = cfg.get('ollama_host', 'http://localhost:11434')
    client = Client(host=ollama_host)

    # Determine dates to process
    if args.dates:
        dates_to_process = [d for d in args.dates if re.match(r'\d{4}-\d{2}-\d{2}', d)]
        if not dates_to_process:
            print("No valid dates provided.")
            return
        print(f"Processing specified dates: {dates_to_process}")
    else:
        dates_to_process = get_dates_to_process(cfg)

    if not dates_to_process:
        print("No dates to process.")
        return

    # Process each date
    total_start = datetime.now()
    for i, date in enumerate(dates_to_process, 1):
        print(f"\n--- Processing {i}/{len(dates_to_process)}: {date} ---")
        try:
            process_single_date(date, cfg, client, instruction_text)
        except Exception as e:
            print(f"Failed on {date}: {str(e)}")
            continue

    elapsed = datetime.now() - total_start
    print(f"\nAll done. Total time: {elapsed}")
    print(f"Results saved in: {cfg['output_dir']}")


if __name__ == '__main__':
    main()