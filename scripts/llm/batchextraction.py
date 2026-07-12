"""
Fixed-parameter batch extraction using Ollama.
Uses a single instruction file (instruction2.txt) with fixed generation parameters:
    temperature = 0.2, top_k = 10, top_p = 0.7
Outputs: {model_name}.csv and {model_name}.log in the specified output directory.
"""

import argparse
import pandas as pd
from ollama import Client
import json
import re
import datetime
import os
import logging
import sys


def setup_logger(log_path: str, console_level: int = logging.INFO, file_level: int = logging.DEBUG) -> logging.Logger:
    """
    Configure a logger that writes to both a file and the console.

    Args:
        log_path: Path to the log file.
        console_level: Logging level for console output.
        file_level: Logging level for file output.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    # File handler
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(file_level)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def main():
    # ---------- Argument Parsing ----------
    parser = argparse.ArgumentParser(
        description="Batch symptom extraction with fixed parameters (instruction2.txt, temp=0.2, top_k=10, top_p=0.7)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gemma3:1b,gemma3:4b,llama3.2:3b,llama3.1:8b,deepseek-r1:7b,qwen3:1.7b,qwen3:8b",
        help="Comma-separated list of model names (as used in Ollama)"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="instruction2.txt",
        help="Path to the instruction file (fixed to instruction2.txt by default)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/train1000.csv",
        help="Path to the input dataset CSV"
    )
    parser.add_argument(
        "--num_symps",
        type=int,
        default=7,
        help="Number of symptom categories"
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="ZS",
        help="Column name containing the text to extract symptoms from"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for result CSVs and log files"
    )
    args = parser.parse_args()

    # Parse model list
    models = [m.strip() for m in args.models.split(',') if m.strip()]

    # Validate input files
    if not os.path.isfile(args.instruction):
        print(f"Error: Instruction file not found: {args.instruction}")
        sys.exit(1)
    if not os.path.isfile(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Fixed generation parameters
    TEMPERATURE = 0.2
    TOP_K = 10
    TOP_P = 0.7

    # Read the instruction once
    with open(args.instruction, "r", encoding="utf8") as fp:
        instruction_content = fp.read()

    # Initialize Ollama client
    client = Client(host="http://localhost:11434")

    # Process each model sequentially
    for model in models:
        model_safe = model.replace(':', '-')
        csv_path = os.path.join(args.output_dir, f"{model_safe}.csv")
        log_path = os.path.join(args.output_dir, f"{model_safe}.log")

        logger = setup_logger(log_path, console_level=logging.INFO, file_level=logging.DEBUG)

        logger.info(f"===== Starting model: {model} =====")
        logger.info(f"Instruction file: {args.instruction}")
        logger.info(f"Parameters: temperature={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}")

        try:
            # Load dataset
            df = pd.read_csv(args.dataset)
            for col in range(args.num_symps):
                df[col] = 0

            start_time = datetime.datetime.now()
            total_samples = len(df)

            # Process each sample
            for idx in range(total_samples):
                content = df.loc[idx, args.feature]
                logger.debug(f"Processing sample {idx}: {content[:50]}...")

                try:
                    response = client.chat(
                        model=model,
                        messages=[
                            {"role": "system", "content": instruction_content},
                            {"role": "user", "content": content},
                        ],
                        stream=False,
                        options={
                            "temperature": TEMPERATURE,
                            "top_k": TOP_K,
                            "top_p": TOP_P,
                        },
                    )

                    # Extract JSON from markdown code block
                    json_matches = re.findall(
                        r"```json\n{0,1}(.*?)\n{0,1}```",
                        response.message.content,
                        re.DOTALL
                    )

                    if json_matches:
                        json_content = json_matches[0]
                        symptom_ids = re.findall(r'"symp":\s{0,1}(\d+)', json_content)
                        if symptom_ids:
                            for sid in symptom_ids:
                                df.loc[idx, int(sid)] = 1
                            logger.debug(f"Sample {idx} updated with symptoms: {symptom_ids}")
                        else:
                            logger.warning(f"Sample {idx}: No symptom IDs found in JSON")
                    else:
                        logger.warning(f"Sample {idx}: No JSON block found in response")

                except Exception as e:
                    logger.error(f"Sample {idx} processing error: {e}")

            # Save results
            df.to_csv(csv_path, index=False)

            # Timing statistics
            elapsed = datetime.datetime.now() - start_time
            total_seconds = elapsed.total_seconds()
            # Format total time as H:M:S.ffffff
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = total_seconds % 60
            time_str = f"{hours}:{minutes:02d}:{seconds:06.3f}".replace('.', '.').ljust(15, '0')[:15]
            logger.info(f"Results saved to: {csv_path}")
            logger.info(f"Total runtime: {time_str}")   
            
            # Write total seconds and sample count to a separate file
            time_file = os.path.join(args.output_dir, f"{model_safe}_time.txt")
            with open(time_file, "w") as f:
                f.write(f"{time_str}\n")
                f.write(f"{total_samples}\n")

        except Exception as e:
            logger.error(f"Model {model} failed: {e}", exc_info=True)
        finally:
            logger.info(f"===== Model {model} finished =====\n")


if __name__ == "__main__":
    main()