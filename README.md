# ARIDs Nowcasting: Pipeline for Acute Respiratory Infectious Diseases Nowcasting Using LLM-Extracted Symptoms

Code associated with a complete pipeline for nowcasting Acute Respiratory Infectious Diseases (ARIDs), including: (1) symptom extraction from unstructured Chinese chief complaints using open‑source large language models (LLMs) via Ollama; (2) evaluation of multiple LLMs against a gold‑standard annotation set; (3) feature selection via Spearman correlation; (4) training and evaluation of traditional machine learning (LR, RF, XGBoost, DT, SVM) and deep learning (GRU, LSTM) models for daily case count prediction; and (5) Z‑score based outbreak detection. This repository does **not** contain any real patient data; a comprehensive mock‑data generator is provided to simulate the required input format, ensuring full reproducibility.

This documentation will be updated when the manuscript is publicly available. The method builds on established nowcasting and syndromic surveillance frameworks, integrating modern LLMs for automated feature extraction from electronic health records.

---

## Summary

Accurate and timely nowcasting of infectious disease cases is critical for public health response. However, traditional approaches often rely on manual chart review or simple rule‑based methods for symptom extraction, which are time‑consuming and do not scale. We propose a fully automated pipeline that:

1. **Extracts six key symptoms** (fever, cough, sore throat, chest pain, myalgia, dyspnea) from unstructured Chinese chief complaints using seven open‑source lightweight LLMs (Gemma‑3‑1B/4B, Qwen‑3‑1.7B/8B, Deepseek‑r1‑7B, Llama‑3.1‑8B, Llama‑3.2‑3B).
2. **Selects the most effective model** based on F1‑score and processing time, using a composite score (70% performance, 30% efficiency).
3. **Builds daily symptom time‑series** from the extracted indicators and combines them with white blood cell count abnormalities and holiday markers.
4. **Trains and evaluates seven nowcasting models** (Linear Regression, Decision Tree, Random Forest, XGBoost, SVM, GRU, LSTM) to predict daily reported ARIDs cases.
5. **Detects outbreak signals** using a Z‑score thresholding method on model predictions, stratified by case volume.

The entire pipeline is reproducible, requiring a Python environment and a running Ollama instance for LLM inference. The mock‑data generator allows users to test the entire workflow without any real data.

## System requirements

The code is supported on all operating systems for which the requisite downloads (see below) are possible. The example code was tested on a Linux server running Ubuntu 22.04 LTS with an NVIDIA GPU (32 GB memory).

### Required software

- **Python** (>= 3.12.3) – follow instructions at https://www.python.org/
- **Ollama** (>= 0.1.0) – follow instructions at https://ollama.com/ (only needed if you plan to run LLM extraction; mock‑data generation and prediction do not require Ollama)
- **Conda** (optional but recommended) – follow instructions at https://docs.conda.io/

All Python dependencies are listed in `requirements.txt` and `environment.yml`. Installation should take less than 30 minutes on a normal desktop computer, depending on internet speed and GPU availability.

## Installation

After installing Python (and optionally Conda) and Ollama, clone this repository and set up the environment:

```bash
git clone https://github.com/FHS0208/ARIDs_nowcasting.git
cd ARIDs_nowcasting
```

Option 1: Using Conda (recommended)

```bash
conda env create -f environment.yml
conda activate arid-nowcasting
```

Option 2: Using pip

```bash
pip install -r requirements.txt
```

If you plan to run the LLM extraction, ensure Ollama is running and pull the desired model (e.g., Llama 3.1 8B):

```bash
ollama pull llama3.1:8b
```

## Instructions for use

After installation, all experiments can be reproduced by running the scripts in order. For a complete walkthrough, please refer to the README.md file in each subdirectory, but the general workflow is as follows.

### Step 1: Generate mock data (no real data needed)

```bash
cd data
python generate_mock_data.py
```

This creates file:

- `data.xlsx` – daily time‑series data

### Step 2: LLM extraction and evaluation (requires Ollama)

```bash
cd scripts/llm
python batchextraction.py --dataset ../../data/train1000.csv --output_dir ../../data
python llm_evaluation.py --data_dir ../../data --output_dir ../../outputs/llm
```

This will generate `symeva.xlsx` (performance metrics for all models) and produce Figures (boxplots and heatmaps) in the output directory.

### Step 3: Feature selection

```bash
cd ../feature_selection
python spearman_selection.py --data ../../data/data.xlsx --output ../../outputs/feature_selection
```

This selects symptom combinations with Spearman correlation |rho| > 0.4 with the target sum.

### Step 4: Train nowcasting models

```bash
cd ../noecasting
python model_ML.py
python model_DL.py
```

All model predictions and performance metrics (R², MAE, MAPE, RMSE) are saved in the output directory.

### Step 5: Outbreak detection

```bash
cd ../outbreak
python z_score_detection.py
```

This evaluates Z‑score thresholds (2.0, 2.5, 3.0) and generates stratified confusion matrices, time‑series plots, and scatter plots.

### (Optional) Real‑time extraction from your own database

If you wish to apply the LLM extraction to your own electronic health records, you need to provide a configuration file `config.json` (copy from `config.json.example` and fill in your database credentials and column names). Then run:

```bash
python extract_symptoms.py --config config.json
```

> Important: Do not commit config.json to version control – it contains sensitive information.

## Repository Structure

```
ARIDs_nowcasting/
├── README.md                           
├── requirements.txt               # Python dependencies (pip)
├── environment.yml                # Conda environment
├── .gitignore                     # Excluded files
│
├── data/                          # Data (mock data generated here)
│   └── generate_mock_data.py     # Generates synthetic data
│
├── instructions/                  # LLM prompts
│   └── instruction0.txt   
│   └── instruction1.txt   
│   └── instruction2.txt           # System prompt for symptom extraction
│
├── scripts/
│   ├── llm/
│   │   ├── batchvalid.py          # Validation set for prompt/parameter tuning
│   │   ├── batchextraction.py     # Batch LLM extraction (fixed params)
│   │   ├── llm_evaluation.py      # Performance evaluation & symeva.xlsx
│   │   └── extract_symptoms.py    # Real‑time database extraction (Ollama)
│   │
│   ├── feature_selection/
│   │   └── spearman_selection.py  # Spearman correlation (|rho| > 0.4)
│   │
│   ├── nowcasting/
│   │   ├── model_ML.py     # LR, RF, XGBoost, DT, SVM
│   │   └── model_DL.py     # GRU, LSTM with hyperparameter search
│   │
│   └── outbreak/
│       └── z_score_detection.py   # Outbreak detection
│
└── outputs/                       # All results saved here (auto‑created)
    ├── llm/
    ├── feature_selection/
    ├── prediction/
    └── outbreak/
```

## Input Data Format (for prediction models)

| Column    | Description                           |
| --------- | ------------------------------------- |
| rq        | Date (YYYY-MM-DD)                     |
| x1 … x57 | Daily symptom combination frequencies |
| up / down | Upper / lower WBC                     |
| normal    | Normal WBC                            |
| sum       | Daily ARIDs case count (target)       |

The `generate_mock_data.py` produces data in this exact format.

## LLM Evaluation Metrics

Each LLM is evaluated on 1,000 gold‑standard chief complaints using:

- Accuracy
- Precision
- Recall
- F1‑score
- Balanced accuracy
- Processing time (total seconds per model)

Statistical tests:

1. Friedman test for global differences across models.
2. Mann‑Whitney U with Bonferroni correction for pairwise comparisons.
3. Bootstrap (1,000 replicates) for 95% confidence intervals of median differences.

## Nowcasting Model Performance

Models are evaluated on the test set (September 1, 2024 – June 30, 2025) using:

- R² (coefficient of determination)
- MAE (mean absolute error)
- MAPE (mean absolute percentage error)
- RMSE (root mean square error)

Hyperparameter tuning is performed via grid search with 5‑fold cross‑validation on the training set.

## License

Apache 2.0 License. Model usage subject to Ollama Community License.
