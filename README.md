# ARIDs_nowcasting
A complete, reproducible pipeline for nowcasting Acute Respiratory Infectious Diseases (ARIDs) using:

- **Open‑source LLMs** (via Ollama) to extract symptoms from unstructured Chinese chief complaints.
- **Traditional machine learning** (LR, RF, XGBoost, DT, SVM) and **deep learning** (GRU, LSTM) for daily case count prediction.
- **Z‑score outbreak detection** based on model predictions for early warning.

This repository provides all code needed to reproduce the analyses described in our manuscript. **No real patient data are included** – a mock data generator is provided to simulate the required input format and allow full pipeline execution.

## Repository Structure

ARIDs_nowcasting/
├── README.md # This file
├── LICENSE # MIT License
├── requirements.txt # Python dependencies (pip)
├── environment.yml # Conda environment
├── .gitignore # Excluded files
│
├── data/ # Data (mock data generated here)
│ └── mock_data_generator.py # Generates synthetic data
│
├── instructions/ # LLM prompts
│ └── instruction2.txt # System prompt for symptom extraction
│
├── scripts/
│ ├── llm/
│ │ ├── batchvalid.py # Validation set for prompt/parameter tuning
│ │ ├── batchextraction.py # Batch LLM extraction (fixed params)
│ │ ├── llm_evaluation.py # Performance evaluation & symeva.xlsx
│ │ └── extract_symptoms.py # Real‑time database extraction (Ollama)
│ │
│ ├── feature_selection/
│ │ └── spearman_selection.py # Spearman correlation (|rho| > 0.4)
│ │
│ ├── prediction/
│ │ ├── run_ml_models.py # LR, RF, XGBoost, DT, SVM
│ │ └── run_deep_models.py # GRU, LSTM with hyperparameter search
│ │
│ └── outbreak/
│ └── z_score_detection.py # Outbreak detection from predictions
│
└── outputs/ # All results saved here (auto‑created)
├── llm/
├── feature_selection/
├── prediction/
└── outbreak/

1. Clone the repository
git clone https://github.com/FHS0208/ARIDs_nowcasting.git
cd ARIDs_nowcasting
2. Set up the environment
Using Conda (recommended):

bash
conda env create -f environment.yml
conda activate arid-nowcasting
Using pip:

bash
pip install -r requirements.txt
3. Generate mock data (no real data required)
bash
cd data
python mock_data_generator.py
This creates:

data.xlsx – daily time‑series with columns rq, x1…x57, up, down, normal, xg, lg, brk, sum.

4. Run the LLM extraction pipeline (requires Ollama)
Make sure Ollama is running and the required model is pulled:

bash
ollama pull llama3.1:8b
Then:

bash
cd scripts/llm
python batchextraction.py --dataset ../../data/train1000.csv --output_dir ../../data
python llm_evaluation.py --data_dir ../../data --output_dir ../../outputs/llm
This will:

Extract symptoms from the mock chief complaints.

Evaluate all LLMs against the gold standard.

Generate symeva.xlsx and performance figures (Figure 3, Figure 4, confusion matrices).

5. Feature selection & prediction
bash
cd ../feature_selection
python spearman_selection.py --data ../../data/data.xlsx --output ../../outputs/feature_selection

cd ../prediction
python run_ml_models.py --data ../../data/data.xlsx --features ../../outputs/feature_selection/selected_features.csv --output ../../outputs/prediction
python run_deep_models.py --data ../../data/data.xlsx --features ../../outputs/feature_selection/selected_features.csv --output ../../outputs/prediction
6. Outbreak detection
bash
cd ../outbreak
python z_score_detection.py --predictions ../../outputs/prediction/lstm_predictions.csv --output ../../outputs/outbreak
Configuration for Database Extraction
If you want to use the real‑time extraction script (extract_symptoms.py) with your own database, you need to provide a config.json file:

bash
cp config.json.example config.json
# Edit config.json with your database credentials and column names
Important: Never commit config.json to version control – it contains sensitive information. The .gitignore file is already set to exclude it.

Input Data Format (for prediction models)
Column	Description
rq	Date (YYYY-MM-DD)
x1 … x57	Daily symptom combination frequencies
up / down	Upper / lower respiratory indicators
normal	Normal / other cases
xg	Daily COVID-19 case count
lg	Daily influenza case count
sum	Daily ARIDs case count (target)
The mock_data_generator.py produces data in this exact format.

LLM Evaluation Metrics
Each LLM is evaluated on 1,000 gold‑standard chief complaints using:

Accuracy

Precision

Recall

F1‑score

Balanced accuracy

Processing time (total seconds per model)

Statistical tests:

Friedman test for global differences across models.

Mann‑Whitney U with Bonferroni correction for pairwise comparisons.


Prediction Model Performance
Models are evaluated on the test set (September 1, 2024 – June 30, 2025) using:

R² (coefficient of determination)

MAE (mean absolute error)

MAPE (mean absolute percentage error)

RMSE (root mean square error)

Hyperparameter tuning is performed via grid search with 5‑fold cross‑validation on the training set.
