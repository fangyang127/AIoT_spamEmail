# Proposal: add_spam_email_classifier

## Background

Spam classification is an important natural language processing task that helps protect users and systems from unwanted or malicious messages. Detecting spam early reduces user annoyance, prevents phishing and fraud, and helps downstream systems (e.g., SMS gateways, email servers, analytics pipelines) focus on high-quality data. For coursework and reproducible ML practice, building a small, well-documented spam classifier is an excellent vehicle to learn preprocessing, model training, evaluation, and deployment via a lightweight UI.

Note: the dataset linked for this project is an SMS spam dataset (sms_spam_no_header.csv). Although this proposal is saved as `add_spam_email_classifier`, the implementation will use the SMS dataset. I recommend renaming the project to "Spam SMS Classifier" for clarity—unless you prefer to treat SMS and email as interchangeable in this assignment.

Dataset
- Source: https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/Chapter03/datasets/sms_spam_no_header.csv
- Format: CSV with (label, message) rows (no header). We'll document exact column mapping after downloading and inspecting the file.

## Objectives

Primary objectives:
- Implement a reproducible ML pipeline in Python that classifies SMS messages as spam or ham (not spam) using scikit-learn.
- Use Logistic Regression as the main algorithm and expose a simple Streamlit UI to demonstrate predictions and model metrics.
- Provide automated evaluation (cross-validation, holdout test) and standard metrics (accuracy, precision, recall, F1, ROC AUC) plus visualizations (confusion matrix, ROC curve).
- Package model artifacts (serialized model and metadata) and include clear instructions for reproducibility.

Secondary objectives:
- Provide unit tests for preprocessing and a smoke test for training.
- Integrate proposal/validation into the OpenSpec workflow (proposal saved here); optionally create an OpenSpec change under `openspec/changes/` if you want stricter validation and CI gating.

## Implementation Plan

Files to create/modify (suggested minimal scaffold):

- `data_loader.py`
  - Responsibilities: download or load `sms_spam_no_header.csv`, parse into DataFrame with columns `label` and `message`, provide simple sampling utilities and a function to compute a data hash for reproducibility.

- `preprocessing.py`
  - Responsibilities: text cleaning (lowercase, remove punctuation), tokenization (simple whitespace/tokenizer or use nltk), optional stopword removal, and a scikit-learn `Pipeline` step for `TfidfVectorizer`.

- `model.py`
  - Responsibilities: define scikit-learn `Pipeline` that composes preprocessing and `LogisticRegression`, functions to train, save (joblib), and load models.

- `train.py`
  - Responsibilities: orchestrate training run: load data, split (train/val/test or cross-validation), hyperparameter search (GridSearchCV or simple default), persist best model and metadata (model card), and emit metrics JSON.

- `evaluate.py`
  - Responsibilities: compute metrics (accuracy, precision, recall, F1), create confusion matrix and ROC curve plots, save evaluation artifacts to `models/<model-name>/`.

- `src/app/streamlit_app.py` (or `app.py`)
  - Responsibilities: provide Streamlit UI to upload sample messages, run model prediction, show metrics, confusion matrix and ROC; dataset explorer (small sample view) and explanation of model metadata.

- `requirements.txt`
  - Pin dependencies (scikit-learn, pandas, numpy, joblib, streamlit, matplotlib, seaborn, nltk).

- `tests/test_preprocessing.py`, `tests/test_model_pipeline.py`
  - Minimal pytest tests: preprocessing outputs expected shapes/types, pipeline runs on a tiny synthetic sample.

- `notebooks/quick_explore.ipynb` (optional)
  - Quick EDA and demonstration of training/eval results for grading or reproducibility.

Project layout (minimal):

```
.
├── data/
│   └── sms_spam_no_header.csv   # downloaded (or script downloads into this dir)
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── app/
│   └── streamlit_app.py
├── models/
│   └── model--logistic--v0.1.joblib
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
└── openspec/
    └── proposals/add_spam_email_classifier.md
```

Implementation notes and choices:
- Use scikit-learn `Pipeline` and `TfidfVectorizer` to keep preprocessing and model bundled.
- Use `LogisticRegression` with `liblinear` or `saga` solver depending on dataset size; set `random_state` for reproducibility.
- Save model artifacts with `joblib.dump()` and include `metadata.json` containing training date, dataset hash, random seed, and metrics.

## Testing Plan

Evaluation strategy:
- Data split: stratified train/test split (e.g., 80/20) with optional cross-validation (5-fold) for robust estimates.
- Metrics: accuracy, precision, recall, F1-score (macro and/or weighted as appropriate), ROC curve and AUC.
- Confusion matrix: plot and numeric counts for TP, TN, FP, FN.
- Threshold analysis: consider classifier probability thresholds to trade off precision/recall; include PR curve if needed.

Automated tests:
- Unit tests:
  - Test text cleaning returns consistent outputs for known inputs.
  - Test tfidf pipeline transforms sample texts producing expected shape.
- Integration (smoke) tests:
  - Run `train.py` on a small sampled subset to ensure end-to-end training completes and produces a model artifact.
  - Evaluate that accuracy on holdout is a finite number and metrics JSON is emitted.

CI checks (recommended):
- `pytest` runs for unit tests.
- Linting/formatting checks (black --check, flake8).

## Expected Output

Deliverables and artifacts:
- Trained model artifact: `models/model--logistic--v0.1.joblib` and `models/metadata.json` (model_name, version, metrics, data_hash).
- Evaluation reports: metrics JSON (accuracy, precision, recall, F1, AUC), confusion matrix PNG, ROC curve PNG.
- Streamlit app that can:
  - Show sample dataset rows
  - Accept user input text and return prediction with probability
  - Visualize evaluation plots and model card
- Tests: pytest passing for basic unit and smoke tests.

Visualizations:
- Confusion matrix heatmap (matplotlib or seaborn).
- ROC curve with AUC annotation.
- Bar chart for precision/recall/F1 by class.

Success criteria
- Pipeline trains and exports a Logistic Regression model that achieves reasonable performance on the provided dataset (e.g., F1 > 0.80 depending on data balance — exact target to be determined after initial EDA).
- Streamlit app runs and demonstrates predictions and evaluation artifacts.

## Dependencies

- Python 3.8+ (recommend 3.10+)
- scikit-learn
- pandas
- numpy
- joblib
- streamlit
- matplotlib
- seaborn
- pytest
- nltk (or spaCy) — optional for tokenization/stopwords

Example `requirements.txt` (minimal):

```
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
joblib
streamlit
matplotlib
seaborn
pytest
nltk
```

## Notes and Assumptions
- The dataset link is publicly available; the CSV has no header and may require mapping columns after download. We'll add a small download/inspection helper in `data_loader.py`.
- The user requested Logistic Regression specifically; we will implement that as the primary model and may include code hooks to compare a second baseline (e.g., RandomForest) later if requested.
- Location: the proposal is saved under `openspec/proposals/` per your request. OpenSpec's canonical workflow expects change proposals under `openspec/changes/<change-id>/`. If you want CI validation (`openspec validate`) or stricter change tracking, I can scaffold `openspec/changes/add-spam-email-classifier/` instead and add spec deltas.

## Next steps (recommended)
1. Confirm whether to treat this as an OpenSpec change proposal under `openspec/changes/` (recommended) or keep under `openspec/proposals/`.
2. I can scaffold the minimal code files (`data_loader.py`, `preprocessing.py`, `model.py`, `train.py`, `evaluate.py`, `app/streamlit_app.py`) and a `requirements.txt`, then run the smoke training on a small subset. Tell me to proceed and I'll implement the scaffold and a small test run.
