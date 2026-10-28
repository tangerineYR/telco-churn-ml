# Reproduction Guide

## Requirements

Use Python 3.11 or 3.12 in a virtual environment.

From the project root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run Tests

The unit tests and synthetic smoke test do not require the Telco dataset or network access.

```bash
python -m unittest discover -s tests -v
python -m scripts.smoke
```

## Download the Dataset

Download the IBM Telco Customer Churn dataset:

```bash
python -m scripts.download_data --output data/Telco-Customer-Churn.csv
```

The downloader validates the dataset schema and records its SHA-256 checksum. It refuses to overwrite an existing file.

You can also use an existing compatible CSV through the `--data-path` option.

## Run the Experiment

```bash
python -m scripts.run_experiment --data-path data/Telco-Customer-Churn.csv --seed 42 --output-dir results-rerun --n-jobs 1
```

The output directory must be empty. Use a new directory to preserve previously generated results.

By default, the experiment evaluates Decision Tree, LinearSVC, RBF SVM, DummyClassifier, and Logistic Regression, including the predefined hyperparameter searches for Decision Tree and LinearSVC.

The `--models` option supports `decision_tree`, `linear_svm`, `rbf_svm`, and `baselines`.

## Validate Results

```bash
python -m scripts.validate_artifacts results-rerun --full
```

The validator checks the experiment results, feature counts, cross-validation records, artifact checksums, and source-code checksums.

To inspect the results included in the repository:

```bash
python -m scripts.validate_artifacts results --full
```

## Experiment Configuration

- **Data split:** Stratified 70% training / 30% test, seed 42.
- **Cross-validation:** Stratified 5-fold CV on training data.
- **Preprocessing:** StandardScaler for numeric features; OneHotEncoder for categorical features.
- **Feature selection:** Logistic Regression RFE, selecting 20 features within each training fold.
- **Hyperparameter selection:** Maximum mean cross-validation Average Precision, with deterministic tie-breaking.
- **Final evaluation:** Held-out test data, evaluated after model settings are fixed.
- **Baselines:** DummyClassifier and Logistic Regression use all encoded features rather than RFE-selected features.

LinearSVC uses `dual="auto"`. No decision threshold tuning is performed.

Training warnings are reported and recorded in `warnings.json`. For complete warning capture within the parent process, use `--n-jobs 1`.

Runtime versions and SHA-256 checksums are recorded in `experiment_manifest.json`. Numerical results may vary across library versions and execution environments.

## Continuous Integration

GitHub Actions runs unit tests, a synthetic smoke test, and validation of the committed result artifacts using Python 3.12.

The full hyperparameter search is not run in CI.