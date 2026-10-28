import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from .preprocessing import make_preprocessor
from .feature_selection import make_rfe

DT_GRID = {"classifier__criterion": ["gini", "entropy"],
           "classifier__max_depth": [3, 5, 7, None],
           "classifier__min_samples_split": [2, 5, 10],
           "classifier__min_samples_leaf": [1, 2, 4]}
SVM_GRID = {"classifier__C": [0.1, 1, 10], "classifier__max_iter": [1000, 5000, 10000]}
SCORING = {"ap": "average_precision", "accuracy": "accuracy", "roc_auc": "roc_auc",
           "precision": make_scorer(precision_score, zero_division=0),
           "recall": make_scorer(recall_score, zero_division=0),
           "f1": make_scorer(f1_score, zero_division=0)}

def make_pipeline(kind, seed=42):
    estimators = {
        "decision_tree": DecisionTreeClassifier(random_state=seed),
        "rbf_svm": SVC(kernel="rbf", random_state=seed),
        "linear_svm": LinearSVC(C=1.0, max_iter=1000, dual="auto", random_state=seed),
        "dummy": DummyClassifier(strategy="most_frequent", random_state=seed),
        "logistic": LogisticRegression(max_iter=5000, class_weight=None, random_state=seed),
    }
    steps = [("preprocess", make_preprocessor())]
    if kind not in {"dummy", "logistic"}:
        steps.append(("rfe", make_rfe(seed)))
    return Pipeline(steps + [("classifier", estimators[kind])])

def choose_best(results):
    # Maximum mean AP, then smallest original ParameterGrid index on exact ties.
    scores = np.asarray(results["mean_test_ap"])
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite CV AP")
    return int(np.argmax(scores))

def make_search(kind, seed=42, n_jobs=1, grid=None):
    if kind not in {"decision_tree", "linear_svm"}:
        raise ValueError("Only historical DT/LinearSVC grids are searched")
    return GridSearchCV(make_pipeline(kind, seed),
                        grid if grid is not None else (DT_GRID if kind == "decision_tree" else SVM_GRID),
                        scoring=SCORING, refit=choose_best,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
                        n_jobs=n_jobs, error_score="raise", return_train_score=False)

def fit_search(X_train, y_train, kind, seed=42, n_jobs=1, grid=None):
    # Deliberately no holdout argument in the selection API.
    return make_search(kind, seed, n_jobs, grid).fit(X_train, y_train)
