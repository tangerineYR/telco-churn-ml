from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, precision_recall_curve)
from sklearn.tree import plot_tree
from .feature_selection import feature_record

def evaluate(y, predicted, scores):
    return {"accuracy": float(accuracy_score(y, predicted)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores)),
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
            "predicted_positive_count": int(np.sum(np.asarray(predicted) == 1))}

def score_model(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X), "decision_function (not calibrated probability)"
    return model.predict_proba(X)[:, 1], "predict_proba (not calibration-validated)"

def plot_decision_tree(fitted_pipeline, path):
    names = feature_record(fitted_pipeline)["selected_feature_names"]
    classifier = fitted_pipeline.named_steps["classifier"]
    fig, ax = plt.subplots(figsize=(26, 12))
    plot_tree(classifier, feature_names=names, class_names=["No churn", "Churn"],
              max_depth=3, filled=True, rounded=True, proportion=True, fontsize=8, ax=ax)
    ax.set_title("Tuned Decision Tree | Top 3 Levels")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def plot_comparison(y, predictions, scores, models, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    if "decision_tree_tuned" in models:
        plot_decision_tree(models["decision_tree_tuned"], directory / "decision_tree.png")
    for name, suffix in [("decision_tree_tuned", "dt"), ("linear_svm_tuned", "svm")]:
        if name in predictions:
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_predictions(y, predictions[name], labels=[0, 1],
                display_labels=["No churn (0)", "Churn (1)"], ax=ax, colorbar=False)
            ax.set_title(name); fig.tight_layout()
            fig.savefig(directory / f"confusion_matrix_{suffix}.png", dpi=150); plt.close(fig)
    for kind in ["roc", "pr"]:
        fig, ax = plt.subplots(figsize=(9, 7))
        for name, score in scores.items():
            if kind == "roc":
                x, yy, _ = roc_curve(y, score); value = roc_auc_score(y, score)
                label = f"{name} (AUC={value:.3f})"
            else:
                yy, x, _ = precision_recall_curve(y, score); value = average_precision_score(y, score)
                label = f"{name} (AP={value:.3f})"
            ax.plot(x, yy, label=label)
        if kind == "roc": ax.plot([0, 1], [0, 1], "--", color="gray")
        else: ax.axhline(np.mean(y), linestyle="--", color="gray", label="Test prevalence")
        ax.set(xlabel="False positive rate" if kind == "roc" else "Recall",
               ylabel="True positive rate" if kind == "roc" else "Precision",
               title="Held-out Test Evaluation | Churn = Yes", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8); ax.grid(alpha=0.2); fig.tight_layout()
        fig.savefig(directory / f"{kind}_comparison.png", dpi=150); plt.close(fig)
