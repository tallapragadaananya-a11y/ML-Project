"""
evaluation.py
=============
Shared evaluation utilities: metric computation and plotting.
Imported by logistic_regression_model.py, isolation_forest_model.py, and
hybrid_model.py so every model is scored the exact same way, making their
results directly comparable.
"""

import matplotlib
matplotlib.use("Agg")  # safe for headless / notebook export
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(name: str, y_true, y_pred, y_score=None) -> dict:
    """Compute Accuracy, Precision, Recall, F1-score and (optionally) ROC-AUC."""
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_score)

    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<10}: {v:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Genuine", "Fraud"], zero_division=0))
    return metrics


def plot_confusion_matrix(y_true, y_pred, title: str, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Genuine", "Fraud"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {out_path}")


def plot_roc_curve(y_true, y_score, name: str, out_path: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, name=name, ax=ax)
    ax.set_title(f"ROC Curve — {name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve -> {out_path}")


def plot_roc_comparison(curves: dict, out_path: str):
    """curves: {model_name: (y_true, y_score)}"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_score) in curves.items():
        RocCurveDisplay.from_predictions(y_true, y_score, name=name, ax=ax)
    ax.set_title("ROC Curves — Model Comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC comparison -> {out_path}")


def plot_metric_comparison(results: list, out_path: str):
    import pandas as pd

    df = pd.DataFrame(results).set_index("Model")
    cols = [c for c in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"] if c in df.columns]
    ax = df[cols].plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(loc="lower right")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved metric comparison -> {out_path}")
