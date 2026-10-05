"""
hybrid_model.py
================
Combines the two independently-trained models (logistic_regression_model.py
and isolation_forest_model.py) into a single hybrid fraud score.

    hybrid_score = lr_weight * P(fraud)_LogisticRegression
                 + (1 - lr_weight) * scaled_anomaly_score_IsolationForest

`lr_weight` (default 0.5) controls how much trust is placed in the
supervised model vs. the unsupervised anomaly detector.

Run standalone (trains both underlying models internally):
    python hybrid_model.py --data creditcard__1___1_.csv --out results --lr_weight 0.5

Note: this file does not duplicate any training logic -- it imports and
reuses train_logistic_regression() from logistic_regression_model.py and
train_isolation_forest() from isolation_forest_model.py directly.
"""

import argparse
import os
import warnings

import numpy as np

from preprocessing import get_train_test_split
from evaluation import evaluate, plot_confusion_matrix, plot_roc_comparison, plot_metric_comparison
from logistic_regression_model import train_logistic_regression, predict as lr_predict
from isolation_forest_model import train_isolation_forest, predict as if_predict

warnings.filterwarnings("ignore")


def combine_scores(lr_proba, if_score, lr_weight: float = 0.5, threshold: float = 0.5):
    """
    Min-max scale the Isolation Forest anomaly score to [0, 1] so it is on
    the same scale as the Logistic Regression probability, then take a
    weighted average. Returns (hybrid_score, hybrid_pred).
    """
    scaled_if = (if_score - if_score.min()) / (if_score.max() - if_score.min() + 1e-12)
    hybrid_score = lr_weight * lr_proba + (1 - lr_weight) * scaled_if
    hybrid_pred = (hybrid_score >= threshold).astype(int)
    return hybrid_score, hybrid_pred


def run(data_path: str, out_dir: str = ".", lr_weight: float = 0.5):
    os.makedirs(out_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = get_train_test_split(data_path)

    # --- Train both underlying models -------------------------------------
    lr_model = train_logistic_regression(X_train, y_train)
    if_model = train_isolation_forest(X_train, y_train)

    lr_proba, lr_pred = lr_predict(lr_model, X_test)
    if_score, if_pred = if_predict(if_model, X_test)

    # --- Combine -------------------------------------------------------------
    hybrid_score, hybrid_pred = combine_scores(lr_proba, if_score, lr_weight)

    # --- Evaluate all three for comparison ----------------------------------
    results = [
        evaluate("Logistic Regression", y_test, lr_pred, lr_proba),
        evaluate("Isolation Forest", y_test, if_pred, if_score),
        evaluate(f"Hybrid (lr_weight={lr_weight})", y_test, hybrid_pred, hybrid_score),
    ]

    plot_confusion_matrix(y_test, hybrid_pred, "Hybrid Model", f"{out_dir}/cm_hybrid.png")
    plot_roc_comparison(
        {
            "Logistic Regression": (y_test, lr_proba),
            "Isolation Forest": (y_test, if_score),
            "Hybrid": (y_test, hybrid_score),
        },
        f"{out_dir}/roc_comparison.png",
    )
    plot_metric_comparison(results, f"{out_dir}/metric_comparison.png")

    import pandas as pd

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{out_dir}/model_results.csv", index=False)
    print(f"\nSaved results table -> {out_dir}/model_results.csv")
    print(results_df.round(4).to_string(index=False))

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Logistic Regression + Isolation Forest fraud detection")
    parser.add_argument("--data", required=True, help="Path to the credit card CSV file")
    parser.add_argument("--out", default=".", help="Directory to write plots/results to")
    parser.add_argument("--lr_weight", type=float, default=0.5, help="Weight given to Logistic Regression (0-1)")
    args = parser.parse_args()
    run(args.data, args.out, args.lr_weight)
