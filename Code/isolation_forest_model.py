"""
isolation_forest_model.py
==========================
Unsupervised component of the hybrid fraud detection system.

Isolation Forest is trained WITHOUT labels — it isolates points that are
easy to separate from the rest of the data (few random splits needed to
isolate them), which tend to correspond to anomalies/fraud. This makes it
capable of flagging fraud patterns that were never seen during training,
which a purely supervised model like Logistic Regression cannot do.

Run standalone:
    python isolation_forest_model.py --data creditcard__1___1_.csv --out results

This trains and evaluates ONLY the Isolation Forest model, independent of
Logistic Regression, and saves its own confusion matrix / ROC curve. It is
also imported by hybrid_model.py.
"""

import argparse
import os
import warnings

from sklearn.ensemble import IsolationForest

from preprocessing import RANDOM_STATE, get_train_test_split
from evaluation import evaluate, plot_confusion_matrix, plot_roc_curve

warnings.filterwarnings("ignore")


def train_isolation_forest(X_train, y_train, contamination: float = None) -> IsolationForest:
    """
    Fit an Isolation Forest. `contamination` (expected fraction of
    anomalies) defaults to the observed fraud rate in the training labels
    -- used only to calibrate the decision threshold, NOT for fitting.
    """
    if contamination is None:
        contamination = max(y_train.mean(), 1e-4)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train)  # unsupervised: labels are NOT passed in
    return model


def predict(model: IsolationForest, X):
    """
    Return (anomaly_score, hard_prediction).
    IsolationForest.predict() -> -1 for anomaly (fraud), 1 for normal.
    decision_function() -> higher = more "normal", so it is inverted
    to produce an anomaly score where higher = more fraud-like.
    """
    raw_pred = model.predict(X)
    pred = (raw_pred == -1).astype(int)
    score = -model.decision_function(X)
    return score, pred


def run(data_path: str, out_dir: str = "."):
    os.makedirs(out_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = get_train_test_split(data_path)

    model = train_isolation_forest(X_train, y_train)
    score, pred = predict(model, X_test)

    result = evaluate("Isolation Forest", y_test, pred, score)

    plot_confusion_matrix(y_test, pred, "Isolation Forest", f"{out_dir}/cm_isolation_forest.png")
    plot_roc_curve(y_test, score, "Isolation Forest", f"{out_dir}/roc_isolation_forest.png")

    return model, score, pred, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & evaluate Isolation Forest for fraud detection")
    parser.add_argument("--data", required=True, help="Path to the credit card CSV file")
    parser.add_argument("--out", default=".", help="Directory to write plots to")
    args = parser.parse_args()
    run(args.data, args.out)
