"""
logistic_regression_model.py
=============================
Supervised component of the hybrid fraud detection system.

Logistic Regression learns the relationship between transaction features
(V1-V28, Time, Amount) and the labeled 'Class' target. class_weight='balanced'
automatically re-weights the loss function to counter the severe class
imbalance (0.17% fraud), without needing to resample the data.

Run standalone:
    python logistic_regression_model.py --data creditcard__1___1_.csv --out results

This trains and evaluates ONLY the Logistic Regression model, independent
of Isolation Forest, and saves its own confusion matrix / ROC curve. It is
also imported by hybrid_model.py so the hybrid script reuses this exact
training logic rather than duplicating it.
"""

import argparse
import os
import warnings

from sklearn.linear_model import LogisticRegression

from preprocessing import RANDOM_STATE, get_train_test_split
from evaluation import evaluate, plot_confusion_matrix, plot_roc_curve

warnings.filterwarnings("ignore")


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Fit a class-weighted Logistic Regression classifier."""
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def predict(model: LogisticRegression, X, threshold: float = 0.5):
    """Return (probability_of_fraud, hard_prediction)."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return proba, pred


def run(data_path: str, out_dir: str = "."):
    os.makedirs(out_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = get_train_test_split(data_path)

    model = train_logistic_regression(X_train, y_train)
    proba, pred = predict(model, X_test)

    result = evaluate("Logistic Regression", y_test, pred, proba)

    plot_confusion_matrix(y_test, pred, "Logistic Regression", f"{out_dir}/cm_logistic_regression.png")
    plot_roc_curve(y_test, proba, "Logistic Regression", f"{out_dir}/roc_logistic_regression.png")

    return model, proba, pred, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & evaluate Logistic Regression for fraud detection")
    parser.add_argument("--data", required=True, help="Path to the credit card CSV file")
    parser.add_argument("--out", default=".", help="Directory to write plots to")
    args = parser.parse_args()
    run(args.data, args.out)
