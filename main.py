"""
main.py
=======
Single entry point that runs the full project pipeline end to end:
Logistic Regression, Isolation Forest, and the Hybrid combination, in one
command. This is what you run for the final demo / report screenshots.

Usage:
    python main.py --data creditcard__1___1_.csv --out results --lr_weight 0.5

Internally this just calls hybrid_model.run(), which itself trains and
evaluates both individual models before combining them -- so running this
one file exercises all three models and produces every plot/metric needed
for the report.
"""

import argparse

from hybrid_model import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full Credit Card Fraud Detection pipeline")
    parser.add_argument("--data", required=True, help="Path to the credit card CSV file")
    parser.add_argument("--out", default="results", help="Directory to write plots/results to")
    parser.add_argument("--lr_weight", type=float, default=0.5, help="Weight given to Logistic Regression (0-1)")
    args = parser.parse_args()
    run(args.data, args.out, args.lr_weight)
