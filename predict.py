
"""Prediction & training CLI for the Optimal Banking Model.

Usage examples:
  # Train a model
  python predict.py train --data kredit_final.csv --out banking_model.pkl

  # Predict on new data (without target column) using a saved model
  python predict.py predict --model banking_model.pkl --data new_applications.csv --out preds.csv

  # Quick business summary (runs evaluation on a labeled CSV)
  python predict.py summary --data kredit_final.csv --model banking_model.pkl
"""
from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path

from prepare_data import load_data
from choose_model import OptimalBankingModel, train_optimal_banking_model

def cmd_train(args):
    df = pd.read_csv(args.data)
    model, cv_results, summary = train_optimal_banking_model(df, target_col=args.target, confidence_threshold=args.threshold)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.out)
    print("Saved model to:", args.out)
    print("CV Summary:", summary)

def cmd_predict(args):
    # Expect a file with ONLY features (no target column).
    X = pd.read_csv(args.data)
    model = OptimalBankingModel()
    model.load_model(args.model)
    res = model.predict_with_confidence(X)
    out_df = pd.DataFrame({
        'prediction': res['predictions'],
        'probability': res['probabilities'],
        'confident': res['confident_mask'].astype(int)
    })
    out_df.to_csv(args.out, index=False)
    print("Wrote predictions to:", args.out)
    print(f"Coverage: {res['coverage']:.2%}")

def cmd_summary(args):
    X, y = load_data(args.data, target_col=args.target)
    model = OptimalBankingModel()
    model.load_model(args.model)
    eval_res = model.evaluate_model(X, y)
    print("Business Impact Summary")
    print("=======================")
    print(f"Coverage: {eval_res.coverage:.1%}")
    print(f"Confident Accuracy: {eval_res.confident_accuracy:.1%}")
    print(f"Balanced Accuracy: {eval_res.confident_balanced_accuracy:.1%}")
    print(f"AUC: {eval_res.confident_auc:.3f}")
    print(f"False Positives: {eval_res.false_positives}, False Negatives: {eval_res.false_negatives}")
    print(f"Cost (5xFP + 1xFN): {eval_res.cost_5xFP_1xFN:.1f}")

def build_parser():
    p = argparse.ArgumentParser(description="Optimal Banking Model CLI")
    sub = p.add_subparsers(dest='cmd', required=True)

    p_train = sub.add_parser('train', help='Train model from labeled CSV')
    p_train.add_argument('--data', required=True, help='CSV with features + target column')
    p_train.add_argument('--target', default='target', help='Target column name (default: target)')
    p_train.add_argument('--threshold', type=float, default=0.85, help='Confidence threshold (default: 0.85)')
    p_train.add_argument('--out', default='banking_model.pkl', help='Where to save the trained model')
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser('predict', help='Predict using a saved model')
    p_pred.add_argument('--model', required=True, help='Path to a saved .pkl created by train')
    p_pred.add_argument('--data', required=True, help='CSV with features only (no target column)')
    p_pred.add_argument('--out', default='predictions.csv', help='Where to save predictions CSV')
    p_pred.set_defaults(func=cmd_predict)

    p_sum = sub.add_parser('summary', help='Print business-oriented evaluation given labeled CSV')
    p_sum.add_argument('--model', required=True, help='Path to a saved .pkl model')
    p_sum.add_argument('--data', required=True, help='Labeled CSV with features + target')
    p_sum.add_argument('--target', default='target', help='Target column name (default: target)')
    p_sum.set_defaults(func=cmd_summary)

    return p

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
