
"""Model selection, training, and evaluation for the Optimal Banking Model.

This module contains the `OptimalBankingModel` class and a convenience
function `train_optimal_banking_model` that mirrors the structure from the presentation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
import joblib

@dataclass
class EvaluationResult:
    coverage: float
    confident_accuracy: float
    confident_balanced_accuracy: float
    confident_auc: float
    confident_samples: int
    total_samples: int
    cost_5xFP_1xFN: float
    false_positives: int
    false_negatives: int
    confusion_matrix: np.ndarray

class OptimalBankingModel:
    def __init__(self, confidence_threshold: float = 0.85):
        """GradientBoosting pipeline with confidence filtering."""
        self.confidence_threshold = float(confidence_threshold)
        self.model: Pipeline | None = None
        self.best_params: Dict[str, Any] | None = None
        self.is_trained: bool = False

    def create_optimized_pipeline(self) -> Pipeline:
        """Create the optimized GradientBoosting pipeline."""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                verbose=0
            ))
        ])
        return pipeline

    def train_with_hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> "OptimalBankingModel":
        """Train model with hyperparameter optimization (GridSearchCV)."""
        base_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(random_state=42, verbose=0))
        ])

        param_grid = {
            'clf__n_estimators': [150, 200, 250],
            'clf__learning_rate': [0.05, 0.1, 0.15],
            'clf__max_depth': [4, 5, 6],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2]
        }

        grid_search = GridSearchCV(
            base_pipeline,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.is_trained = True
        return self

    def predict_with_confidence(self, X: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        proba = self.model.predict_proba(X)[:, 1]
        high_conf = (proba >= self.confidence_threshold) | (proba <= (1 - self.confidence_threshold))
        preds = np.full(len(X), -1)
        idx = np.where(high_conf)[0]
        preds[idx] = (proba[idx] >= 0.5).astype(int)
        return {
            'predictions': preds,
            'probabilities': proba,
            'confident_mask': high_conf,
            'coverage': float(np.mean(high_conf)),
        }

        # Note: value -1 indicates 'requires human review'.

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> EvaluationResult:
        res = self.predict_with_confidence(X_test)
        preds = res['predictions']
        proba = res['probabilities']
        mask = res['confident_mask']
        coverage = res['coverage']

        if np.sum(mask) == 0:
            return EvaluationResult(
                coverage=0.0,
                confident_accuracy=0.0,
                confident_balanced_accuracy=0.0,
                confident_auc=float('nan'),
                confident_samples=0,
                total_samples=len(X_test),
                cost_5xFP_1xFN=float('nan'),
                false_positives=0,
                false_negatives=0,
                confusion_matrix=np.array([[0, 0], [0, 0]])
            )

        y_c = y_test[mask]
        p_c = preds[mask]
        proba_c = proba[mask]

        acc = accuracy_score(y_c, p_c)
        bacc = balanced_accuracy_score(y_c, p_c)
        auc = roc_auc_score(y_c, proba_c) if len(np.unique(y_c)) > 1 else float('nan')
        cm = confusion_matrix(y_c, p_c)

        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            cost = 5 * fp + 1 * fn
        else:
            fp = fn = 0
            cost = float('nan')

        return EvaluationResult(
            coverage=coverage,
            confident_accuracy=acc,
            confident_balanced_accuracy=bacc,
            confident_auc=auc,
            confident_samples=int(np.sum(mask)),
            total_samples=len(X_test),
            cost_5xFP_1xFN=float(cost),
            false_positives=int(fp),
            false_negatives=int(fn),
            confusion_matrix=cm
        )

    def cross_validate_performance(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Tuple[List[EvaluationResult], Dict[str, float]]:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        results: List[EvaluationResult] = []

        for tr, te in skf.split(X, y):
            m = OptimalBankingModel(self.confidence_threshold).train_with_hyperparameter_tuning(X.iloc[tr], y.iloc[tr])
            r = m.evaluate_model(X.iloc[te], y.iloc[te])
            results.append(r)

        df = pd.DataFrame([{
            'acc': r.confident_accuracy,
            'cov': r.coverage,
            'cost': r.cost_5xFP_1xFN,
            'samples': r.confident_samples
        } for r in results])

        summary = {
            'mean_accuracy': float(df['acc'].mean()),
            'std_accuracy': float(df['acc'].std()),
            'mean_coverage': float(df['cov'].mean()),
            'std_coverage': float(df['cov'].std()),
            'mean_cost': float(df['cost'].mean()),
            'mean_samples_per_fold': float(df['samples'].mean()),
        }
        return results, summary

    def save_model(self, filepath: str) -> None:
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        joblib.dump({
            'model': self.model,
            'confidence_threshold': self.confidence_threshold,
            'best_params': self.best_params
        }, filepath)

    def load_model(self, filepath: str) -> None:
        saved = joblib.load(filepath)
        self.model = saved['model']
        self.confidence_threshold = float(saved['confidence_threshold'])
        self.best_params = saved.get('best_params')
        self.is_trained = True

def train_optimal_banking_model(df: pd.DataFrame, target_col: str = 'target', confidence_threshold: float = 0.85):
    """High-level training entry point returning (model, cv_results, summary)."""
    y = (df[target_col] == 1).astype(int)
    X = df.drop(columns=[target_col]).copy()
    model = OptimalBankingModel(confidence_threshold=confidence_threshold)
    model.train_with_hyperparameter_tuning(X, y)
    cv_results, summary = model.cross_validate_performance(X, y)
    return model, cv_results, summary
