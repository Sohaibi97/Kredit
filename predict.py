import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix,
    average_precision_score, roc_curve, precision_recall_curve, precision_score, recall_score
)

import matplotlib.pyplot as plt


# =========================
# Utils
# =========================

def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))

def gini_from_auc(auc: Optional[float]) -> float:
    return (2*auc - 1) if (auc is not None and not np.isnan(auc)) else np.nan

def choose_test_size(y: pd.Series, min_pos=30, min_neg=30, default=0.10, max_test=0.20) -> float:
    N = len(y)
    p = y.mean() if N > 0 else 0.5
    for t in [default, 0.12, 0.15, 0.18, max_test]:
        if (t*N*p >= min_pos) and (t*N*(1-p) >= min_neg):
            return t
    return max_test


# =========================
# Core class (LR @ 0.70)
# =========================

@dataclass
class LRBanksConfig:
    confidence_threshold: float = 0.70       # Gate: nur sehr sichere Fälle werden automatisiert
    decision_threshold: float = 0.50         # Klassifikationsschwelle für "good" vs "bad"
    cv_folds: int = 5
    random_state: int = 42

class LogisticBankingModel:
    def __init__(self, config: LRBanksConfig = LRBanksConfig()):
        self.config = config
        self.model: Optional[Pipeline] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.is_trained: bool = False
        self.feature_names_: Optional[list] = None

    def _make_pipeline(self) -> Pipeline:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=self.config.random_state))
        ])

    def fit_with_cv(self, X: pd.DataFrame, y: pd.Series) -> "LogisticBankingModel":
        self.feature_names_ = list(X.columns)
        pipe = self._make_pipeline()
        param_grid = {
            'clf__C': [0.01, 0.1, 1, 10, 100],
            'clf__penalty': ['l1', 'l2'],
            'clf__solver': ['liblinear']  # passend für l1/l2
        }
        gs = GridSearchCV(
            pipe, param_grid,
            cv=StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state),
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
        gs.fit(X, y)
        self.model = gs.best_estimator_
        self.best_params = gs.best_params_
        self.is_trained = True
        print("\nBest Params:", self.best_params)
        print(f"Best CV ROC-AUC: {gs.best_score_:.3f}")
        return self

    # ---- helper: predictions & gating ----
    def _confidence_mask(self, proba: np.ndarray) -> np.ndarray:
        thr = self.config.confidence_threshold
        return (proba >= thr) | (proba <= (1 - thr))

    # ---- prediction with abstention (confidence) ----
    def predict_with_confidence(self, X: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model must be trained first.")
        proba = self.model.predict_proba(X)[:, 1]
        mask = self._confidence_mask(proba)
        preds = np.full(len(X), -1)  # -1 = human review
        # >>> Klassifikationsschwelle nutzt decision_threshold <<<
        preds[mask] = (proba[mask] >= self.config.decision_threshold).astype(int)
        return {
            'probabilities': proba,
            'predictions': preds,
            'confident_mask': mask,
            'coverage': float(np.mean(mask))
        }

    # ---- evaluation (confident subset + overall) ----
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        y_arr = y.values if hasattr(y, 'values') else y
        res = self.predict_with_confidence(X)
        proba, preds, mask = res['probabilities'], res['predictions'], res['confident_mask']
        coverage = res['coverage']

        # overall (ohne Abstain) – Klassifikation mit decision_threshold
        overall_preds = (proba >= self.config.decision_threshold).astype(int)
        if len(np.unique(y_arr)) > 1:
            overall_auc = roc_auc_score(y_arr, proba)
            overall_pr = average_precision_score(y_arr, proba)
        else:
            overall_auc = overall_pr = np.nan
        overall_ks = ks_statistic(y_arr, proba)
        overall_precision = precision_score(y_arr, overall_preds, zero_division=0)
        overall_recall = recall_score(y_arr, overall_preds, zero_division=0)

        # confident subset
        if mask.sum() == 0:
            cm = np.array([[0, 0], [0, 0]])
            conf_acc = conf_bacc = conf_auc = conf_pr = conf_ks = conf_prec = conf_rec = np.nan
            fp = fn = cost = np.nan
            conf_samples = 0
        else:
            y_conf = y_arr[mask]
            p_conf = proba[mask]
            yhat_conf = (p_conf >= self.config.decision_threshold).astype(int)

            conf_acc = accuracy_score(y_conf, yhat_conf)
            conf_bacc = balanced_accuracy_score(y_conf, yhat_conf)
            conf_prec = precision_score(y_conf, yhat_conf, zero_division=0)
            conf_rec = recall_score(y_conf, yhat_conf, zero_division=0)
            if len(np.unique(y_conf)) > 1:
                conf_auc = roc_auc_score(y_conf, p_conf)
                conf_pr  = average_precision_score(y_conf, p_conf)
                conf_ks  = ks_statistic(y_conf, p_conf)
            else:
                conf_auc = conf_pr = conf_ks = np.nan

            cm = confusion_matrix(y_conf, yhat_conf, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan, np.nan, np.nan, np.nan)
            cost = 5*fp + 1*fn
            conf_samples = int(mask.sum())

        return {
            'coverage': coverage,
            'confident_samples': conf_samples,
            'total_samples': len(y_arr),

            'confident_accuracy': conf_acc,
            'confident_balanced_accuracy': conf_bacc,
            'confident_precision': conf_prec,
            'confident_recall': conf_rec,
            'confident_auc': conf_auc,
            'confident_pr_auc': conf_pr,
            'confident_ks': conf_ks,
            'confident_gini': gini_from_auc(conf_auc),
            'confusion_matrix': cm,
            'false_positives': fp,
            'false_negatives': fn,
            'cost_5xFP_1xFN': cost,

            'overall_auc': overall_auc,
            'overall_pr_auc': overall_pr,
            'overall_ks': overall_ks,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall
        }

    # ---- Threshold-Kosten auf Training via OOF-CV optimieren (kein Test-Leak) ----
    @staticmethod
    def _conf_cost_at_t(y_true: np.ndarray, proba: np.ndarray, conf_thr: float, dec_thr: float):
        mask = (proba >= conf_thr) | (proba <= (1 - conf_thr))
        if mask.sum() == 0:
            return dict(cost=np.inf, FP=np.nan, FN=np.nan, acc=np.nan, prec=np.nan, rec=np.nan,
                        coverage=0.0, samples=0)
        y_c = y_true[mask]
        yhat = (proba[mask] >= dec_thr).astype(int)
        cm = confusion_matrix(y_c, yhat, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        cost = 5*fp + 1*fn
        return dict(
            cost=float(cost),
            FP=int(fp), FN=int(fn),
            acc=float(accuracy_score(y_c, yhat)),
            prec=float(precision_score(y_c, yhat, zero_division=0)),
            rec=float(recall_score(y_c, yhat, zero_division=0)),
            coverage=float(np.mean(mask)),
            samples=int(mask.sum())
        )

    def optimize_decision_threshold_on_training(
        self, X: pd.DataFrame, y: pd.Series,
        grid: np.ndarray = np.linspace(0.50, 0.80, 31)
    ) -> Dict[str, Any]:
        """
        Verwendet Stratified K-Fold, sammelt OOF-Probas und minimiert die Kosten (5×FP + 1×FN)
        nur in der confident-Zone. Setzt anschließend self.config.decision_threshold.
        """
        print("\nOptimizing decision threshold on TRAIN via OOF-CV (cost=5×FP + 1×FN)...")
        skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        oof_idx = np.zeros(len(y), dtype=bool)
        oof_proba = np.zeros(len(y), dtype=float)

        for fold, (tri, tei) in enumerate(skf.split(X, y), 1):
            Xtr, Xval = X.iloc[tri], X.iloc[tei]
            ytr = y.iloc[tri]
            tmp = LogisticBankingModel(self.config)
            tmp.fit_with_cv(Xtr, ytr)  # inner tuning
            oof_proba[tei] = tmp.model.predict_proba(Xval)[:, 1]
            oof_idx[tei] = True
            print(f"  Collected OOF probabilities for fold {fold}")

        assert oof_idx.all(), "OOF probabilities missing for some samples."

        # sweep thresholds
        records = []
        y_true = y.values
        for t in grid:
            m = self._conf_cost_at_t(y_true, oof_proba, self.config.confidence_threshold, t)
            m['threshold'] = float(t)
            records.append(m)
        curve = pd.DataFrame(records)

        # choose best threshold: min cost → tie-break by lower FP → higher acc → higher coverage
        curve_sorted = curve.sort_values(['cost','FP','acc','coverage'], ascending=[True, True, False, False])
        best_row = curve_sorted.iloc[0]
        best_t = float(best_row['threshold'])
        self.config.decision_threshold = best_t

        print(f"Best decision threshold (OOF): {best_t:.3f} | cost={best_row['cost']:.1f} "
              f"| FP={int(best_row['FP'])} | FN={int(best_row['FN'])} "
              f"| acc={best_row['acc']:.3f} | cov={best_row['coverage']:.3f}")
        return {'best_threshold': best_t, 'curve': curve, 'best_row': best_row.to_dict()}

    # ---- CV on training only (for expectation) ----
    def cross_validate_training(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        print("\nCross-Validation on TRAIN (expectations):")
        skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        rows = []
        for k, (tri, tei) in enumerate(skf.split(X, y), 1):
            Xtr, Xte = X.iloc[tri], X.iloc[tei]
            ytr, yte = y.iloc[tri], y.iloc[tei]
            tmp = LogisticBankingModel(self.config)
            tmp.fit_with_cv(Xtr, ytr)  # inner GS
            m = tmp.evaluate(Xte, yte)
            m['fold'] = k
            rows.append(m)
            print(f"  Fold {k}: Acc(conf)={m['confident_accuracy']:.3f} | Cov={m['coverage']:.3f}")
        return pd.DataFrame(rows)

    # ---- Explainability: top coefficients (standardized space) ----
    def top_coefficients(self, k: int = 15) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("Model must be trained.")
        clf = self.model.named_steps['clf']
        if not hasattr(clf, 'coef_'):
            raise ValueError("No coefficients available.")
        coefs = clf.coef_[0]
        feats = self.feature_names_ or [f"f{i}" for i in range(len(coefs))]
        df = pd.DataFrame({'feature': feats, 'coef': coefs, 'abs_coef': np.abs(coefs)})
        return df.sort_values('abs_coef', ascending=False).head(k).reset_index(drop=True)

    # ---- Plots (test set) ----
    def plot_diagnostics(self, X: pd.DataFrame, y: pd.Series, topk: int = 15, figsize=(16, 12)):
        if not self.is_trained:
            raise ValueError("Model must be trained.")
        y_arr = y.values if hasattr(y, 'values') else y
        pred = self.predict_with_confidence(X)
        proba, mask = pred['probabilities'], pred['confident_mask']
        conf_thr = self.config.confidence_threshold
        dec_thr = self.config.decision_threshold

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Logistic Regression (conf={conf_thr:.2f}; decision={dec_thr:.2f}) — Diagnostics', fontsize=15)

        # ROC
        fpr, tpr, roc_thresholds = roc_curve(y_arr, proba)
        auc_roc = np.trapz(tpr, fpr)
        axes[0,0].plot(fpr, tpr, lw=2, label=f'AUC={auc_roc:.3f}')
        axes[0,0].plot([0,1], [0,1], lw=1, linestyle='--')
        # mark decision threshold point on ROC (nearest)
        idx = np.argmin(np.abs(roc_thresholds - dec_thr))
        axes[0,0].scatter(fpr[idx], tpr[idx], s=80, label=f'Decision t={dec_thr:.2f}')
        axes[0,0].set_title('ROC Curve'); axes[0,0].set_xlabel('FPR'); axes[0,0].set_ylabel('TPR')
        axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

        # PR
        precision, recall, pr_thresholds = precision_recall_curve(y_arr, proba)
        # pr_thresholds length = n-1; sichere Indexwahl:
        pr_idx = np.argmin(np.abs(pr_thresholds - dec_thr)) if len(pr_thresholds) > 0 else None
        pr_auc = np.trapz(precision[::-1], recall[::-1])  # approx area
        axes[0,1].plot(recall, precision, lw=2, label=f'PR-AUC≈{pr_auc:.3f}')
        baseline = y_arr.mean()
        axes[0,1].axhline(baseline, linestyle='--', lw=1, label=f'Baseline={baseline:.3f}')
        if pr_idx is not None and pr_idx < len(recall):
            axes[0,1].scatter(recall[pr_idx], precision[pr_idx], s=80, label=f'Decision t={dec_thr:.2f}')
        axes[0,1].set_title('Precision-Recall'); axes[0,1].set_xlabel('Recall'); axes[0,1].set_ylabel('Precision')
        axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

        # Probability distributions
        axes[0,2].hist(proba[y_arr==0], bins=30, alpha=0.7, density=True, label='Bad (0)')
        axes[0,2].hist(proba[y_arr==1], bins=30, alpha=0.7, density=True, label='Good (1)')
        axes[0,2].axvline(conf_thr, linestyle='--', lw=2, label=f'Conf {conf_thr:.2f}')
        axes[0,2].axvline(1-conf_thr, linestyle='--', lw=2)
        axes[0,2].axvline(dec_thr, color='k', lw=2, label=f'Dec {dec_thr:.2f}')
        axes[0,2].set_title('Probability Distribution'); axes[0,2].legend(); axes[0,2].grid(alpha=0.3)

        # Accuracy vs Coverage (confidence sweep, decision fixed)
        tgrid = np.arange(0.50, 0.96, 0.01)
        covs, accs = [], []
        for t in tgrid:
            m = (proba >= t) | (proba <= (1-t))
            if m.sum() > 0:
                y_c = y_arr[m]
                accs.append(accuracy_score(y_c, (proba[m] >= dec_thr)))
                covs.append(np.mean(m))
            else:
                accs.append(np.nan); covs.append(0.0)
        axes[1,0].plot(covs, accs, lw=2)
        # mark current
        cur_eval = self.evaluate(X, y)
        axes[1,0].scatter([cur_eval['coverage']], [cur_eval['confident_accuracy']], s=80, label='Current config')
        axes[1,0].axhline(0.85, linestyle=':', label='Banking min Acc 0.85')
        axes[1,0].axvline(0.30, linestyle=':', label='Min Coverage 0.30')
        axes[1,0].set_title('Accuracy vs Coverage (confidence sweep)'); axes[1,0].set_xlabel('Coverage'); axes[1,0].set_ylabel('Conf. Accuracy')
        axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

        # Confident Confusion Matrix
        axes[1,1].set_title('Confusion (confident subset)')
        if mask.sum() > 0:
            y_c = y_arr[mask]; yhat_c = (proba[mask] >= dec_thr).astype(int)
            cm = confusion_matrix(y_c, yhat_c, labels=[0,1])
            im = axes[1,1].imshow(cm, interpolation='nearest')
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[1,1].text(j, i, int(cm[i,j]), ha='center', va='center',
                                   color='white' if cm[i,j] > cm.max()/2 else 'black')
            axes[1,1].set_xticks([0,1]); axes[1,1].set_yticks([0,1])
            axes[1,1].set_xticklabels(['Bad(0)','Good(1)']); axes[1,1].set_yticklabels(['Bad(0)','Good(1)'])
        else:
            axes[1,1].text(0.5, 0.5, 'No confident predictions', ha='center', va='center', transform=axes[1,1].transAxes)

        # Top coefficients (explainability)
        top = self.top_coefficients(k=topk)
        axes[1,2].barh(top['feature'][::-1], top['coef'][::-1])
        axes[1,2].set_title(f'Top {topk} coefficients (standardized)')
        axes[1,2].set_xlabel('Coefficient')
        plt.tight_layout()
        plt.show()

    # Optional: Plot Kostenkurve (decision threshold)
    def plot_decision_threshold_cost_curve(self, y_true: np.ndarray, proba: np.ndarray,
                                           grid: np.ndarray = np.linspace(0.50, 0.80, 31)):
        recs = []
        for t in grid:
            m = self._conf_cost_at_t(y_true, proba, self.config.confidence_threshold, t)
            m['threshold'] = float(t)
            recs.append(m)
        df = pd.DataFrame(recs)
        plt.figure(figsize=(6,4))
        plt.plot(df['threshold'], df['cost'], lw=2, label='Cost (5×FP + 1×FN)')
        plt.axvline(self.config.decision_threshold, linestyle='--', label=f'Chosen t={self.config.decision_threshold:.2f}')
        plt.xlabel('Decision Threshold'); plt.ylabel('Cost'); plt.title('Cost vs Decision Threshold')
        plt.grid(alpha=0.3); plt.legend(); plt.show()
        return df


# =========================
# Workflow (CV on train + holdout test)
# =========================

def prepare_banking_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if 'target' not in df.columns:
        raise ValueError("DataFrame must contain 'target' column with values 1 (good) and 2 (bad).")
    y = (df['target'] == 1).astype(int)
    X = df.drop(columns=['target']).copy()
    print(f"Prepared data: n={len(X)}, d={X.shape[1]}, good%={y.mean():.2%}")
    return X, y

def train_lr_banking_with_holdout(
    df: pd.DataFrame,
    confidence_threshold: float = 0.70,
    inner_cv_folds: int = 5,
    test_size: Optional[float] = None,
    run_train_cv_summary: bool = True,
    optimize_decision_threshold: bool = True
):
    # Prepare + split
    X, y = prepare_banking_data(df)
    if test_size is None:
        test_size = choose_test_size(y, min_pos=30, min_neg=30, default=0.10, max_test=0.20)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    print(f"Split: train={len(X_tr)}, test={len(X_te)} (test_size={test_size:.0%})")

    # Train with CV on TRAIN only
    cfg = LRBanksConfig(confidence_threshold=confidence_threshold, cv_folds=inner_cv_folds)
    model = LogisticBankingModel(cfg).fit_with_cv(X_tr, y_tr)

    # Optional: decision threshold optimization on TRAIN via OOF
    if optimize_decision_threshold:
        opt = model.optimize_decision_threshold_on_training(X_tr, y_tr, grid=np.linspace(0.50, 0.80, 31))
        # (Optional) OOF-Kostenkurve plotten:
        # model.plot_decision_threshold_cost_curve(y_tr.values, ???)  # OOF proba steckt in opt['curve']

    # Optional CV summary (outer CV across train) – zur Erwartung
    cv_df = None
    if run_train_cv_summary:
        cv_df = model.cross_validate_training(X_tr, y_tr)
        print("\nTraining CV (conf subset) summary:")
        print(cv_df[['fold','confident_accuracy','coverage','cost_5xFP_1xFN']].round(3).to_string(index=False))

    # One-shot evaluation on untouched TEST
    print(f"\nUsing decision_threshold={model.config.decision_threshold:.2f}")
    test_metrics = model.evaluate(X_te, y_te)
    print("\n===== HOLDOUT TEST RESULTS =====")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.3f}" if not np.isnan(v) else f"{k}: NA")
        else:
            print(f"{k}: {v}")

    # Plots on TEST
    model.plot_diagnostics(X_te, y_te, topk=15)

    # Decision Markdown (explainability)
    top = model.top_coefficients(k=10)
    md = f"""
# Entscheidungsnotiz – Logistic Regression @ {int(confidence_threshold*100)}% Confidence, Decision @ {model.config.decision_threshold:.2f}

**Warum LR @ 0.70 / Decision {model.config.decision_threshold:.2f}?**  
- **Trennschärfe** (conf): ROC-AUC = {test_metrics['confident_auc']:.3f}, PR-AUC = {test_metrics['confident_pr_auc']:.3f}, KS = {test_metrics['confident_ks']:.3f}  
- **Accuracy (conf)**: {test_metrics['confident_accuracy']:.3f} bei **Coverage** {test_metrics['coverage']:.3f}  
- **Risiko**: FP={test_metrics['false_positives']}, FN={test_metrics['false_negatives']}, **Cost(5×FP+1×FN)**={test_metrics['cost_5xFP_1xFN']:.1f}  
- **Interpretierbarkeit**: Signifikante Koeffizienten (standardisiert) → fachliche Nachvollziehbarkeit.

**Top-Koeffizienten (absolut) – Richtung zeigt Einfluss auf „good“ (1):**  
{top.to_string(index=False)}
"""
    print("\nMarkdown für Doku/Review:\n", md)
    return model, (X_tr, X_te, y_tr, y_te), test_metrics, cv_df


# =========================
# Example main
# =========================
if __name__ == "__main__":
    #  CSV einlesen (spalte 'target' = {1: good, 2: bad})
    df = pd.read_csv("kredit_final.csv")
    model, splits, test_metrics, cv_df = train_lr_banking_with_holdout(
        df,
        confidence_threshold=0.70,
        inner_cv_folds=5,
        test_size=None,
        run_train_cv_summary=True,
        optimize_decision_threshold=True
    )

    print("Dieses Skript definiert LR@0.70-Workflow mit OOF-optimiertem Decision-Threshold, CV (Train) + Holdout-Test + Plots + Erklär-Markdown.")
    print("Nutzen: train_lr_banking_with_holdout(df, confidence_threshold=0.70)")
