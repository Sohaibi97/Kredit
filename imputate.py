# imputate.py — read df from kredit_clean.csv, strip prefixes, run imputations, save to file
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, precision_score, make_scorer

# ======================================================================
# Config
# ======================================================================
INPUT_PATH = "kredit_clean.csv"     # produced by your preprocessing script
OUTPUT_PATH = "kredit_imputed.csv"  # write imputed result here

RANDOM_STATE = 42
OUTER_FOLDS = 10
BIG_C = 1e10                      # ~ no regularization for LogReg (L2)
C_GRID_EMP = [0.1, 1.0, 10.0]     # for Employment -> alpha_grid = 1/C
ALPHA_GRID_EMP = [1.0 / c for c in C_GRID_EMP]

# ======================================================================
# Helpers
# ======================================================================
def clip_round(arr, low=0, high=4):
    arr = np.rint(arr)
    return np.clip(arr, low, high).astype(int)

def ordinal_acc(y_true, y_pred_cont, low=0, high=4):
    yp = clip_round(y_pred_cont, low, high)
    return (yp == y_true.astype(int)).mean()

def assert_no_nan(df_like, name="X"):
    bad = df_like.isna().sum()
    if hasattr(bad, "sum"):
        tot = int(bad.sum())
        assert tot == 0, f"{name} enthält {tot} NaNs:\n{bad[bad>0].sort_values(ascending=False).head(15)}"
    else:
        assert not np.isnan(df_like).any(), f"{name} enthält NaNs."

# ======================================================================
# Main
# ======================================================================
def main():
    # 0) Load data
    df = pd.read_csv(INPUT_PATH)

    # Strip ColumnTransformer prefixes, e.g. "bin__job_bin" -> "job_bin"
    df.columns = [c.split("__", 1)[-1] if "__" in c else c for c in df.columns]

    TARGET_COL = "target" if "target" in df.columns else None

    # Purpose dummy columns (expected names)
    purpose_cols = [
        "purpose_Consumption",
        "purpose_Investment/Human Capital",
        "purpose_Other/Repairs",
    ]
    missing_purpose = [c for c in purpose_cols if c not in df.columns]
    if missing_purpose:
        raise KeyError(f"Purpose dummy columns missing in '{INPUT_PATH}': {missing_purpose}")

    # ==================================================================
    # 1) PURPOSE imputieren — Robuste Label-Ableitung mit argmax für Multi-Aktive
    # ==================================================================
    P = df[purpose_cols].astype(float)
    
    # Zeilen mit mindestens einer aktiven Dummy (> 0) als Basis verwenden
    row_sum = P.fillna(0).sum(axis=1)
    has_any_active = (row_sum > 0)
    print(f"[PURPOSE] Zeilen ohne Label (alles 0/NaN): {int((row_sum == 0).sum())}")
    print(f"[PURPOSE] Zeilen mit mehr als einer aktiven Klasse: {int((row_sum > 1).sum())}")

    df["purpose_cat"] = pd.Series(pd.NA, index=df.index, dtype="object")
    # Verwende argmax für alle Zeilen mit mindestens einer aktiven Klasse
    df.loc[has_any_active, "purpose_cat"] = P.loc[has_any_active].idxmax(axis=1)

    mask_purpose = df["purpose_cat"].notna()

    # Features/Labels (exclude purpose dummies, employment_since_ord, job_bin, and target if present)
    exclude_for_purpose = purpose_cols + ["purpose_cat", "employment_since_ord", "job_bin"]
    if TARGET_COL:
        exclude_for_purpose.append(TARGET_COL)
    feature_cols_p = [c for c in df.columns if c not in exclude_for_purpose]

    X_purpose = df.loc[mask_purpose, feature_cols_p].copy()
    y_purpose = df.loc[mask_purpose, "purpose_cat"].copy()

    assert_no_nan(X_purpose, "X_purpose")
    assert_no_nan(y_purpose.to_frame(), "y_purpose")

    # Use lower max_iter; explicit multinomial
    logreg_multi = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        penalty="l2",
        C=BIG_C,
        max_iter=2000,
        class_weight=None,
        random_state=RANDOM_STATE
    )
    outer_cv_p = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Standard precision_weighted scoring (ohne make_scorer wrapper)
    scoring_purpose = {
        "precision_weighted": "precision_weighted",
        "accuracy": "accuracy",
    }
    cvres_p = cross_validate(logreg_multi, X_purpose, y_purpose, cv=outer_cv_p,
                             scoring=scoring_purpose, n_jobs=-1)
    print(
        "[PURPOSE] 10-fold CV  precision_weighted: "
        f"{cvres_p['test_precision_weighted'].mean():.3f} ± {cvres_p['test_precision_weighted'].std():.3f} | "
        f"acc: {cvres_p['test_accuracy'].mean():.3f} ± {cvres_p['test_accuracy'].std():.3f}"
    )

    # Fit and impute missing purpose rows (if any remain)
    logreg_multi.fit(X_purpose, y_purpose)
    if df["purpose_cat"].isna().any():
        X_purpose_missing = df.loc[~mask_purpose, feature_cols_p].copy()
        # ensure identical column order
        X_purpose_missing = X_purpose_missing.reindex(columns=feature_cols_p)
        assert_no_nan(X_purpose_missing, "X_purpose_missing")
        df.loc[~mask_purpose, "purpose_cat"] = logreg_multi.predict(X_purpose_missing)

    # Write back clean one-hot purpose dummies
    df.loc[:, purpose_cols] = 0
    for cat in df["purpose_cat"].dropna().unique():
        df.loc[df["purpose_cat"] == cat, cat] = 1
    df.drop(columns=["purpose_cat"], inplace=True)
    assert_no_nan(df[purpose_cols], "purpose dummy cols")

    # ==================================================================
    # 2) JOB imputieren (binäre LogReg, class_weight='balanced')
    # ==================================================================
    if "job_bin" not in df.columns:
        raise KeyError("Column 'job_bin' missing in the input CSV.")
    if "employment_since_ord" not in df.columns:
        raise KeyError("Column 'employment_since_ord' missing in the input CSV.")

    mask_job = df["job_bin"].notna().astype(bool)

    exclude_for_job = ["job_bin", "employment_since_ord"]
    if TARGET_COL:
        exclude_for_job.append(TARGET_COL)
    feature_cols_j = [c for c in df.columns if c not in exclude_for_job]

    X_job = df.loc[mask_job, feature_cols_j].copy()
    y_job = df.loc[mask_job, "job_bin"].astype(int).copy()

    assert_no_nan(X_job, "X_job")
    assert_no_nan(y_job.to_frame(), "y_job")

    outer_cv_j = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    logreg_bin = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=BIG_C,
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    scoring_job = {
        "precision_weighted": "precision_weighted",
        "accuracy": "accuracy",
    }
    cvres_j = cross_validate(logreg_bin, X_job, y_job, cv=outer_cv_j,
                             scoring=scoring_job, n_jobs=-1)
    print(
        "[JOB] 10-fold CV  precision_weighted: "
        f"{cvres_j['test_precision_weighted'].mean():.3f} ± {cvres_j['test_precision_weighted'].std():.3f} | "
        f"acc: {cvres_j['test_accuracy'].mean():.3f} ± {cvres_j['test_accuracy'].std():.3f}"
    )

    logreg_bin.fit(X_job, y_job)
    if df["job_bin"].isna().any():
        X_job_missing = df.loc[~mask_job, feature_cols_j].copy()
        X_job_missing = X_job_missing.reindex(columns=feature_cols_j)
        assert_no_nan(X_job_missing, "X_job_missing")
        df.loc[~mask_job, "job_bin"] = logreg_bin.predict(X_job_missing)

    assert df["job_bin"].isna().sum() == 0, "job_bin hat noch Missing-Werte."

    # ==================================================================
    # 3) EMPLOYMENT imputieren (RidgeCV), Reporting MAE & OrdAcc
    # ==================================================================
    mask_emp = df["employment_since_ord"].notna().astype(bool)

    exclude_for_emp = ["employment_since_ord"]
    if TARGET_COL:
        exclude_for_emp.append(TARGET_COL)
    feature_cols_e = [c for c in df.columns if c not in exclude_for_emp]

    X_emp = df.loc[mask_emp, feature_cols_e].copy()
    y_emp = df.loc[mask_emp, "employment_since_ord"].copy()

    assert_no_nan(X_emp, "X_emp")
    assert_no_nan(y_emp.to_frame(), "y_emp")

    outer_cv_e = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mae_list, ordacc_list = [], []
    for tr_idx, te_idx in outer_cv_e.split(X_emp):
        Xtr, Xte = X_emp.iloc[tr_idx], X_emp.iloc[te_idx]
        ytr, yte = y_emp.iloc[tr_idx], y_emp.iloc[te_idx]

        ridge_cv = RidgeCV(alphas=ALPHA_GRID_EMP, scoring="neg_mean_absolute_error", cv=5)
        ridge_cv.fit(Xtr, ytr)
        ypred = ridge_cv.predict(Xte)

        mae_list.append(mean_absolute_error(yte, ypred))
        ordacc_list.append(ordinal_acc(yte.values, ypred, low=0, high=4))

    print(f"[EMPLOYMENT] {OUTER_FOLDS}-fold CV MAE: mean={np.mean(mae_list):.3f} ± {np.std(mae_list):.3f}")
    print(f"[EMPLOYMENT] {OUTER_FOLDS}-fold CV Ordinal-Accuracy: mean={np.mean(ordacc_list):.3f} ± {np.std(ordacc_list):.3f}")

    # Final model for employment & impute missing
    ridge_cv_full = RidgeCV(alphas=ALPHA_GRID_EMP, scoring="neg_mean_absolute_error", cv=5)
    ridge_cv_full.fit(X_emp, y_emp)
    best_alpha = ridge_cv_full.alpha_
    best_C_emp = 1.0 / best_alpha
    print(f"[EMPLOYMENT] Best alpha={best_alpha:.6g} -> Best C≈{best_C_emp:.6g}")

    if df["employment_since_ord"].isna().any():
        X_emp_missing = df.loc[~mask_emp, feature_cols_e].copy()
        X_emp_missing = X_emp_missing.reindex(columns=feature_cols_e)
        assert_no_nan(X_emp_missing, "X_emp_missing")
        emp_pred = ridge_cv_full.predict(X_emp_missing)
        df.loc[~mask_emp, "employment_since_ord"] = clip_round(emp_pred, low=0, high=4)

    # Final sanity checks
    assert df[purpose_cols].isna().sum().sum() == 0, "Purpose-Dummies haben NaNs."
    assert df["job_bin"].isna().sum() == 0, "job_bin hat NaNs."
    assert df["employment_since_ord"].isna().sum() == 0, "employment_since_ord hat NaNs."

    print("✓ Imputation abgeschlossen.")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()