import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

# ======================================================================
# CONFIG (defaults; can be overridden via preprocess_data args)
# ======================================================================
FILE_PATH = "kredit.dat"
CSV_PATH = "kredit_clean.csv"
SAVE_CSV = True

# ======================================================================
# CORE LOGIC wrapped in a function so it can be imported and reused
# ======================================================================
def preprocess_data(file_path: str = FILE_PATH, csv_path: str = CSV_PATH, save_csv: bool = SAVE_CSV) -> pd.DataFrame:
    # -------------------- PREPROCESSING --------------------
    print("📁 Loading raw data...")
    colnames = [
        "checking_status", "duration_months", "credit_history", "purpose",
        "credit_amount", "savings", "employment_since", "installment_rate_pct",
        "personal_status_sex", "debtors", "residence_since", "property",
        "age_years", "installment_plans", "housing", "existing_credits",
        "job", "dependents", "telephone", "foreign_worker", "target"
    ]

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=colnames,
        na_values=["?"]
    )

    # 2) 'foreign_worker' aus ethischen Gründen löschen
    if "foreign_worker" in df.columns:
        df = df.drop(columns=["foreign_worker"])

    # 3) personal_status_sex -> Status (Single vs. Not Single)
    status_map = {"A93": "Single", "A95": "Single", "A91": "Not Single", "A92": "Not Single", "A94": "Not Single"}
    df["Status"] = df["personal_status_sex"].map(status_map).astype("category")
    df = df.drop(columns=["personal_status_sex"], errors="ignore")

    # 4) Job -> binär (Unskilled vs. Qualified) als Kategorie (NaN bleibt NaN)
    unskilled = {"A171", "A172"}
    qualified = {"A173", "A174", "A175"}
    def job_to_binary(code):
        if pd.isna(code): return pd.NA
        code = str(code)
        if code in unskilled: return "Unskilled"
        if code in qualified: return "Qualified"
        return pd.NA
    df["job"] = df["job"].apply(job_to_binary).astype("category")

    # 5) Purpose -> 3 Stufen (NaN bleibt NaN)
    consumption = {"A40", "A41", "A42", "A43", "A44", "A47"}
    investment = {"A46", "A48", "A49"}
    other_repairs = {"A45", "A410"}
    def purpose_to_coarse(code):
        if pd.isna(code): return pd.NA
        code = str(code)
        if code in consumption: return "Consumption"
        if code in investment: return "Investment/Human Capital"
        if code in other_repairs: return "Other/Repairs"
        return pd.NA
    df["purpose"] = df["purpose"].apply(purpose_to_coarse).astype("category")

    # 6) Ordinale Kodierung (NaN -> <NA> via Int64)
    df["savings_ord"] = df["savings"].map({"A61":1,"A62":2,"A63":3,"A64":4,"A65":0}).astype("Int64")
    df["checking_status_ord"] = df["checking_status"].map({"A11":1,"A12":2,"A13":3,"A14":0}).astype("Int64")
    df["employment_since_ord"] = df["employment_since"].map({"A71":0,"A72":1,"A73":2,"A74":3,"A75":4}).astype("Int64")
    df["credit_history_ord"] = df["credit_history"].map({"A30":4,"A31":3,"A32":2,"A33":1,"A34":0}).astype("Int64")

    # 7) One-Hot / binär mit NaN-Durchreichung
    def one_hot_with_nan(source_series: pd.Series, prefix: str) -> pd.DataFrame:
        """Erzeugt One-Hot-Dummies in Int64 und setzt bei NaN in der Quelle
        die gesamte Dummy-Zeile auf <NA>, statt auf 0."""
        dums = pd.get_dummies(source_series, prefix=prefix, dtype="Int64")
        mask = source_series.isna()
        if mask.any():
            dums.loc[mask, :] = pd.NA
            dums = dums.astype("Int64")
        return dums

    # debtors
    df["debtors_cat"] = df["debtors"].map({"A101":"none","A102":"co_applicant","A103":"guarantor"})
    df = pd.concat([df, one_hot_with_nan(df["debtors_cat"], "debtors")], axis=1)

    # property
    df["property_cat"] = df["property"].map({"A121":"Immobilien","A122":"Versicherung","A123":"Auto","A124":"None"})
    df = pd.concat([df, one_hot_with_nan(df["property_cat"], "property")], axis=1)

    # installment plans
    df["installment_plans_cat"] = df["installment_plans"].map({"A141":"bank","A142":"stores","A143":"none"})
    df = pd.concat([df, one_hot_with_nan(df["installment_plans_cat"], "instplan")], axis=1)

    # housing
    df["housing_cat"] = df["housing"].map({"A151":"rent","A152":"own","A153":"free"})
    df = pd.concat([df, one_hot_with_nan(df["housing_cat"], "housing")], axis=1)

    # telephone (NaN bleibt NaN)
    df["telephone_bin"] = df["telephone"].map({"A191":0,"A192":1}).astype("Int64")

    # 8) Numerischer Satz für Status & Purpose (mit NaN-Durchreichung)
    df["Status_bin"] = df["Status"].map({"Single":1, "Not Single":0}).astype("Int64")

    purpose_dummies = one_hot_with_nan(df["purpose"], "purpose")
    df = pd.concat([df, purpose_dummies], axis=1)

    # Optional: job zusätzlich numerisch (NaN bleibt NaN)
    df["job_bin"] = df["job"].map({"Unskilled":0, "Qualified":1}).astype("Int64")

    # 9) Original-Kategorien droppen
    drop_cols = [
        "savings","checking_status","employment_since","credit_history",
        "debtors","property","installment_plans","housing","telephone",
        "debtors_cat","property_cat","installment_plans_cat","housing_cat",
        "Status","purpose","job"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print("Preprocessing completed")

    # -------------------- SCALING --------------------
    print("⚖️ Applying scaling...")

    robust_cols = ["credit_amount"]

    standard_cont_cols = [
        "duration_months",
        "age_years",
        "installment_rate_pct",
        "residence_since",
        "existing_credits",
        "dependents",
    ]

    standard_ordinal_cols = [
        "savings_ord",
        "checking_status_ord",
        "employment_since_ord",
        "credit_history_ord",
    ]

    binary_cols = [
        "debtors_co_applicant", "debtors_guarantor", "debtors_none",
        "property_Auto", "property_Immobilien", "property_None", "property_Versicherung",
        "instplan_bank", "instplan_none", "instplan_stores",
        "housing_free", "housing_own", "housing_rent",
        "telephone_bin", "Status_bin",
        "purpose_Consumption", "purpose_Investment/Human Capital", "purpose_Other/Repairs",
        "job_bin",
    ]

    used_cols = robust_cols + standard_cont_cols + standard_ordinal_cols + binary_cols
    X = df[used_cols].copy()

    robust_num = Pipeline(steps=[("scale", RobustScaler())])
    standard_cont = Pipeline(steps=[("scale", StandardScaler())])
    standard_ordinal = Pipeline(steps=[("scale", StandardScaler())])
    passthrough_bin = "passthrough"

    preprocess = ColumnTransformer(
        transformers=[
            ("robust_num", robust_num, robust_cols),
            ("standard_cont", standard_cont, standard_cont_cols),
            ("standard_ordinal", standard_ordinal, standard_ordinal_cols),
            ("bin", passthrough_bin, binary_cols),
        ],
        remainder="drop"
    )
    preprocess.set_output(transform="pandas")

    X_scaled_df = preprocess.fit_transform(X)

    # Wieder ins Original einbauen (target und andere Spalten bleiben unberührt)
    df = pd.concat([df.drop(columns=used_cols), X_scaled_df], axis=1)

    print("✅ Scaling completed")

    # -------------------- OUTPUT --------------------
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved cleaned file to {csv_path}")

    return df

# ======================================================================
# SIMPLE CLI ENTRYPOINT
# ======================================================================
def main():
    """Run preprocessing with module defaults and save to CSV."""
    preprocess_data(FILE_PATH, CSV_PATH, SAVE_CSV)

if __name__ == "__main__":
    main()

