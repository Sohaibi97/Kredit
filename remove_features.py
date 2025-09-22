# remove_feature.py
import pandas as pd

INPUT_PATH = "kredit_imputed.csv"
OUTPUT_PATH = "kredit_final.csv"

# Columns to drop (from your analysis)
FEATURES_TO_DROP = [
    "debtors_guarantor",
    "debtors_none",
    "dependents",
    "existing_credits",
    "instplan_stores",
    "job_bin",
    "property_Auto",
    "property_Versicherung",
    "purpose_Investment/Human Capital",
    "purpose_Other/Repairs",
    "residence_since",
    "savings_ord",
    "telephone_bin",
]

def main():
    df = pd.read_csv(INPUT_PATH)

    # Drop only those that exist (skip missing quietly)
    cols_to_drop = [c for c in FEATURES_TO_DROP if c in df.columns]
    df_final = df.drop(columns=cols_to_drop)

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}  (rows={len(df_final)}, cols={len(df_final.columns)})")

if __name__ == "__main__":
    main()