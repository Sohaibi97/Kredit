# main.py
from prepare_data import main as prepare_data_main
from imputate import main as imputate_main
from remove_features import main as remove_features_main
import pandas as pd
from predict import train_optimal_banking_model

def main():
    print("=== STEP 1: Preparing data ===")
    prepare_data_main()

    print("\n=== STEP 2: Imputation ===")
    imputate_main()

    print("\n=== STEP 3: Remove features ===")
    remove_features_main()

    print("\n=== STEP 4: Train & Evaluate Model ===")
    # Load final dataset
    df_final = pd.read_csv("kredit_final.csv")
    model, cv_results, summary = train_optimal_banking_model(df_final)

    print("\nPipeline completed successfully!")
    return model, cv_results, summary

if __name__ == "__main__":
    main()
