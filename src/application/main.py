import pandas as pd
from sklearn.model_selection import train_test_split
from loaders import sql_load
from psycoptts.add_outcomes import add_outcome_from_csv

def load_all_patients():
    view = "[FOR_kohorte_indhold_pt_journal_inkl_2021_feb2022]"
    sql = "SELECT * FROM [fct]." + view 

    df = sql_load(sql, database="USR_PS_FORSK", chunksize = None)
    return df["dw_ek_borger"]

if __name__ == "__main__":
    stratify_cols = ["outcome1", "outcome2"]
    random_state = 42

    all_patient_ids = load_all_patients().to_frame()

    out_df = add_outcome_from_csv(all_patient_ids, "outcome_ids/lung_cancer_ids.csv", "lung_cancer")

    # TODO
    # Generate in_df from the csvs using add_outcome_from_csv()

    X_train, X_intermediate = train_test_split(
        out_df["dw_ek_borger"],
        test_size=0.4,
        random_state=random_state,
        stratify=all_patient_ids[stratify_cols],
    )

    X_test, X_val = train_test_split(
        X_intermediate,
        test_size=0.5,
        random_state=random_state,
        stratify=X_intermediate[stratify_cols],
    )

    X_train.to_csv("csv/train_ids.csv")
    X_val.to_csv("csv/val_ids.csv")
    X_test.to_csv("csv/test_ids.csv")
