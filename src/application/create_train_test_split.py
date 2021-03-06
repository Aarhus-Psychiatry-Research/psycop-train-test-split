import urllib
import urllib.parse
from collections import defaultdict

import pandas as pd
from psycoptts.add_outcomes import add_outcome_from_csv
from psycoptts.stratify_by_each_category_individually import (
    stratified_split_by_each_category,
)
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from wasabi import msg

from pathlib import Path


def load_patient_ids(view="FOR_kohorte_demografi_inkl_2021_feb2022"):
    view = f"{view}"
    query = "SELECT * FROM [fct]." + view

    msg.info(f"Getting patient IDs with query: {query}")

    driver = "SQL Server"
    server = "BI-DPA-PROD"
    database = "USR_PS_Forsk"

    params = urllib.parse.quote(
        f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes"
    )
    engine = create_engine(
        "mssql+pyodbc:///?odbc_connect=%s" % params, poolclass=NullPool
    )
    conn = engine.connect().execution_options(stream_results=True)

    df = pd.read_sql(query, conn, chunksize=None)
    msg.good("Finished loading patients IDs")
    return df[["dw_ek_borger"]]


if __name__ == "__main__":

    OUTCOME_ID_PATH = Path(
        "\\\\TSCLIENT\\P\\MANBER01\\documentLibrary\\train-test-splits\\outcome_ids"
    )
    outcomes = [
        "transition_to_schizophrenia",
        "inpatient_forced_admissions",
        "outpatient_forced_admissions",
        "mammarian_cancer",
        "lung_cancer",
        "t2d",
        "acute_sedatives",
    ]

    random_state = 42

    combined_df = load_patient_ids()
    n_in_split = {c: 0 for c in ["total", "train", "test", "val"]}

    n_in_split["total"] = combined_df.shape[0]

    # Generate unsplit_outcome_props to test the split
    unsplit_outcome_props = defaultdict(lambda: 0)

    for outcome in outcomes:
        combined_df = add_outcome_from_csv(
            df_in=combined_df,
            df_outcome_path=OUTCOME_ID_PATH / (outcome + ".csv"),
            new_colname=outcome,
            id_colname="dw_ek_borger",
        )

    train_prop = 0.7
    test_of_intermediate_prop = 0.5
    # Meaning that the prop of the dataset that ends in val is (1 - train_prop) * val_and_test_prop (e.g. 0.3 * 0.5 = 0.15)

    msg.info("Starting train/intermediate split")
    X_train, X_intermediate = stratified_split_by_each_category(
        combined_df,
        test_prop=(1 - train_prop),
        random_state=random_state,
        stratify_cols=outcomes,
    )
    msg.good("Completed train/intermediate split")

    msg.info("Starting test/val split")
    X_val, X_test = stratified_split_by_each_category(
        X_intermediate,
        test_prop=test_of_intermediate_prop,
        random_state=random_state,
        stratify_cols=outcomes,
    )
    msg.good("Completed test/val split")

    n_in_split = {}

    n_in_split["total"] = combined_df.shape[0]
    n_in_split["train"] = X_train.shape[0]
    n_in_split["test"] = X_test.shape[0]
    n_in_split["val"] = X_val.shape[0]

    for split in ["train", "test", "val"]:
        msg.info(
            f"Prop of patients in {split}: {round(n_in_split[split]/n_in_split['total'], 4)}"
        )

    train_outcome_props = defaultdict(lambda: 0)

    for outcome in outcomes:
        train_outcome_prop = round(
            X_train[X_train[outcome] == 1].shape[0] / n_in_split["train"], 4
        )
        test_outcome_prop = round(
            X_test[X_test[outcome] == 1].shape[0] / n_in_split["test"], 4
        )
        val_outcome_prop = round(
            X_val[X_val[outcome] == 1].shape[0] / n_in_split["val"], 4
        )

        msg.info(
            f"(U|TEST|VAL|TRAIN): {unsplit_outcome_props[outcome]} | {test_outcome_prop} | {val_outcome_prop} | {train_outcome_prop} | {outcome[:15]}"
        )

    X_train["dw_ek_borger"].to_csv("splits/train_ids.csv", index=False)
    X_val["dw_ek_borger"].to_csv("splits/val_ids.csv", index=False)
    X_test["dw_ek_borger"].to_csv("splits/test_ids.csv", index=False)
    msg.good("Splits complete!")
