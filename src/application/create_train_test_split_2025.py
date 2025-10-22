"""
Creates randomized train/test split of data up until March 2025 (from CVD_T2D_kohorte_demografi_marts_2025).
"""

import urllib
import urllib.parse

import pandas as pd

# from psycoptts.add_outcomes import add_outcome_from_csv
# from psycoptts.stratify_by_each_category_individually import (
#     stratified_split_by_each_category,
# )
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from wasabi import msg


from sklearn.model_selection import train_test_split


from sql_writer import write_df_to_sql


def load_patient_ids(view="CVD_T2D_kohorte_demografi_marts_2025"):  # TODO fh:
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
    random_state = 42

    combined_df = load_patient_ids()
    n_in_split = {c: 0 for c in ["total", "train", "test", "val"]}

    n_in_split["total"] = combined_df.shape[0]

    train_prop = 0.7
    test_of_intermediate_prop = 0.5
    # Meaning that the prop of the dataset that ends in val is (1 - train_prop) * val_and_test_prop (e.g. 0.3 * 0.5 = 0.15)

    msg.info("Starting train/intermediate split")
    X_train, X_intermediate = train_test_split(
        combined_df["dw_ek_borger"],
        random_state=random_state,
        shuffle=True,
        train_size=train_prop,
    )
    msg.good("Completed train/intermediate split")

    msg.info("Starting test/val split")
    X_val, X_test = train_test_split(
        X_intermediate,
        random_state=random_state,
        shuffle=True,
        test_size=test_of_intermediate_prop,
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

    write_df_to_sql(df=X_train, table_name="psycop_train_ids_2025")
    write_df_to_sql(df=X_val, table_name="psycop_val_ids_2025")
    write_df_to_sql(df=X_test, table_name="psycop_test_ids_2025")

    msg.good("Splits complete and saved to sql!")
