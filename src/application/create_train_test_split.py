import pandas as pd
from sklearn.model_selection import train_test_split
from psycoptts.add_outcomes import add_outcome_from_csv

from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

import urllib
import urllib.parse

from wasabi import msg


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

    for outcome in outcomes:
        combined_df = add_outcome_from_csv(
            df_in=combined_df,
            df_outcome_path=f"outcome_ids/{outcome}.csv",
            new_colname=outcome,
            id_colname="dw_ek_borger",
        )
        msg.good(f"Added {outcome}")

    X_train, X_intermediate = train_test_split(
        combined_df,
        test_size=0.3,
        random_state=random_state,
        stratify=combined_df[outcomes],
    )

    X_test, X_val = train_test_split(
        X_intermediate,
        test_size=0.5,
        random_state=random_state,
        stratify=X_intermediate[outcomes],
    )

    X_train["dw_ek_borger"].to_csv("splits/train_ids.csv", index=False)
    X_val["dw_ek_borger"].to_csv("splits/val_ids.csv", index=False)
    X_test["dw_ek_borger"].to_csv("splits/test_ids.csv", index=False)
    msg.good("Splits complete!")
