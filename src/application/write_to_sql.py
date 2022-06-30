from psycopmlutils.writers.sql_writer import write_df_to_sql

import pandas as pd
from pathlib import Path


def load_split(split):
    return pd.read_csv(SPLIT_PATH / (f"{split}_ids.csv"))


if __name__ == "__main__":

    SPLIT_PATH = Path(__file__).parent.parent.parent / "splits"

    for split in ["train", "val", "test"]:
        df = load_split(split)
        table_name = f"[psycop_{split}_ids]"
        write_df_to_sql(
            df, table_name=table_name, rows_per_chunk=5000, if_exists="replace"
        )

