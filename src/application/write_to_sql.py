from psycopmlutils.writers.sql_writer import write_df_to_sql

import pandas as pd
from pathlib import Path

from psycopmlutils.loaders.raw.sql_load import sql_load


def load_split(split):
    return pd.read_csv(SPLIT_PATH / (f"{split}_ids.csv"))


if __name__ == "__main__":
    SPLIT_PATH = Path(__file__).parent.parent.parent / "splits"

    for split in ["train", "val", "test"]:
        df = load_split(split)

        # Check for duplicates in df
        duplicates = df[df.duplicated()]
        print(f"{split}: {duplicates.shape[0]} duplicates")

        table_name = f"psycop_{split}_ids"

        # Delete the sql table
        try:
            sql_load(query=f"DROP TABLE [fct].[psycop_{split}_ids]")
        except Exception as e:
            print(e)

        # Write the dataframe to the SQL server
        print(f"Writing {table_name} to SQL server")

        write_df_to_sql(
            df, table_name=table_name, rows_per_chunk=5000, if_exists="replace"
        )
