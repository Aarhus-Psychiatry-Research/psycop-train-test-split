import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    stratify_cols = ["outcome1", "outcome2"]
    random_state = 42

    # TODO
    # Generate in_df from the csvs using add_outcome_from_csv()

    X_train, X_intermediate = train_test_split(
        in_df["dw_ek_borger"],
        test_size=0.4,
        random_state=random_state,
        stratify=in_df[stratify_cols],
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
