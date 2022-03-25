from testing_utils import *
from sklearn.model_selection import train_test_split


def test_stratified_split():
    in_df_str = """dw_ek_borger,outcome1_name,outcome2_name
                        1, 1, 0
                        2, 1, 0
                        3, 1, 0
                        4, 1, 0
                        5, 0, 1
                        6, 0, 1
                        7, 0, 1
                        8, 0, 1
                        9, 0, 1
                        10, 0, 1
                        """
    in_df = str_to_df(in_df_str)

    X_train, X_test = train_test_split(
        in_df["dw_ek_borger"],
        test_size=0.50,
        random_state=42,
        stratify=in_df[["outcome1_name", "outcome2_name"]],
    )

    pass


test_stratified_split()
