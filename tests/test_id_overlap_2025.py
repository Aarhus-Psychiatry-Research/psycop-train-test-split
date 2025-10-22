from pathlib import Path

import pandas as pd

SPLIT_PATH = Path(__file__).parent.parent / "splits"


def test_overlap_2025(split_path: str):
    """Check for overlapping ids in splits"""

    train = pd.read_csv(split_path / "train_ids_2025.csv")["dw_ek_borger"]
    test = pd.read_csv(split_path / "test_ids_2025.csv")["dw_ek_borger"]
    val = pd.read_csv(split_path / "val_ids_2025.csv")["dw_ek_borger"]

    assert val[val.isin(train)].empty
    assert test[test.isin(train)].empty


if __name__ == "__main__":
    SPLIT_PATH = Path(__file__).parent.parent / "splits"

    test_overlap_2025(SPLIT_PATH)
