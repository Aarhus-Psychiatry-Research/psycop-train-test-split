from pathlib import Path

import pandas as pd
import pytest

SPLIT_PATH = Path(__file__).parent.parent / "splits"


def _load_split(split: str):
    return pd.read_csv(SPLIT_PATH / f"{split}_ids.csv")["dw_ek_borger"]


def test_overlap():
    """Check for overlapping ids in splits"""
    if not SPLIT_PATH.exists():
        pytest.skip(reason="No splits directory")

    train = _load_split("train")
    val = _load_split("val")
    test = _load_split("test")

    assert val[val.isin(train)].empty
    assert test[test.isin(train)].empty
