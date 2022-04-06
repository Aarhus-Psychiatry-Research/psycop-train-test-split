from typing import List, Union
import pandas as pd
import numpy as np
from psycoptts.stratify_by_each_category_individually import (
    stratified_split_by_each_category,
)

import random
import pytest

binary = [0, 1]
common_props = [0.95, 0.05]


def test_props():
    outcomes = ["cancer", "t2d", "schizophrenia"]
    n = 1000
    test_prop = 0.3
    common_prop = 0.2
    schizo_prop = 0.01

    common_values = get_list_of_n_01_with_prop_equal_1(prop=common_prop, n=n)
    uncommon_values = get_list_of_n_01_with_prop_equal_1(prop=schizo_prop, n=n)

    # Benchmark and test
    t2d_props = []
    schizo_props = []

    for i in range(1, 50):
        unsplit_df = pd.DataFrame()

        for outcome in outcomes:
            random.shuffle(common_values)
            random.shuffle(uncommon_values)

            if outcome != "schizophrenia":
                unsplit_df[outcome] = common_values
            elif outcome == "schizophrenia":
                unsplit_df[outcome] = uncommon_values

        train, test = stratified_split_by_each_category(
            df=unsplit_df, test_prop=test_prop, stratify_cols=outcomes
        )

        t2d_props.append(test[test["t2d"] == 1].shape[0] / (n * test_prop))
        schizo_props.append(test[test["schizophrenia"] == 1].shape[0] / (n * test_prop))

    avg_t2d_prop = sum(t2d_props) / len(t2d_props)
    assert avg_t2d_prop == pytest.approx(common_prop, abs=0.01)

    avg_schizo_prop = sum(schizo_props) / len(schizo_props)
    assert avg_schizo_prop == pytest.approx(schizo_prop, abs=0.01)


def get_list_of_n_01_with_prop_equal_1(prop, n):
    vals = [0 for i in range(int(n * (1 - prop)))]
    vals += [1 for i in range(int(n * prop))]

    return vals


def get_proportion_of_list_equal_to_val(val: Union[float, int], list: List) -> float:
    """Get proportion of items in list that are equal to val

    Args:
        val (Union[float, int]): Value to check for

    Returns:
        float: The proportion of items matching the condition
    """
    return len([i for i in list if i == val]) / len(list)
