from collections import defaultdict
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
    common_outcomes = ["cancer", "t2d"]
    uncommon_outcomes = ["schizophrenia"]
    all_outcomes = common_outcomes + uncommon_outcomes

    n = 120_000
    test_prop = 0.3
    common_prop = 0.02
    uncommon_prop = 0.004

    # Generate list of values to sample from
    common_values = get_list_of_n_01_with_prop_equal_1(prop=common_prop, n=n)
    uncommon_values = get_list_of_n_01_with_prop_equal_1(prop=uncommon_prop, n=n)

    # Initialise
    expected_props = defaultdict(lambda: 0)
    simulated_test_props = defaultdict(list)

    for i in range(1):
        unsplit_df = pd.DataFrame()

        for outcome in common_outcomes:
            expected_props[outcome] = common_prop
            random.shuffle(common_values)
            unsplit_df[outcome] = common_values

        for outcome in uncommon_outcomes:
            expected_props[outcome] = uncommon_prop
            random.shuffle(uncommon_values)
            unsplit_df[outcome] = uncommon_values

        train, test = stratified_split_by_each_category(
            df=unsplit_df, test_prop=test_prop, stratify_cols=common_outcomes
        )

        for outcome in all_outcomes:
            outcome_prop_of_test = test[test[outcome] == 1].shape[0] / (n * test_prop)
            simulated_test_props[outcome].append(outcome_prop_of_test)

    # Aggregate proportions and get avg. across simulations
    for outcome in all_outcomes:
        simulated_test_prop = simulated_test_props[outcome]
        avg_outcome_prop = sum(simulated_test_prop) / len(simulated_test_prop)
        assert avg_outcome_prop == pytest.approx(expected_props[outcome], rel=0.05)


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


test_props()
