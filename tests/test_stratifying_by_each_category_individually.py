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
    outcomes = {"common": ["cancer", "t2d"], "uncommon": ["schizophrenia"]}
    all_outcomes = outcomes["common"] + outcomes["uncommon"]

    n = 12_000
    split_props = {"train": 0.7, "test": 0.3}
    outcome_type_props = {"common": 0.02, "uncommon": 0.004}

    # Generate list of values to sample from
    values = {
        outcome_type: create_list_of_vals_with_prop_of_ones(
            prop=outcome_type_props[outcome_type], n=n
        )
        for outcome_type in ["common", "uncommon"]
    }

    # Initialise
    expected_props = defaultdict(lambda: 0)
    simulated_props = {c: defaultdict(list) for c in ["train", "test"]}

    for i in range(5):
        unsplit_df = pd.DataFrame()

        for outcome_type in ["common", "uncommon"]:
            for outcome in outcomes[outcome_type]:
                # Generate expected props and values for the current outcome
                expected_props[outcome] = outcome_type_props[outcome_type]
                random.shuffle(values[outcome_type])
                unsplit_df[outcome] = values[outcome_type]

        splits = {"train": None, "test": None}

        splits["train"], splits["test"] = stratified_split_by_each_category(
            df=unsplit_df, test_size=split_props["test"], stratify=all_outcomes
        )

        for outcome in all_outcomes:
            for split_type in ["train", "test"]:
                split = splits[split_type]

                n_with_outcome_in_split = split[split[outcome] == 1].shape[0]
                simulated_prop = n_with_outcome_in_split / (
                    n * (split_props[split_type])
                )

                simulated_props[split_type][outcome].append(simulated_prop)

    # Aggregate proportions and get avg. across simulations
    print("\n")
    for split_type in ["test", "train"]:
        for outcome in all_outcomes:
            simulated_test_props = simulated_props[split_type][outcome]
            avg_simulated_outcome_prop = sum(simulated_test_props) / len(
                simulated_test_props
            )
            print(
                f"    {split_type}: {expected_props[outcome]} | {avg_simulated_outcome_prop} – {outcome} – (Expected | Simulated)"
            )
            assert avg_simulated_outcome_prop == pytest.approx(
                expected_props[outcome], rel=0.05
            )


def create_list_of_vals_with_prop_of_ones(prop, n):
    vals = [0 for i in range(int(n * (1 - prop)))]
    vals += [1 for i in range(int(n * prop))]

    # Fix in cases where rounding means that len of vals is smaller than n
    if len(vals) != n:
        vals += [0]

    return vals


def get_proportion_of_list_equal_to_val(val: Union[float, int], list: List) -> float:
    """Get proportion of items in list that are equal to val

    Args:
        val (Union[float, int]): Value to check for

    Returns:
        float: The proportion of items matching the condition
    """
    return len([i for i in list if i == val]) / len(list)
