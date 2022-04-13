from collections import defaultdict
import numpy as np
from pandas import DataFrame
from typing import Optional, List, Tuple, Union, Dict
import random


def stratified_split_by_each_category(
    df: DataFrame,
    test_prop: float,
    stratify_cols: Optional[List[str]] = None,
) -> Tuple[DataFrame, DataFrame]:
    """Splits a dataset into train, test, balancing each stratify_col individually.
    It doesn't consider each combination of stratify_col variables as unique.
    See README for an example.

    Args:
        df (DataFrame): Contains the ids to split, as well as columns to stratify on
        test_prop (Union[float, int]): Test proportion.
        stratify_cols (Optional[List[str]], optional): Which columns to stratify on. Defaults to None.

    Returns:
        Tuple[DataFrame, DataFrame]: Returns a tuple of train, test dfs.
    """
    no_cat_str = "no_cat"

    # Generate target ns
    target_n_in_test_by_cat = {
        strat_col: df[df[strat_col] == 1].shape[0] * test_size for strat_col in stratify
    }.copy()

    target_total_n_in_test = test_prop * df.shape[0]

    df["is_test"] = np.zeros(df.shape[0])

    # Create a stack of idx to pull from
    patient_ids_for_each_cat = {
        col: df.index[df[col] == 1].tolist() for col in stratify
    }

    # Set random state for reproducibility
    random.seed(random_state)

    # Shuffle the idx
    for key in patient_ids_for_each_cat.keys():
        random.shuffle(patient_ids_for_each_cat[key])

    # Mark idx that don't belong to any stratification-col
    df[no_cat_str] = np.where(df[stratify].sum(axis=1) == 0, 1, 0)
    patient_ids_for_each_cat[no_cat_str] = df.index[df[no_cat_str] == 1].tolist()

    # Initialise counters for while-loop
    n_in_test = 0
    n_in_test_by_category = defaultdict(lambda: 0)
    all_cats_reached_target_n_in_test = False

    while n_in_test < target_total_n_in_test:
        ratio_target_current_prop = {
            cat: n_in_test_by_category[cat] / target_n_in_test_by_cat[cat]
            for cat in stratify
        }

        if get_minimum_val_in_dict(ratio_target_current_prop) >= 1:
            all_cats_reached_target_n_in_test = True

        if all_cats_reached_target_n_in_test:
            cat_to_add_to = no_cat_str
        else:
            # Sample from the category that is the furthest from the desired prop
            cat_to_add_to = min(
                ratio_target_current_prop, key=ratio_target_current_prop.get
            )

        idx = patient_ids_for_each_cat[cat_to_add_to].pop()

        # If an idx belongs to multiple categories, also remove if from those that aren't currently cat_to_add_to
        # This is __super__ inefficient (95% of current runtime), but running for 120.000 idxs takes 45 seconds.
        # Acceptable runtime for current application.
        remove_value_from_all_lists_it_exists_in(
            value=idx, dict_of_lists=patient_ids_for_each_cat
        )

        # Increment counter in all cats the idx belongs to
        for cat in stratify:
            n_in_test_by_category[cat] += df[cat][idx]

        df.at[idx, "is_test"] = 1

        n_in_test += 1

    train = df[df["is_test"] == 0].drop(columns=["is_test", no_cat_str])
    test = df[df["is_test"] == 1].drop(columns=["is_test", no_cat_str])

    return train, test


def remove_value_from_all_lists_it_exists_in(
    value: Union[int, float], dict_of_lists: Dict[str, List]
):
    """Takes a dict and looks through all items in its values (lists). Removes any item that matches the value argument.

    Args:
        value (Union[int, float]): Float or integer to look for.
        dict_of_lists (Dict[str, List]): A dict of lists shaped like {"key1": [val1, val2], "key2": [val3, val4]}
    """
    for list_ in dict_of_lists.values():
        if value in list_:
            list_.remove(value)


def get_minimum_val_in_dict(dict: Dict[str, Union[float, int]]) -> str:
    """Find the smallest value for all keys in dict

    Args:
        dict (Dict[str, list]): Dictionary shaped like {"key1": [val1, val2]}

    Returns:
        str: The smallest val.
    """
    key_with_min_val = min(dict, key=dict.get)
    return dict[key_with_min_val]
