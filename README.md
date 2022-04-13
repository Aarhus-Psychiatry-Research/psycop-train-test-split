# psycop-train-test-split

[![python versions](https://img.shields.io/badge/Python-%3E=3.7-blue)](https://github.com/centre-for-humanities-computing/conspiracies)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)

Code for splitting data derived from the PSYCOP project into train test splits.

If each combination of stratification_cols is considered a stratum, some strata might have only one member. E.g. there may only be one patient that gets both lung_cancer and schizophrenia.

But we don't care about each combination being balanced, we only care about each category being balanced individually.

E.g. if we have two binary variables, cancer and schizophrenia:

Unsplit

patient_id	cancer	schizophrenia
1	1	1
2	1	0
3	0	1
One of these categories, cancer + schizophrenia has only one individual – and thus can't get balanced! sklearn then outputs

ValueError: The least populated class in y has only 1 member, which is too few. 
The minimum number of groups for any class cannot be less than 2.
But we don't care about that, we just want both cancer and schizophrenia to be balanced. A split like this would be great for us:

Train

patient_id	cancer	schizophrenia
1	1	1
Test

patient_id	cancer	schizophrenia
2	1	0
3	0	1
This is balanced, because in both train and test there's exactly one individual with cancer, and one with schizophrenia.
