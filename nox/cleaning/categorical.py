import numpy as np
from nox.util import _get_columns


def _determine_default_category(x_train, col, replacement_categories):


    unique_vals_col = x_train[col].unique()
    for potential_category in replacement_categories:

        # If the potential category is not already a category, it becomes the default missing category
        if potential_category not in unique_vals_col:
            new_category_name = potential_category
            break

    return new_category_name