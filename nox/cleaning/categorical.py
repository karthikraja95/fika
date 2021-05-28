import numpy as np
from nox.util import _get_columns


def _determine_default_category(x_train, col, replacement_categories):
    
    """
    A utility function to help determine the default category name for a column that has missing
    categorical values. 
    
    It takes in a list of possible values and if any the first value in the list
    that is not a value in the column is the category that will be used to replace missing values.
    """

    unique_vals_col = x_train[col].unique()
    for potential_category in replacement_categories:

        # If the potential category is not already a category, it becomes the default missing category
        if potential_category not in unique_vals_col:
            new_category_name = potential_category
            break

    return new_category_name