import pandas as pd
from nox.util import _get_columns, _numeric_input_conditions, drop_replace_columns
from sklearn.impute import SimpleImputer

def replace_missing_mean_median_mode(
    x_train, x_test=None, list_of_cols=[], strategy=""
):

    if strategy != "most_frequent":
        list_of_cols = _numeric_input_conditions(list_of_cols, x_train)
    else:
        list_of_cols = _get_columns(list_of_cols, x_train)

    imp = SimpleImputer(strategy=strategy)

    fit_data = imp.fit_transform(x_train[list_of_cols])
    fit_df = pd.DataFrame(fit_data, columns=list_of_cols)
    x_train = drop_replace_columns(x_train, list_of_cols, fit_df)

    if x_test is not None:
        fit_x_test = imp.transform(x_test[list_of_cols])
        fit_test_df = pd.DataFrame(fit_x_test, columns=list_of_cols)
        x_test = drop_replace_columns(x_test, list_of_cols, fit_test_df)

    return x_train, x_test