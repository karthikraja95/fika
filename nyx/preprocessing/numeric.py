import pandas as pd
from nyx.util import _numeric_input_conditions, drop_replace_columns
from sklearn.preprocessing import MinMaxScaler, RobustScaler

SCALER = {"minmax": MinMaxScaler, "robust": RobustScaler}

def scale(
    x_train,
    x_test=None,
    list_of_cols=[],
    method="minmax",
    keep_col=False,
    **algo_kwargs
):

    list_of_cols = _numeric_input_conditions(list_of_cols, x_train)
    scaler = SCALER[method](**algo_kwargs)

    scaled_data = scaler.fit_transform(x_train[list_of_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=list_of_cols)
    x_train = drop_replace_columns(x_train, list_of_cols, scaled_df, keep_col=keep_col)

    if x_test is not None:
        scaled_x_test = scaler.transform(x_test[list_of_cols])
        scaled_test_df = pd.DataFrame(scaled_x_test, columns=list_of_cols)
        x_test = drop_replace_columns(
            x_test, list_of_cols, scaled_test_df, keep_col=keep_col
        )

    return x_train, x_test