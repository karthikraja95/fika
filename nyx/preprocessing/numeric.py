import pandas as pd
from nyx.util import _numeric_input_conditions, drop_replace_columns
from sklearn.preprocessing import MinMaxScaler, RobustScaler

SCALER = {"minmax": MinMaxScaler, "robust": RobustScaler}