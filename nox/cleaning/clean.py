import warnings
import numpy as np
import pandas as pd

from sklearn.impute import MissingIndicator, KNNImputer

from nox.cleaning import util
from nox.cleaning import categorical as cat
from nox.cleaning import numeric as num
from nox.util import _input_columns, _numeric_input_conditions

# Create cleaning code by creating different func
class Clean(object):

    
    def drop_column_missing_threshold(self, threshold: float):

        if threshold > 1 or threshold < 0:
            raise ValueError("Threshold cannot be greater than 1 or less than 0.")

        criteria_meeting_columns = self.train_data.columns[
            self.train_data.isnull().mean() < threshold
        ]

        self.train_data = self.train_data[criteria_meeting_columns]

        if self.test_data is not None:
            self.test_data = self.test_data[criteria_meeting_columns]

        return self
