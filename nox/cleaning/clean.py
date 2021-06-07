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

        """
        Remove columns from the dataframe that have greater than or equal to the threshold value of missing values.
        Example: Remove columns where >= 50% of the data is missing.
        
        Parameters
        ----------
        threshold : float
            Value between 0 and 1 that describes what percentage of a column can be missing values.
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.drop_column_missing_threshold(0.5)
        """

        if threshold > 1 or threshold < 0:
            raise ValueError("Threshold cannot be greater than 1 or less than 0.")

        criteria_meeting_columns = self.train_data.columns[
            self.train_data.isnull().mean() < threshold
        ]

        self.train_data = self.train_data[criteria_meeting_columns]

        if self.test_data is not None:
            self.test_data = self.test_data[criteria_meeting_columns]

        return self


    def drop_constant_columns(self):

        """
        Remove columns from the data that only have one unique value.
                
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.drop_constant_columns()
        """        

        # If the number of unique values is not 0(all missing) or 1(constant or constant + missing)
        keep_columns = []

        for col in self.train_data.columns:
            try:
                if self.train_data.nunique()[col] not in [0, 1]:
                    keep_columns.append(col)
            except Exception as e:
                print(f"Column {col} could not be processed.")

        self.train_data = self.train_data[keep_columns]

        if self.test_data is not None:
            self.test_data = self.test_data[keep_columns]

        return self


    def drop_unique_columns(self):

        """
        Remove columns from the data that only have one unique value.
                
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.drop_unique_columns()
        """

        # If the number of unique values is not 0(all missing) or 1(constant or constant + missing)
        keep_columns = list(
            filter(
                lambda x: self.train_data.nunique()[x] != self.train_data.shape[0],
                self.train_data.columns,
            )
        )

        self.train_data = self.train_data[keep_columns]

        if self.test_data is not None:
            self.test_data = self.test_data[keep_columns]

        return self       

    def drop_rows_missing_threshold(self, threshold: float):
        
