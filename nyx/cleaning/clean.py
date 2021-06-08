import warnings
import numpy as np
import pandas as pd

from sklearn.impute import MissingIndicator, KNNImputer

from nyx.cleaning import util
from nyx.cleaning import categorical as cat
from nyx.cleaning import numeric as num
from nyx.util import _input_columns, _numeric_input_conditions

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

        """
        Remove rows from the dataframe that have greater than or equal to the threshold value of missing rows.
        Example: Remove rows where > 50% of the data is missing.
        Parameters
        ----------
        threshold : float
            Value between 0 and 1 that describes what percentage of a row can be missing values.
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.drop_rows_missing_threshold(0.5)    
        """

        if threshold > 1 or threshold < 0:
            raise ValueError("Threshold cannot be greater than 1 or less than 0.")

        self.train_data = self.train_data.dropna(
            thresh=round(self.train_data.shape[1] * threshold), axis=0
        )

        if self.test_data is not None:
            self.test_data = self.test_data.dropna(
                thresh=round(self.test_data.shape[1] * threshold), axis=0
            )

        return self

    def replace_missing_mean(self, *list_args, list_of_cols=[]):

        """
        Replaces missing values in every numeric column with the mean of that column.
        If no columns are supplied, missing values will be replaced with the mean in every numeric column.
        Mean: Average value of the column. Effected by outliers.
        If a list of columns is provided use the list, otherwise use arguemnts.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to
        list_of_cols : list, optional
            Specific columns to apply this technique to, by default []
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.replace_missing_mean('col1', 'col2')
        >>> data.replace_missing_mean(['col1', 'col2'])
        """

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.train_data, self.test_data,) = num.replace_missing_mean_median_mode(
            x_train=self.train_data,
            x_test=self.test_data,
            list_of_cols=list_of_cols,
            strategy="mean",
        )

        return self


