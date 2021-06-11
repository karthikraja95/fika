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

    def replace_missing_median(self, *list_args, list_of_cols=[]):

        """
        Replaces missing values in every numeric column with the median of that column.
        If no columns are supplied, missing values will be replaced with the mean in every numeric column.
        Median: Middle value of a list of numbers. Equal to the mean if data follows normal distribution. Not effected much by anomalies.
        If a list of columns is provided use the list, otherwise use arguemnts.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            Specific columns to apply this technique to., by default []
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.replace_missing_median('col1', 'col2')
        >>> data.replace_missing_median(['col1', 'col2'])
        """


        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.train_data, self.test_data,) = num.replace_missing_mean_median_mode(
            x_train=self.train_data,
            x_test=self.test_data,
            list_of_cols=list_of_cols,
            strategy="median",
        )

        return self

    def replace_missing_mostcommon(self, *list_args, list_of_cols=[]):

        """
        Replaces missing values in every numeric column with the most common value of that column
        Mode: Most common value.
        If a list of columns is provided use the list, otherwise use arguemnts.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.replace_missing_mostcommon('col1', 'col2')
        >>> data.replace_missing_mostcommon(['col1', 'col2'])
        """

        ## If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.train_data, self.test_data,) = num.replace_missing_mean_median_mode(
            x_train=self.train_data,
            x_test=self.test_data,
            list_of_cols=list_of_cols,
            strategy="most_frequent",
        )

        return self


    def replace_missing_constant(
        self, *list_args, list_of_cols=[], constant=0, col_mapping=None
    ):

        """
        Replaces missing values in every numeric column with a constant.
        If no columns are supplied, missing values will be replaced with the mean in every numeric column.
        If a list of columns is provided use the list, otherwise use arguemnts.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        constant : int or float, optional
            Numeric value to replace all missing values with , by default 0
        col_mapping : dict, optional
            Dictionary mapping {'ColumnName': `constant`}, by default None
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.replace_missing_constant(col_mapping={'a': 1, 'b': 2, 'c': 3})
        >>> data.replace_missing_constant('col1', 'col2', constant=2)
        >>> data.replace_missing_constant(['col1', 'col2'], constant=3)
        
        """

    # If a list of columns is provided use the list, otherwise use arguemnts.
        if col_mapping:
            col_to_constant = col_mapping
        else:
            col_to_constant = _input_columns(list_args, list_of_cols)

        if isinstance(col_to_constant, dict):
            self.x_train, self.x_test = cat.replace_missing_new_category(
                x_train=self.x_train,
                x_test=self.x_test,
                col_to_category=col_to_constant,
            )
        elif isinstance(col_to_constant, list):
            self.x_train, self.x_test = cat.replace_missing_new_category(
                x_train=self.x_train,
                x_test=self.x_test,
                col_to_category=col_to_constant,
                constant=constant,
            )
        else:
            self.x_train, self.x_test = cat.replace_missing_new_category(
                x_train=self.x_train, x_test=self.x_test, constant=constant,
            )

        return self

    def replace_missing_new_category(
        self, *list_args, list_of_cols=[], new_category=None, col_mapping=None
    ):

        """
        Replaces missing values in categorical column with its own category. The categories can be autochosen
        from the defaults set.
        For numeric categorical columns default values are: -1, -999, -9999
        For string categorical columns default values are: "Other", "Unknown", "MissingDataCategory"
        If a list of columns is provided use the list, otherwise use arguemnts.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        new_category : str, int, or float, optional
            Category to replace missing values with, by default None
        col_mapping : dict, optional
           Dictionary mapping {'ColumnName': `constant`}, by default None
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.replace_missing_new_category(col_mapping={'col1': "Green", 'col2': "Canada", 'col3': "December"})
        >>> data.replace_missing_new_category('col1', 'col2', 'col3', new_category='Blue')
        >>> data.replace_missing_new_category(['col1', 'col2', 'col3'], new_category='Blue')
        """

        # If dictionary mapping is provided, use that otherwise use column
        if col_mapping:
            col_to_category = col_mapping
        else:
            # If a list of columns is provided use the list, otherwise use arguemnts.
            col_to_category = _input_columns(list_args, list_of_cols)

        self.x_train, self.x_test = cat.replace_missing_new_category(
            x_train=self.x_train,
            x_test=self.x_test,
            col_to_category=col_to_category,
            constant=new_category,
        )

        return self

    def replace_missing_remove_row(self, *list_args, list_of_cols=[]):


        """
        Remove rows where the value of a column for those rows is missing.
        If a list of columns is provided use the list, otherwise use arguemnts.
        
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.replace_missing_remove_row('col1', 'col2')
        >>> data.replace_missing_remove_row(['col1', 'col2'])
        """


        # If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        self.x_train = self.x_train.dropna(axis=0, subset=list_of_cols)

        if self.x_test is not None:
            self.x_test = self.x_test.dropna(axis=0, subset=list_of_cols)

        return self


    def drop_duplicate_rows(self, *list_args, list_of_cols=[]):


        """
        Remove rows from the data that are exact duplicates of each other and leave only 1.
        This can be used to reduce processing time or performance for algorithms where
        duplicates have no effect on the outcome (i.e DBSCAN)
        If a list of columns is provided use the list, otherwise use arguemnts.
       
        Parameters
        ----------
        list_args : str(s), optional
            Specific columns to apply this technique to.
        list_of_cols : list, optional
            A list of specific columns to apply this technique to., by default []
       
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.drop_duplicate_rows('col1', 'col2') # Only look at columns 1 and 2
        >>> data.drop_duplicate_rows(['col1', 'col2'])
        >>> data.drop_duplicate_rows()
        """

        # If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        self.x_train = self.x_train.drop_duplicates(list_of_cols)

        if self.x_test is not None:
            self.test_data = self.test_data.drop_duplicates(list_of_cols)

        return self

    def drop_duplicate_columns(self):

        """
        Remove columns from the data that are exact duplicates of each other and leave only 1.
        
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.drop_duplicate_columns()
        """

        self.train_data = self.train_data.T.drop_duplicates().T

        if self.test_data is not None:
            self.test_data = self.test_data.T.drop_duplicates().T

        return self



    def replace_missing_random_discrete(self, *list_args, list_of_cols=[]):

        # If a list of columns is provided use the list, otherwise use arguemnts.
        list_of_cols = _input_columns(list_args, list_of_cols)

        for col in list_of_cols:
            probabilities = self.x_train[col].value_counts(normalize=True)

            missing_data = self.x_train[col].isnull()
            self.x_train.loc[missing_data, col] = np.random.choice(
                probabilities.index,
                size=len(self.x_train[missing_data]),
                replace=True,
                p=probabilities.values,
            )

            if self.x_test is not None:
                missing_data = self.x_test[col].isnull()
                self.x_test.loc[missing_data, col] = np.random.choice(
                    probabilities.index,
                    size=len(self.x_test[missing_data]),
                    replace=True,
                    p=probabilities.values,
                )

        return self