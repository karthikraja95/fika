import copy
import os
import re

import ipywidgets as widgets
import numpy as np
import pandas as pd

from nyx.config import shell
from nyx.stats.stats import Stats
from nyx.util import (
    CLEANING_CHECKLIST,
    DATA_CHECKLIST,
    ISSUES_CHECKLIST,
    MULTI_ANALYSIS_CHECKLIST,
    PREPARATION_CHECKLIST,
    UNI_ANALYSIS_CHECKLIST,
    _get_columns,
    _get_attr_,
    _get_item_,
    _interpret_data,
    label_encoder,
)
from nyx.visualizations.visualizations import Visualizations
from IPython import get_ipython
from IPython.display import HTML, display
from ipywidgets import Layout


class Analysis(Visualizations, Stats):
    """
    Core class thats run analytical techniques.
    Parameters
    -----------
    x_train: pd.DataFrame
        Training data or aethos data object
    x_test: pd.DataFrame
        Test data, by default None
    target: str
        For supervised learning problems, the name of the column you're trying to predict.
    """

    def __init__(
        self, x_train, x_test=None, target="",
    ):

        self.x_train = x_train
        self.x_test = x_test
        self.target = target
        self.target_mapping = None

    def __repr__(self):

        return self.x_train.head().to_string()

    def _repr_html_(self):  # pragma: no cover

        if self.target:
            cols = self.features + [self.target]
        else:
            cols = self.features

        return self.x_train[cols].head()._repr_html_()

    def __getitem__(self, column):
        return _get_item_(self, column)

    def __getattr__(self, key):
        return _get_attr_(self, key)


    def __deepcopy__(self, memo):

        x_test = self.x_test.copy() if self.x_test is not None else None

        new_inst = type(self)(
            x_train=self.x_train.copy(), x_test=x_test, target=self.target,
        )

        new_inst.target_mapping = self.target_mapping

        return new_inst

    @property
    def features(self):
        """Features for modelling"""

        cols = self.x_train.columns.tolist()

        if self.target:
            cols.remove(self.target)

        return cols

    @property
    def y_test(self):
        """
        Property function for the testing predictor variable
        """

        if self.x_test is not None:
            if self.target:
                return self.x_test[self.target]
            else:
                return None
        else:
            return None

    @y_test.setter
    def y_test(self, value):
        """
        Setter function for the testing predictor variable
        """

        if self.x_test is not None:
            if self.target:
                self.x_test[self.target] = value
            else:
                self.target = "label"
                self.x_test["label"] = value
                print('Added a target (predictor) field (column) named "label".')


    @property
    def y_train(self):
        """
        Property function for the training predictor variable
        """

        return self.x_train[self.target] if self.target else None

    @y_train.setter
    def y_train(self, value):
        """
        Setter function for the training predictor variable
        """

        if self.target:
            self.x_train[self.target] = value
        else:
            self.target = "label"
            self.x_train["label"] = value
            print('Added a target (predictor) field (column) named "label".')

    @property
    def columns(self):
        """
        Property to return columns in the dataset.
        """

        return self.x_train.columns.tolist()

    @property
    def missing_values(self):
        """
        Property function that shows how many values are missing in each column.
        """

        dataframes = list(
            filter(lambda x: x is not None, [self.x_train, self.x_test,],)
        )

        missing_df = []
        for ind, dataframe in enumerate(dataframes):
            caption = (
                "Train set missing values." if ind == 0 else "Test set missing values."
            )

            if not dataframe.isnull().values.any():
                print("No missing values!")  # pragma: no cover
            else:
                total = dataframe.isnull().sum().sort_values(ascending=False)
                percent = (
                    dataframe.isnull().sum() / dataframe.isnull().count()
                ).sort_values(ascending=False)
                missing_data = pd.concat(
                    [total, percent], axis=1, keys=["Total", "Percent"]
                )

                missing_df.append(
                    missing_data.style.format({"Percent": "{:.2%}"}).set_caption(
                        caption
                    )
                )