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