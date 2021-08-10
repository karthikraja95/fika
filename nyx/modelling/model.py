import os
import warnings
import pandas as pd
import sklearn
import numpy as np
import copy


from pathlib import Path
from IPython.display import display
from ipywidgets import widgets
from ipywidgets.widgets.widget_layout import Layout

from nyx.config import shell
from nyx.config.config import _global_config
from nyx.model_analysis.unsupervised_model_analysis import UnsupervisedModelAnalysis
from nyx.model_analysis.text_model_analysis import TextModelAnalysis
from nyx.modelling import text
from nyx.modelling.util import (
    _get_cv_type,
    _make_img_project_dir,
    _run_models_parallel,
    add_to_queue,
    run_crossvalidation,
    run_gridsearch,
    to_pickle,
    track_model,
)
from nyx.templates.template_generator import TemplateGenerator as tg
from nyx.util import _input_columns, split_data, _get_attr_, _get_item_

warnings.simplefilter("ignore", FutureWarning)

class ModelBase(object):
    def __init__(
        self,
        x_train,
        target,
        x_test=None,
        test_split_percentage=0.2,
        exp_name="my-experiment",
    ):

        self._models = {}
        self._queued_models = {}
        self.exp_name = exp_name

        problem = "c" if type(self).__name__ == "Classification" else "r"

        self.x_train = x_train
        self.x_test = x_test
        self.target = target
        self.test_split_percentage = test_split_percentage
        self.target_mapping = None

        if self.x_test is None and not type(self).__name__ == "Unsupervised":
            # Generate train set and test set.
            self.x_train, self.x_test = split_data(
                self.x_train, test_split_percentage, self.target, problem
            )
            self.x_train = self.x_train.reset_index(drop=True)
            self.x_test = self.x_test.reset_index(drop=True)

    def __getitem__(self, key):

        return _get_item_(self, key)

    def __getattr__(self, key):

        # For when doing multi processing when pickle is reconstructing the object
        if key in {"__getstate__", "__setstate__"}:
            return object.__getattr__(self, key)

        if key in self._models:
            return self._models[key]

        return _get_attr_(self, key)

    def __setattr__(self, key, value):

        if key not in self.__dict__ or hasattr(self, key):
            # any normal attributes are handled normally
            dict.__setattr__(self, key, value)
        else:
            self.__setitem__(key, value)

    def __setitem__(self, key, value):

        if key in self.__dict__:
            dict.__setitem__(self.__dict__, key, value)

    def __repr__(self):

        return self.x_train.head().to_string()

    def _repr_html_(self):  # pragma: no cover

        if self.target:
            cols = self.features + [self.target]
        else:
            cols = self.features

        return self.x_train[cols].head()._repr_html_()


    def __deepcopy__(self, memo):

        x_test = self.x_test.copy() if self.x_test is not None else None

        new_inst = type(self)(
            x_train=self.x_train.copy(),
            target=self.target,
            x_test=x_test,
            test_split_percentage=self.test_split_percentage,
            exp_name=self.exp_name,
        )

        new_inst.target_mapping = self.target_mapping
        new_inst._models = self._models
        new_inst._queued_models = self._queued_models

        return new_inst

    @property
    def features(self):
        """Features for modelling"""

        cols = self.x_train.columns.tolist()

        if self.target:
            cols.remove(self.target)

        return cols

    @property
    def train_data(self):
        """Training data used for modelling"""

        return self.x_train[self.features]

    @train_data.setter
    def train_data(self, val):
        """Setting for train_data"""

        val[self.target] = self.y_train
        self.x_train = val

    @property
    def test_data(self):
        """Testing data used to evaluate models"""

        return self.x_test[self.features] if self.x_test is not None else None

    @test_data.setter
    def test_data(self, val):
        """Test data setter"""

        val[self.target] = self.y_test
        self.x_test = val

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
    def columns(self):
        """
        Property to return columns in the dataset.
        """

        return self.x_train.columns.tolist()