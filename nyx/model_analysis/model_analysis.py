import itertools
import math
import os
import warnings
from collections import OrderedDict
from itertools import compress

import xgboost as xgb
import numpy as np
import pandas as pd
import sklearn
from IPython.display import HTML, SVG, display

from nyx.config.config import _global_config
from nyx.feature_engineering.util import sklearn_dim_reduction
from nyx.model_analysis.model_explanation import MSFTInterpret, Shap
from nyx.modelling.util import (
    to_pickle,
    track_artifacts,
    _get_cv_type,
    run_crossvalidation,
)
from nyx.templates.template_generator import TemplateGenerator as tg
from nyx.visualizations.visualizations import Visualizations
from nyx.stats.stats import Stats
from nyx.model_analysis.constants import (
    PROBLEM_TYPE,
    SHAP_LEARNERS,
)

class ModelAnalysisBase(Visualizations, Stats):

    # TODO: Add more SHAP use cases

    def _repr_html(self):

        if hasattr(self, "x_test"):
            data = self.test_results
        else:
            data = self.train_results

        return data

    @property
    def train_results(self):

        data = self.x_train.copy()
        data["predicted"] = self.model.predict(data)
        data["actual"] = self.y_train

        return data

    @property
    def test_results(self):

        data = self.x_test
        data["actual"] = self.y_test

        data["predicted"] = self.y_pred

        return data

    def to_pickle(self):
        """
        Writes model to a pickle file.
        Examples
        --------
        >>> m = Model(df)
        >>> m_results = m.LogisticRegression()
        >>> m_results.to_pickle()
        """

        to_pickle(self.model, self.model_name)

