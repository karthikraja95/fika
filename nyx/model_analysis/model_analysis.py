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

    def to_service(self, project_name: str):
        """
        Creates an app.py, requirements.txt and Dockerfile in `~/.nyx/projects` and the necessary folder structure
        to run the model as a microservice.
        
        Parameters
        ----------
        project_name : str
            Name of the project that you want to create.
        Examples
        --------
        >>> m = Model(df)
        >>> m_results = m.LogisticRegression()
        >>> m_results.to_service('your_proj_name')
        """

        to_pickle(self.model, self.model_name, project=True, project_name=project_name)
        tg.generate_service(project_name, f"{self.model_name}.pkl", self.model)

        print("To run:")
        print("\tdocker build -t `image_name` ./")
        print("\tdocker run -d --name `container_name` -p `port_num`:80 `image_name`")

class SupervisedModelAnalysis(ModelAnalysisBase):
    def __init__(self, model, x_train, x_test, y_train, y_test, model_name):

        self.model = model
        self.model_name = model_name
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.features = x_test.columns
        self.y_pred = self.model.predict(
            self.x_test[self.features]
        )  # Specifying columns for XGBoost
        self.run_id = None

        if hasattr(model, "predict_proba"):
            self.probabilities = self.model.predict_proba(self.x_test[self.features])

        self.shap = Shap(
            self.model,
            self.model_name,
            self.x_train,
            self.x_test,
            self.y_test,
            SHAP_LEARNERS[type(self.model)],
        )
        self.interpret = MSFTInterpret(
            self.model,
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            PROBLEM_TYPE[type(self.model)],
        )



    

