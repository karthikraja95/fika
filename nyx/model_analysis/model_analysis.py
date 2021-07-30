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

    def model_weights(self):
        """
        Prints and logs all the features ranked by importance from most to least important.
        
        Returns
        -------
        dict
            Dictionary of features and their corresponding weights
        
        Raises
        ------
        AttributeError
            If model does not have coefficients to display
        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.model_weights()
        """

        report_strings = []

        try:
            model_dict = dict(zip(self.features, self.model.coef_.flatten()))
        except Exception as e:
            raise AttributeError("Model does not have coefficients to view.")

        sorted_features = OrderedDict(
            sorted(model_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)
        )

        for feature, weight in sorted_features.items():
            report_string = "\t{} : {:.2f}".format(feature, weight)
            report_strings.append(report_string)

            print(report_string.strip())

    def summary_plot(self, output_file="", **summaryplot_kwargs):
        """
        Create a SHAP summary plot, colored by feature values when they are provided.
        For a list of all kwargs please see the Shap documentation : https://shap.readthedocs.io/en/latest/#plots
        Parameters
        ----------
        output_file: str
            Output file name including extension (.png, .jpg, etc.) to save image as.
        max_display : int
            How many top features to include in the plot (default is 20, or 7 for interaction plots), by default None
            
        plot_type : "dot" (default for single output), "bar" (default for multi-output), "violin", or "compact_dot"
            What type of summary plot to produce. Note that "compact_dot" is only used for SHAP interaction values.
        color : str or matplotlib.colors.ColorMap 
            Color spectrum used to draw the plot lines. If str, a registered matplotlib color name is assumed.
        axis_color : str or int 
            Color used to draw plot axes.
        title : str 
            Title of the plot.
        alpha : float 
            Alpha blending value in [0, 1] used to draw plot lines.
        show : bool 
            Whether to automatically display the plot.
        sort : bool
            Whether to sort features by importance, by default True
        color_bar : bool 
            Whether to draw the color bar.
        auto_size_plot : bool 
            Whether to automatically size the matplotlib plot to fit the number of features displayed. If False, specify the plot size using matplotlib before calling this function.
        layered_violin_max_num_bins : int
            Max number of bins, by default 20
        **summaryplot_kwargs
            For more info see https://shap.readthedocs.io/en/latest/#plots
        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.summary_plot()
        """

        if self.shap is None:
            raise NotImplementedError(
                f"SHAP is not implemented yet for {str(type(self))}"
            )

        self.shap.summary_plot(output_file=output_file, **summaryplot_kwargs)

        if _global_config["track_experiments"]:  # pragma: no cover
            track_artifacts(self.run_id, self.model_name)



    

