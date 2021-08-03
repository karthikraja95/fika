import numpy as np
import pandas as pd
import sklearn
import warnings
import math

from nyx.config.config import _global_config
from .model_analysis import SupervisedModelAnalysis

class RegressionModelAnalysis(SupervisedModelAnalysis):
    def __init__(
        self, model, x_train, x_test, target, model_name,
    ):
        """
        Class to analyze Regression models through metrics, global/local interpretation and visualizations.
        Parameters
        ----------
        model : str or Model Object
            Sklearn, XGBoost, LightGBM Model object or .pkl file of the objects.
        x_train : pd.DataFrame
            Training Data used for the model.
        x_test : pd.DataFrame
            Test data used for the model.
        target : str
            Target column in the DataFrame
        model_name : str
            Name of the model for saving images and model tracking purposes
        """

        # TODO: Add check for pickle file

        super().__init__(
            model,
            x_train.drop([target], axis=1),
            x_test.drop([target], axis=1),
            x_train[target],
            x_test[target],
            model_name,
        )

    def plot_predicted_actual(self, output_file="", **scatterplot_kwargs):
        """
        Plots the actual data vs. predictions
        Parameters
        ----------
        output_file : str, optional
            Output file name, by default ""
        """

        self._viz.scatterplot(
            x="actual",
            y="predicted",
            data=self.test_results,
            title="Actual vs. Predicted",
            output_file=output_file,
            **scatterplot_kwargs
        )