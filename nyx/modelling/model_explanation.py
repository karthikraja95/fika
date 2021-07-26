import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import warnings

from nyx.config import IMAGE_DIR

warnings.simplefilter("ignore", UserWarning)

class Shap(object):
    def __init__(self, model, model_name, x_train, x_test, y_test, learner: str):

        import lightgbm as lgb
        import shap

        self.model = model
        self.model_name = model_name
        self.x_train = x_train
        self.x_test = x_test
        self.y_test = y_test

        if learner == "linear":
            self.explainer = shap.LinearExplainer(
                self.model, self.x_train, feature_dependence="independent"
            )
        elif learner == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif learner == "kernel":
            if hasattr(self.model, "predict_proba"):
                func = self.model.predict_proba
            else:
                func = self.model.predict

            self.explainer = shap.KernelExplainer(func, self.x_train)
        else:
            raise ValueError(f"Learner: {learner} is not supported yet.")

        self.expected_value = self.explainer.expected_value
        self.shap_values = np.array(self.explainer.shap_values(self.x_test)).astype(
            float
        )

        if isinstance(self.model, lgb.sklearn.LGBMClassifier) and isinstance(
            self.expected_value, np.float
        ):
            self.shap_values = self.shap_values[1]

        # Calculate misclassified values
        self.misclassified_values = self._calculate_misclassified()

        # As per SHAP guidelines, test data needs to be dense for plotting functions
        self.x_test_array = self.x_test.values

    def summary_plot(self, output_file="", **summaryplot_kwargs):
        """
        Plots a SHAP summary plot.
        Parameters
        ----------
        output_file: str
            Output file name including extension (.png, .jpg, etc.) to save image as.
        """

        import shap

        shap.summary_plot(
            self.shap_values,
            self.x_test_array,
            feature_names=self.x_train.columns,
            show=False,
            **summaryplot_kwargs,
        )

        if output_file:  # pragma: no cover
            pl.savefig(os.path.join(IMAGE_DIR, self.model_name, output_file))

    def decision_plot(
        self, num_samples=0.25, sample_no=None, output_file="", **decisionplot_kwargs
    ):
        """
        Plots a SHAP decision plot.
        
        Parameters
        ----------
        num_samples : int, float, or 'all', optional
            Number of samples to display, if less than 1 it will treat it as a percentage, 'all' will include all samples
            , by default 0.25
        sample_no : int, optional
            Sample number to isolate and analyze, if provided it overrides num_samples, by default None
        Returns
        -------
        DecisionPlotResult 
            If return_objects=True (the default). Returns None otherwise.
        """

        import shap

        return_objects = decisionplot_kwargs.pop("return_objects", True)
        highlight = decisionplot_kwargs.pop("highlight", None)

        if sample_no is not None:
            if sample_no < 1 or not isinstance(sample_no, int):
                raise ValueError("Sample number must be greater than 1.")

            samples = slice(sample_no - 1, sample_no)
        else:
            if num_samples == "all":
                samples = slice(0, len(self.x_test_array))
            elif num_samples <= 0:
                raise ValueError(
                    "Number of samples must be greater than 0. If it is less than 1, it will be treated as a percentage."
                )
            elif num_samples > 0 and num_samples < 1:
                samples = slice(0, int(num_samples * len(self.x_test_array)))
            else:
                samples = slice(0, num_samples)

        if highlight is not None:
            highlight = highlight[samples]

        s = shap.decision_plot(
            self.expected_value,
            self.shap_values[samples],
            self.x_train.columns,
            return_objects=return_objects,
            highlight=highlight,
            show=False,
            **decisionplot_kwargs,
        )

        if output_file:  # pragma: no cover
            pl.savefig(os.path.join(IMAGE_DIR, self.model_name, output_file))

        return s
    
    def force_plot(self, sample_no=None, output_file="", **forceplot_kwargs):
        """
        Plots a SHAP force plot.
        """

        import shap

        shap_values = forceplot_kwargs.pop("shap_values", self.shap_values)

        if sample_no is not None:
            if sample_no < 1 or not isinstance(sample_no, int):
                raise ValueError("Sample number must be greater than 1.")

            samples = slice(sample_no - 1, sample_no)
        else:
            samples = slice(0, len(shap_values))

        s = shap.force_plot(
            self.expected_value,
            shap_values[samples],
            self.x_train.columns,
            **forceplot_kwargs,
        )

        if output_file:  # pragma: no cover
            pl.savefig(os.path.join(IMAGE_DIR, self.model_name, output_file))

        return s

    def dependence_plot(
        self, feature, interaction=None, output_file="", **dependenceplot_kwargs
    ):
        """
        Plots a SHAP dependence plot.
        """

        import shap

        interaction = dependenceplot_kwargs.pop("interaction_index", interaction)

        shap.dependence_plot(
            feature,
            self.shap_values,
            self.x_test,
            interaction_index=interaction,
            show=False,
            **dependenceplot_kwargs,
        )

        if output_file:  # pragma: no cover
            pl.savefig(os.path.join(IMAGE_DIR, self.model_name, output_file))
