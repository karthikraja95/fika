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