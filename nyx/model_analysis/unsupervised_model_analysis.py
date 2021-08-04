import os

from nyx.config.config import _global_config
from nyx.modelling.util import track_artifacts
from .model_analysis import ModelAnalysisBase

class UnsupervisedModelAnalysis(ModelAnalysisBase):
    def __init__(self, model, data, model_name):
        """
        Class to analyze Unsupervised models through metrics and visualizations.
        Parameters
        ----------
        model : str or Model Object
            Sklearn Model object or .pkl file of the object.
        data : pd.DataFrame
            Training Data used for the model.
        model_name : str
            Name of the model for saving images and model tracking purposes
        """

        self.model = model
        self.x_train = data
        self.model_name = model_name
        self.cluster_col = "predicted"

        if hasattr(self.model, "predict"):
            self.y_pred = self.model.predict(self.x_train)
        else:
            self.y_pred = self.model.fit_predict(self.x_train)

        self.x_train[self.cluster_col] = self.y_pred