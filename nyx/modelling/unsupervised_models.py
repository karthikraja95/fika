import pandas as pd

from nyx.modelling.model import ModelBase
from nyx.config import shell
from nyx.model_analysis.unsupervised_model_analysis import UnsupervisedModelAnalysis
from nyx.analysis import Analysis
from nyx.cleaning.clean import Clean
from nyx.preprocessing.preprocess import Preprocess
from nyx.feature_engineering.feature import Feature
from nyx.visualizations.visualizations import Visualizations
from nyx.stats.stats import Stats
from nyx.modelling.util import add_to_queue

class Unsupervised(
    ModelBase, Analysis, Clean, Preprocess, Feature, Visualizations, Stats
):
    def __init__(
        self, x_train, exp_name="my-experiment",
    ):
        """
        Class to run analysis, transform your data and run Unsupervised algorithms.
        Parameters
        -----------
        x_train: pd.DataFrame
            Training data or aethos data object
        exp_name : str
            Experiment name to be tracked in MLFlow.
        """

        super().__init__(
            x_train, "", x_test=None, test_split_percentage=0.2, exp_name=exp_name,
        )