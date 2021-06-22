import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
import scipy as sc

from scipy.stats.stats import ks_2samp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from collections import Counter
from typing import Union
from nyx.stats.util import run_2sample_ttest


class Stats(object):


    def predict_data_sample(self):

        """
        Identifies how similar the train and test set distribution are by trying to predict whether each sample belongs
        to the train or test set using Random Forest, 10 Fold Stratified Cross Validation.
        The lower the F1 score, the more similar the distributions are as it's harder to predict which sample belongs to which distribution.
        Credit: https://www.kaggle.com/nanomathias/distribution-of-test-vs-training-data#1.-t-SNE-Distribution-Overview
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.predict_data_sample()
        """

        if self.x_test is None or not self.target:
            raise ValueError(
                "Test data or target field must be set. They can be set by assigning values to the `target` or the `x_test` variable."
            )

        x_train = self.x_train.drop(self.target, axis=1)
        x_test = self.x_test.drop(self.target, axis=1)

        x_train["label"] = 1
        x_test["label"] = 0

        data = pd.concat([x_train, x_test], axis=0)
        label = data["label"].tolist()

        predictions = cross_val_predict(
            ExtraTreesClassifier(n_estimators=100),
            data.drop(columns=["label"]),
            label,
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        )

        print(classification_report(data["label"].tolist(), predictions))

        return self

    def ks_feature_distribution(self, threshold=0.1, show_plots=True):

        import swifter
        from tqdm import tqdm

       
     

        
                        
