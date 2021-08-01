import os
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics

from nyx.config import IMAGE_DIR
from nyx.config.config import _global_config
from .model_analysis import SupervisedModelAnalysis
from nyx.modelling.util import track_artifacts

class ClassificationModelAnalysis(SupervisedModelAnalysis):
    def __init__(
        self, model, x_train, x_test, target, model_name,
    ):
        """
        Class to analyze Classification models through metrics, global/local interpretation and visualizations.
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
            x_train.drop(target, axis=1),
            x_test.drop(target, axis=1),
            x_train[target],
            x_test[target],
            model_name,
        )

        self.multiclass = len(np.unique(list(self.y_train) + list(self.y_test))) > 2

        self.classes = [
            str(item) for item in np.unique(list(self.y_train) + list(self.y_test))
        ]

    def accuracy(self, **kwargs):
        """
        It measures how many observations, both positive and negative, were correctly classified.
        
        Returns
        -------
        float
            Accuracy
        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.accuracy()
        """

        return metrics.accuracy_score(self.y_test, self.y_pred, **kwargs)

    def balanced_accuracy(self, **kwargs):
        """
        The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets.
        It is defined as the average of recall obtained on each class.
        The best value is 1 and the worst value is 0 when adjusted=False.
        
        Returns
        -------
        float
            Balanced accuracy
        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.balanced_accuracy()
        """

        return metrics.balanced_accuracy_score(self.y_test, self.y_pred, **kwargs)

    def average_precision(self, **kwargs):
        """
        AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold,
        with the increase in recall from the previous threshold used as the weight
        
        Returns
        -------
        float
            Average Precision Score
        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.average_precision()
        """

        if hasattr(self.model, "decision_function"):
            return metrics.average_precision_score(
                self.y_test, self.model.decision_function(self.x_test), **kwargs
            )
        else:
            return np.nan

    def roc_auc(self, **kwargs):
        """
        This metric tells us that this metric shows how good at ranking predictions your model is.
        It tells you what is the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.
        
        Returns
        -------
        float
            ROC AUC Score
        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.roc_auc()
        """

        multi_class = kwargs.pop("multi_class", "ovr")

        if self.multiclass:
            roc_auc = metrics.roc_auc_score(
                self.y_test, self.probabilities, multi_class=multi_class, **kwargs
            )
        else:
            if hasattr(self.model, "decision_function"):
                roc_auc = metrics.roc_auc_score(
                    self.y_test, self.model.decision_function(self.x_test), **kwargs
                )
            else:
                roc_auc = np.nan

        return roc_auc

    def zero_one_loss(self, **kwargs):
        """
        Return the fraction of misclassifications (float), else it returns the number of misclassifications (int).
        
        The best performance is 0.
        
        Returns
        -------
        float
            Zero one loss
        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.zero_one_loss()
        """

        return metrics.zero_one_loss(self.y_test, self.y_pred, **kwargs)

    def recall(self, **kwargs):
        """
        The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. 
        
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.
        
        Returns
        -------
        float
            Recall
        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.recall()
        """

        avg = kwargs.pop("average", "macro")

        if self.multiclass:
            return metrics.recall_score(self.y_test, self.y_pred, average=avg, **kwargs)
        else:
            return metrics.recall_score(self.y_test, self.y_pred, **kwargs)


    def precision(self, **kwargs):
        """
        The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
        
        The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
        The best value is 1 and the worst value is 0.
        
        Returns
        -------
        float
            Precision
        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.precision()
        """

        avg = kwargs.pop("average", "macro")

        if self.multiclass:
            return metrics.precision_score(
                self.y_test, self.y_pred, average=avg, **kwargs
            )
        else:
            return metrics.precision_score(self.y_test, self.y_pred, **kwargs)

    def matthews_corr_coef(self, **kwargs):
        """
        The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications.
        It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.
        The MCC is in essence a correlation coefficient value between -1 and +1. 
        A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction.
        The statistic is also known as the phi coefficient. 
        
        Returns
        -------
        float
            Matthews Correlation Coefficient
        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.mathews_corr_coef()
        """

        return metrics.matthews_corrcoef(self.y_test, self.y_pred, **kwargs)

    def log_loss(self, **kwargs):
        """
        Log loss, aka logistic loss or cross-entropy loss.
        This is the loss function used in (multinomial) logistic regression and extensions of it
        such as neural networks, defined as the negative log-likelihood of the true labels given a probabilistic classifier’s predictions.
        
        Returns
        -------
        Float
            Log loss
        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.log_loss()
        """

        if self.probabilities is not None:
            return metrics.log_loss(self.y_test, self.probabilities, **kwargs)
        else:
            return np.nan

    def jaccard(self, **kwargs):
        """
        The Jaccard index, or Jaccard similarity coefficient,
        defined as the size of the intersection divided by the size of the union of two label sets,
        is used to compare set of predicted labels for a sample to the corresponding set of labels in y_true.
        
        Returns
        -------
        float
            Jaccard Score
        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.jaccard()
        """

        avg = kwargs.pop("average", "macro")

        if self.multiclass:
            return metrics.jaccard_score(
                self.y_test, self.y_pred, average=avg, **kwargs
            )
        else:
            return metrics.jaccard_score(self.y_test, self.y_pred, **kwargs)

    def hinge_loss(self, **kwargs):
        """
        Computes the average distance between the model and the data using hinge loss, a one-sided metric that considers only prediction errors.
        
        Returns
        -------
        float
            Hinge loss
        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.hinge_loss()
        """

        if hasattr(self.model, "decision_function"):
            return metrics.hinge_loss(
                self.y_test, self.model.decision_function(self.x_test), **kwargs
            )
        else:
            return np.nan

    def hamming_loss(self, **kwargs):
        """
        The Hamming loss is the fraction of labels that are incorrectly predicted.
        
        Returns
        -------
        float
            Hamming loss
        Examples
        --------
        >>> m = model.LogisticRegression()
        >>> m.hamming_loss()
        """

        return metrics.hamming_loss(self.y_test, self.y_pred, **kwargs)