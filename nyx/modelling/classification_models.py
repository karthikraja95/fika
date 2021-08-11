import pandas as pd

from .model import ModelBase
from nyx.config import shell
from nyx.model_analysis.classification_model_analysis import (
    ClassificationModelAnalysis,
)
from nyx.analysis import Analysis
from nyx.cleaning.clean import Clean
from nyx.preprocessing.preprocess import Preprocess
from nyx.feature_engineering.feature import Feature
from nyx.visualizations.visualizations import Visualizations
from nyx.stats.stats import Stats
from nyx.modelling.util import add_to_queue

class Classification(
    ModelBase, Analysis, Clean, Preprocess, Feature, Visualizations, Stats
):
    def __init__(
        self,
        x_train,
        target,
        x_test=None,
        test_split_percentage=0.2,
        exp_name="my-experiment",
    ):
        """
        Class to run analysis, transform your data and run Classification algorithms.
        Parameters
        -----------
        x_train: pd.DataFrame
            Training data or aethos data object
        target: str
            For supervised learning problems, the name of the column you're trying to predict.
        x_test: pd.DataFrame
            Test data, by default None
        test_split_percentage: float
            Percentage of data to split train data into a train and test set, by default 0.2.
        exp_name : str
            Experiment name to be tracked in MLFlow.
        """

        super().__init__(
            x_train,
            target,
            x_test=x_test,
            test_split_percentage=test_split_percentage,
            exp_name=exp_name,
        )


    # NOTE: This entire process may need to be reworked.
    @add_to_queue
    def LogisticRegression(
        self,
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="log_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a logistic regression model.
        For more Logistic Regression info, you can view them here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        If running grid search, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold
        Possible scoring metrics: 
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’
        
        Parameters
        ----------
        cv_type : {kfold, strat-kfold}, Crossvalidation Generator, optional
            Cross validation method, by default None
        gridsearch : dict, optional
            Parameters to gridsearch, by default None
        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'
        model_name : str, optional
            Name for this model, by default "log_reg"
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1
        penalty : str, ‘l1’, ‘l2’, ‘elasticnet’ or ‘none’, optional (default=’l2’)
            Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties. 
            ‘elasticnet’ is only supported by the ‘saga’ solver. If ‘none’ (not supported by the liblinear solver), no regularization is applied.
        tol : float, optional (default=1e-4)
            Tolerance for stopping criteria.
        C : float, optional (default=1.0)
            Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
        class_weight : dict or ‘balanced’, optional (default=None)
            Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
            The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
            Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
        
        Returns
        -------
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results
        Examples
        --------
        >>> model.LogisticRegression()
        >>> model.LogisticRegression(model_name='lg_1', C=0.001)
        >>> model.LogisticRegression(cv_type='kfold')
        >>> model.LogisticRegression(gridsearch={'C':[0.01, 0.02]}, cv_type='strat-kfold')
        >>> model.LogisticRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import LogisticRegression

        solver = kwargs.pop("solver", "lbfgs")

        model = LogisticRegression

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            solver=solver,
            **kwargs,
        )

        return model

    @add_to_queue
    def RidgeClassification(
        self,
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="ridge_cls",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Ridge Classification model.
        For more Ridge Regression parameters, you can view them here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier        
        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold
        Possible scoring metrics: 
            - ‘accuracy’ 	
            - ‘balanced_accuracy’ 	
            - ‘average_precision’ 	
            - ‘brier_score_loss’ 	
            - ‘f1’ 	
            - ‘f1_micro’ 	
            - ‘f1_macro’ 	
            - ‘f1_weighted’ 	
            - ‘f1_samples’ 	
            - ‘neg_log_loss’ 	
            - ‘precision’	
            - ‘recall’ 	
            - ‘jaccard’ 	
            - ‘roc_auc’
        
        Parameters
        ----------
        cv_type : {kfold, strat-kfold}, Crossvalidation Generator, optional
            Cross validation method, by default None
        gridsearch : dict, optional
            Parameters to gridsearch, by default None
        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'
        model_name : str, optional
            Name for this model, by default "ridge_cls"
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1
        alpha : float
            Regularization strength; must be a positive float.
            Regularization improves the conditioning of the problem and reduces the variance of the estimates.
            Larger values specify stronger regularization.
            Alpha corresponds to C^-1 in other linear models such as LogisticRegression or LinearSVC.
        fit_intercept : boolean
            Whether to calculate the intercept for this model.
            If set to false, no intercept will be used in calculations (e.g. data is expected to be already centered).
        normalize : boolean, optional, default False
            This parameter is ignored when fit_intercept is set to False.
            If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
        tol : float, optional (default=1e-4)
            Tolerance for stopping criteria.
        class_weight : dict or ‘balanced’, optional (default=None)
            Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
            The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
            Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
        Returns
        -------
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results
        Examples
        --------
        >>> model.RidgeClassification()
        >>> model.RidgeClassification(model_name='rc_1, tol=0.001)
        >>> model.RidgeClassification(cv_type='kfold')
        >>> model.RidgeClassification(gridsearch={'alpha':[0.01, 0.02]}, cv_type='strat-kfold')
        >>> model.RidgeClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import RidgeClassifier

        model = RidgeClassifier

        model = self._run_supervised_model(
            model,
            model_name,
            ClassificationModelAnalysis,
            cv_type=cv_type,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model