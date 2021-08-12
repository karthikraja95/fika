import pandas as pd

from .model import ModelBase
from nyx.config import shell
from nyx.model_analysis.regression_model_analysis import RegressionModelAnalysis
from nyx.analysis import Analysis
from nyx.cleaning.clean import Clean
from nyx.preprocessing.preprocess import Preprocess
from nyx.feature_engineering.feature import Feature
from nyx.visualizations.visualizations import Visualizations
from nyx.stats.stats import Stats
from nyx.modelling.util import add_to_queue

class Regression(
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
        Class to run analysis, transform your data and run Regression algorithms.
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

    @add_to_queue
    def LinearRegression(
        self,
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="lin_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Linear Regression.
        For more Linear Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
        If running gridsearch, the implemented cross validators are:
            - 'kfold' for KFold
            - 'strat-kfold' for StratifiedKfold
        Possible scoring metrics: 
            - ‘explained_variance’ 	 
            - ‘max_error’ 	 
            - ‘neg_mean_absolute_error’ --> MAE	 
            - ‘neg_mean_squared_error’ --> MSE 	 
            - ‘neg_mean_squared_log_error’ --> MSLE
            - ‘neg_median_absolute_error’ --> MeAE 	 
            - ‘r2’
        
        Parameters
        ----------
        cv : bool, optional
            If True run crossvalidation on the model, by default None.
        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None
        score : str, optional
            Scoring metric to evaluate models, by default ‘neg_mean_squared_error’
        model_name : str, optional
            Name for this model, by default "lin_reg"
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1        	
        fit_intercept : boolean, optional, default True
            whether to calculate the intercept for this model.
            If set to False, no intercept will be used in calculations (e.g. data is expected to be already centered).
        normalize : boolean, optional, default False
            This parameter is ignored when fit_intercept is set to False.
            If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
        Returns
        -------
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results
        Examples
        --------
        >>> model.LinearRegression()
        >>> model.LinearRegression(model_name='m1', normalize=True)
        >>> model.LinearRegression(cv=10)
        >>> model.LinearRegression(gridsearch={'normalize':[True, False]}, cv='strat-kfold')
        >>> model.LinearRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import LinearRegression

        model = LinearRegression

        model = self._run_supervised_model(
            model,
            model_name,
            RegressionModelAnalysis,
            cv_type=cv_type,
            gridsearch=gridsearch,
            score=score,
            run=run,
            verbose=verbose,
            **kwargs,
        )

        return model
