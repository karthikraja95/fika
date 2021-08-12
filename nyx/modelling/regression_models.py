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

    @add_to_queue
    def BayesianRidgeRegression(
        self,
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="bayridge_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Bayesian Ridge Regression model.
        For more Linear Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge
        and https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression 
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
            Name for this model, by default "bayridge_reg"
        
            Name of column for labels that are generated, by default "bayridge_reg_predictions"
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1
        
        n_iter : int, optional
            Maximum number of iterations. Default is 300. Should be greater than or equal to 1.
        tol : float, optional
            Stop the algorithm if w has converged. Default is 1.e-3.
            
        alpha_1 : float, optional
            Hyper-parameter : shape parameter for the Gamma distribution prior over the alpha parameter. Default is 1.e-6
        alpha_2 : float, optional
            Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter. Default is 1.e-6.
        lambda_1 : float, optional
            Hyper-parameter : shape parameter for the Gamma distribution prior over the lambda parameter. Default is 1.e-6.
        lambda_2 : float, optional
            Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter. Default is 1.e-6
        fit_intercept : boolean, optional, default True
            Whether to calculate the intercept for this model.
            The intercept is not treated as a probabilistic parameter and thus has no associated variance.
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
        >>> model.BayesianRidgeRegression()
        >>> model.BayesianRidgeRegression(model_name='alpha_1', C=0.0003)
        >>> model.BayesianRidgeRegression(cv=10)
        >>> model.BayesianRidgeRegression(gridsearch={'alpha_2':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.BayesianRidgeRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import BayesianRidge

        model = BayesianRidge

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

    @add_to_queue
    def ElasticnetRegression(
        self,
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="elastic",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Elastic Net regression with combined L1 and L2 priors as regularizer.
        
        For more Linear Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet 
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
            Name for this model, by default "elastic"
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1   
        
        alpha : float, optional
            Constant that multiplies the penalty terms.
            Defaults to 1.0. See the notes for the exact mathematical meaning of this parameter.
            ``alpha = 0`` is equivalent to an ordinary least square, solved by the LinearRegression object.
            For numerical reasons, using alpha = 0 with the Lasso object is not advised.
            Given this, you should use the LinearRegression object.
        l1_ratio : float
            The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
            For l1_ratio = 0 the penalty is an L2 penalty.
            For l1_ratio = 1 it is an L1 penalty.
            For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
        fit_intercept : bool
            Whether the intercept should be estimated or not.
            If False, the data is assumed to be already centered.
        normalize : boolean, optional, default False
            This parameter is ignored when fit_intercept is set to False.
            If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
            If you wish to standardize, please use sklearn.preprocessing.
        precompute : True | False | array-like
            Whether to use a precomputed Gram matrix to speed up calculations.
            The Gram matrix can also be passed as argument.
            For sparse input this option is always True to preserve sparsity.
        max_iter : int, optional
            The maximum number of iterations
        tol : float, optional
            The tolerance for the optimization: if the updates are smaller than tol, the optimization code checks the dual gap for optimality and continues until it is smaller than tol.
        
        positive : bool, optional
            When set to True, forces the coefficients to be positive.
        selection : str, default ‘cyclic’
            If set to ‘random’, a random coefficient is updated every iteration rather than looping over features sequentially by default.
            This (setting to ‘random’) often leads to significantly faster convergence especially when tol is higher than 1e-4.
        Returns
        -------
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results
        Examples
        --------
        >>> model.ElasticNetRegression()
        >>> model.ElasticNetRegression(model_name='m1', alpha=0.0003)
        >>> model.ElasticNetRegression(cv=10)
        >>> model.ElasticNetRegression(gridsearch={'alpha':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.ElasticNetRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import ElasticNet

        model = ElasticNet

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

    @add_to_queue
    def LassoRegression(
        self,
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="lasso",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Lasso Regression Model trained with L1 prior as regularizer (aka the Lasso)
        Technically the Lasso model is optimizing the same objective function as the Elastic Net with l1_ratio=1.0 (no L2 penalty).   
        For more Lasso Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso
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
            Name for this model, by default "lasso"
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1        
        
        alpha : float, optional
            Constant that multiplies the L1 term.
            Defaults to 1.0. alpha = 0 is equivalent to an ordinary least square, solved by the LinearRegression object.
            For numerical reasons, using alpha = 0 with the Lasso object is not advised.
            Given this, you should use the LinearRegression object.
        fit_intercept : boolean, optional, default True
            Whether to calculate the intercept for this model.
            If set to False, no intercept will be used in calculations (e.g. data is expected to be already centered).
        normalize : boolean, optional, default False
            This parameter is ignored when fit_intercept is set to False.
            If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
            
        precompute : True | False | array-like, default=False
            Whether to use a precomputed Gram matrix to speed up calculations.
            If set to 'auto' let us decide. The Gram matrix can also be passed as argument.
            For sparse input this option is always True to preserve sparsity.
        max_iter : int, optional
            The maximum number of iterations
        
        tol : float, optional
            The tolerance for the optimization:
             if the updates are smaller than tol, the optimization code checks the dual gap for optimality and continues until it is smaller than tol.
        
        positive : bool, optional
            When set to True, forces the coefficients to be positive.
        selection : str, default ‘cyclic’
            If set to ‘random’, a random coefficient is updated every iteration rather than looping over features sequentially by default.
            This (setting to ‘random’) often leads to significantly faster convergence especially when tol is higher than 1e-4.
        Returns
        -------
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results
        Examples
        --------
        >>> model.LassoRegression()
        >>> model.LassoRegression(model_name='m1', alpha=0.0003)
        >>> model.LassoRegression(cv=10)
        >>> model.LassoRegression(gridsearch={'alpha':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.LassoRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import Lasso

        model = Lasso

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

    @add_to_queue
    def RidgeRegression(
        self,
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="ridge_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Ridge Regression model. 
        For more Ridge Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
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
            Name for this model, by default "ridge"
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1        
        
        alpha : {float, array-like}, shape (n_targets)
            Regularization strength; must be a positive float.
            Regularization improves the conditioning of the problem and reduces the variance of the estimates.
            Larger values specify stronger regularization.
            Alpha corresponds to C^-1 in other linear models such as LogisticRegression or LinearSVC.
            If an array is passed, penalties are assumed to be specific to the targets. Hence they must correspond in number.
        
        fit_intercept : boolean
            Whether to calculate the intercept for this model.
            If set to false, no intercept will be used in calculations (e.g. data is expected to be already centered).
        normalize : boolean, optional, default False
            This parameter is ignored when fit_intercept is set to False.
            If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
        
        max_iter : int, optional
            Maximum number of iterations for conjugate gradient solver.
        tol : float
            Precision of the solution.
        Returns
        -------
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results
        Examples
        --------
        >>> model.RidgeRegression()
        >>> model.RidgeRegression(model_name='m1', alpha=0.0003)
        >>> model.RidgeRegression(cv=10)
        >>> model.RidgeRegression(gridsearch={'alpha':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.RidgeRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import Ridge

        model = Ridge

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



