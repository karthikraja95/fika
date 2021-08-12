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

    @add_to_queue
    def SGDRegression(
        self,
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="sgd_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a SGD Regression model. 
        Linear model fitted by minimizing a regularized empirical loss with SGD
        SGD stands for Stochastic Gradient Descent: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate).
        The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net).
        If the parameter update crosses the 0.0 value because of the regularizer, the update is truncated to 0.0 to allow for learning sparse models and achieve online feature selection.
        For more SGD Regression info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor
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
            Name for this model, by default "sgd_reg"
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1        
        
        loss : str, default: ‘squared_loss’
            The loss function to be used.
            
            The possible values are ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’
            The ‘squared_loss’ refers to the ordinary least squares fit.
            ‘huber’ modifies ‘squared_loss’ to focus less on getting outliers correct by switching from squared to linear loss past a distance of epsilon.
            ‘epsilon_insensitive’ ignores errors less than epsilon and is linear past that; this is the loss function used in SVR.
            ‘squared_epsilon_insensitive’ is the same but becomes squared loss past a tolerance of epsilon.
        penalty : str, ‘none’, ‘l2’, ‘l1’, or ‘elasticnet’
            The penalty (aka regularization term) to be used.
            Defaults to ‘l2’ which is the standard regularizer for linear SVM models.
            ‘l1’ and ‘elasticnet’ might bring sparsity to the model (feature selection) not achievable with ‘l2’.
        alpha : float
            Constant that multiplies the regularization term.
            Defaults to 0.0001 Also used to compute learning_rate when set to ‘optimal’.
        l1_ratio : float
            The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
            l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
            Defaults to 0.15.
        fit_intercept : bool
            Whether the intercept should be estimated or not.
            If False, the data is assumed to be already centered.
            Defaults to True.
        max_iter : int, optional (default=1000)
            The maximum number of passes over the training data (aka epochs).
            It only impacts the behavior in the fit method, and not the partial_fit.
        tol : float or None, optional (default=1e-3)
            The stopping criterion. 
            If it is not None, the iterations will stop when (loss > best_loss - tol) for n_iter_no_change consecutive epochs.
        shuffle : bool, optional
            Whether or not the training data should be shuffled after each epoch. Defaults to True.
        epsilon : float
            Epsilon in the epsilon-insensitive loss functions; only if loss is ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.
            
            For ‘huber’, determines the threshold at which it becomes less important to get the prediction exactly right.
            For epsilon-insensitive, any differences between the current prediction and the correct label are ignored if they are less than this threshold.
        
        learning_rate : string, optional
            The learning rate schedule:
                ‘constant’:
                    eta = eta0
                ‘optimal’:
                    eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.
                ‘invscaling’: [default]
                    eta = eta0 / pow(t, power_t)
                ‘adaptive’:
                    eta = eta0, as long as the training keeps decreasing.
                    Each time n_iter_no_change consecutive epochs fail to decrease the training loss by tol or fail to increase validation score by tol if early_stopping is True,
                    the current learning rate is divided by 5.
        eta0 : double
            The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules.
            The default value is 0.01.
        power_t : double
            The exponent for inverse scaling learning rate [default 0.5].
        early_stopping : bool, default=False
            Whether to use early stopping to terminate training when validation score is not improving.
            If set to True, it will automatically set aside a fraction of training data as validation 
            and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.
        validation_fraction : float, default=0.1
            The proportion of training data to set aside as validation set for early stopping.
            Must be between 0 and 1. Only used if early_stopping is True.
        n_iter_no_change : int, default=5
            Number of iterations with no improvement to wait before early stopping.
        average : bool or int, optional
            When set to True, computes the averaged SGD weights and stores the result in the coef_ attribute.
            If set to an int greater than 1, averaging will begin once the total number of samples seen reaches average.
            So average=10 will begin averaging after seeing 10 samples.
        Returns
        -------
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results
        Examples
        --------
        >>> model.SGDRegression()
        >>> model.SGDRegression(model_name='m1', alpha=0.0003)
        >>> model.SGDRegression(cv=10)
        >>> model.SGDRegression(gridsearch={'alpha':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.SGDRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import SGDRegressor

        model = SGDRegressor

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
    def ADABoostRegression(
        self,
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="ada_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains an AdaBoost Regression model.
        An AdaBoost classifier is a meta-estimator that begins by fitting a regressor on the original dataset and then fits additional copies of the regressor on the same dataset
        but where the weights of incorrectly classified instances are adjusted such that subsequent regressors focus more on difficult cases.
        For more AdaBoost info, you can view it here:https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor
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
        cv : int, Crossvalidation Generator, optional
            Cross validation method, by default None
        gridsearch : dict, optional
            Parameters to gridsearch, by default None
        score : str, optional
            Scoring metric to evaluate models, by default ‘neg_mean_squared_error’
        model_name : str, optional
            Name for this model, by default "ada_reg"
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1
        base_estimator : object, optional (default=None)
            The base estimator from which the boosted ensemble is built.
            Support for sample weighting is required, as well as proper classes_ and n_classes_ attributes.
            If None, then the base estimator is DecisionTreeRegressor(max_depth=3)
        n_estimators : integer, optional (default=50)
            The maximum number of estimators at which boosting is terminated.
            In case of perfect fit, the learning procedure is stopped early.
        learning_rate : float, optional (default=1.)
            Learning rate shrinks the contribution of each classifier by learning_rate.
            There is a trade-off between learning_rate and n_estimators.
        loss : {‘linear’, ‘square’, ‘exponential’}, optional (default=’linear’)
            The loss function to use when updating the weights after each boosting iteration.
        Returns
        -------
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results
        Examples
        --------
        >>> model.AdaBoostRegression()
        >>> model.AdaBoostRegression(model_name='m1', learning_rate=0.0003)
        >>> model.AdaBoostRegression(cv=10)
        >>> model.AdaBoostRegression(gridsearch={'learning_rate':[0.01, 0.02]}, cv='strat-kfold')
        >>> model.AdaBoostRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import AdaBoostRegressor

        model = AdaBoostRegressor

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
    def BaggingRegression(
        self,
        cv_type=None,
        gridsearch=None,
        score="neg_mean_squared_error",
        model_name="bag_reg",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Bagging Regressor model.
        A Bagging classifier is an ensemble meta-estimator that fits base regressors each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction.
        Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.
        For more Bagging Classifier info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor
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
        cv : int, Crossvalidation Generator, optional
            Cross validation method, by default None
        gridsearch : dict, optional
            Parameters to gridsearch, by default None
        score : str, optional
            Scoring metric to evaluate models, by default ‘neg_mean_squared_error’
        model_name : str, optional
            Name for this model, by default "bag_reg"
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1
        base_estimator : object or None, optional (default=None)
            The base estimator to fit on random subsets of the dataset.
            If None, then the base estimator is a decision tree.
        n_estimators : int, optional (default=10)
            The number of base estimators in the ensemble.
        max_samples : int or float, optional (default=1.0)
            The number of samples to draw from X to train each base estimator.
                If int, then draw max_samples samples.
                If float, then draw max_samples * X.shape[0] samples.
        max_features : int or float, optional (default=1.0)
            The number of features to draw from X to train each base estimator.
                If int, then draw max_features features.
                If float, then draw max_features * X.shape[1] features.
        bootstrap : boolean, optional (default=True)
            Whether samples are drawn with replacement. If False, sampling without replacement is performed.
        bootstrap_features : boolean, optional (default=False)
            Whether features are drawn with replacement.
        oob_score : bool, optional (default=False)
            Whether to use out-of-bag samples to estimate the generalization error.
        Returns
        -------
        RegressionModelAnalysis
            RegressionModelAnalysis object to view results and analyze results
        Examples
        --------
        >>> model.BaggingRegression()
        >>> model.BaggingRegression(model_name='m1', n_estimators=100)
        >>> model.BaggingRegression(cv=10)
        >>> model.BaggingRegression(gridsearch={'n_estimators':[100, 200]}, cv='strat-kfold')
        >>> model.BaggingRegression(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import BaggingRegressor

        model = BaggingRegressor

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


