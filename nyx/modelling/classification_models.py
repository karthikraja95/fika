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

    @add_to_queue
    def SGDClassification(
        self,
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="sgd_cls",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Linear classifier (SVM, logistic regression, a.o.) with SGD training.
        For more info please view it here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier
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
            Name for this model, by default "sgd_cls"
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1
        
        loss : str, default: ‘hinge’
            The loss function to be used. Defaults to ‘hinge’, which gives a linear SVM.
            The possible options are ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, or a regression loss: ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.
            The ‘log’ loss gives logistic regression, a probabilistic classifier. 
            ‘modified_huber’ is another smooth loss that brings tolerance to outliers as well as probability estimates. 
            ‘squared_hinge’ is like hinge but is quadratically penalized. 
            ‘perceptron’ is the linear loss used by the perceptron algorithm.
            The other losses are designed for regression but can be useful in classification as well; see SGDRegressor for a description.
        penalty : str, ‘none’, ‘l2’, ‘l1’, or ‘elasticnet’
            The penalty (aka regularization term) to be used.
            Defaults to ‘l2’ which is the standard regularizer for linear SVM models.
            ‘l1’ and ‘elasticnet’ might bring sparsity to the model (feature selection) not achievable with ‘l2’.
        
        alpha : float
            Constant that multiplies the regularization term. Defaults to 0.0001 Also used to compute learning_rate when set to ‘optimal’.
        l1_ratio : float
            The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Defaults to 0.15.
        fit_intercept : bool
            Whether the intercept should be estimated or not. If False, the data is assumed to be already centered. Defaults to True.
        max_iter : int, optional (default=1000)
            The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit.
        tol : float or None, optional (default=1e-3)
            The stopping criterion. If it is not None, the iterations will stop when (loss > best_loss - tol) for n_iter_no_change consecutive epochs.
        shuffle : bool, optional
            Whether or not the training data should be shuffled after each epoch. Defaults to True.
        epsilon : float
            Epsilon in the epsilon-insensitive loss functions; only if loss is ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’. For ‘huber’, determines the threshold at which it becomes less important to get the prediction exactly right. For epsilon-insensitive, any differences between the current prediction and the correct label are ignored if they are less than this threshold.
        learning_rate : string, optional
            The learning rate schedule:
            ‘constant’:
                eta = eta0
            ‘optimal’: [default]
                eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.
            ‘invscaling’:
                eta = eta0 / pow(t, power_t)
            ‘adaptive’:
                eta = eta0, as long as the training keeps decreasing. Each time n_iter_no_change consecutive epochs fail to decrease the training loss by tol or fail to increase validation score by tol if early_stopping is True, the current learning rate is divided by 5.
        eta0 : double
            The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules. The default value is 0.0 as eta0 is not used by the default schedule ‘optimal’.
        power_t : double
            The exponent for inverse scaling learning rate [default 0.5].
        early_stopping : bool, default=False
            Whether to use early stopping to terminate training when validation score is not improving.
            If set to True, it will automatically set aside a stratified fraction of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.
        validation_fraction : float, default=0.1
            The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True.
        n_iter_no_change : int, default=5
            Number of iterations with no improvement to wait before early stopping.
        class_weight : dict, {class_label: weight} or “balanced” or None, optional
            Preset for the class_weight fit parameter.
            Weights associated with classes. If not given, all classes are supposed to have weight one.
            The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
        average : bool or int, optional
            When set to True, computes the averaged SGD weights and stores the result in the coef_ attribute.
            If set to an int greater than 1, averaging will begin once the total number of samples seen reaches average. So average=10 will begin averaging after seeing 10 samples.
        Returns
        -------
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results
        
        Examples
        --------
        >>> model.SGDClassification()
        >>> model.SGDClassification(model_name='rc_1, tol=0.001)
        >>> model.SGDClassification(cv_type='kfold')
        >>> model.SGDClassification(gridsearch={'alpha':[0.01, 0.02]}, cv_type='strat-kfold')
        >>> model.SGDClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.linear_model import SGDClassifier

        model = SGDClassifier

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

    @add_to_queue
    def ADABoostClassification(
        self,
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="ada_cls",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains an AdaBoost classification model.
        An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset
        but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.
        For more AdaBoost info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
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
            Name for this model, by default "ada_cls"
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1
        base_estimator : object, optional (default=None)
            The base estimator from which the boosted ensemble is built.
            Support for sample weighting is required, as well as proper classes_ and n_classes_ attributes.
            If None, then the base estimator is DecisionTreeClassifier(max_depth=1)
        n_estimators : integer, optional (default=50)
            The maximum number of estimators at which boosting is terminated.
            In case of perfect fit, the learning procedure is stopped early.
        learning_rate : float, optional (default=1.)
            Learning rate shrinks the contribution of each classifier by learning_rate.
            There is a trade-off between learning_rate and n_estimators.
        Returns
        -------
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results
        Examples
        --------
        >>> model.AdaBoostClassification()
        >>> model.AdaBoostClassification(model_name='rc_1, learning_rate=0.001)
        >>> model.AdaBoostClassification(cv_type='kfold')
        >>> model.AdaBoostClassification(gridsearch={'n_estimators': [50, 100]}, cv_type='strat-kfold')
        >>> model.AdaBoostClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import AdaBoostClassifier

        model = AdaBoostClassifier

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

    @add_to_queue
    def BaggingClassification(
        self,
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="bag_cls",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Bagging classification model.
        A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction.
        Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.
        For more Bagging Classifier info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
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
            Name for this model, by default "bag_cls"
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
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results
        Examples
        --------
        >>> model.BaggingClassification()
        >>> model.BaggingClassification(model_name='m1', n_estimators=100)
        >>> model.BaggingClassification(cv_type='kfold')
        >>> model.BaggingClassification(gridsearch={'n_estimators':[100, 200]}, cv_type='strat-kfold')
        >>> model.BaggingClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import BaggingClassifier

        model = BaggingClassifier

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

    @add_to_queue
    def GradientBoostingClassification(
        self,
        cv_type=None,
        gridsearch=None,
        score="accuracy",
        model_name="grad_cls",
        run=True,
        verbose=1,
        **kwargs,
    ):
        # region
        """
        Trains a Gradient Boosting classification model.
        GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions.
        In each stage n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. 
        Binary classification is a special case where only a single regression tree is induced.
        For more Gradient Boosting Classifier info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier   
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
        cv_type : bool, optional
            If True run crossvalidation on the model, by default None.
        gridsearch : int, Crossvalidation Generator, optional
            Cross validation method, by default None
        score : str, optional
            Scoring metric to evaluate models, by default 'accuracy'
        model_name : str, optional
            Name for this model, by default "grad_cls"
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1
        loss : {‘deviance’, ‘exponential’}, optional (default=’deviance’)
            loss function to be optimized. ‘deviance’ refers to deviance (= logistic regression) for classification with probabilistic outputs. 
            For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm.
            
        learning_rate : float, optional (default=0.1)
            learning rate shrinks the contribution of each tree by learning_rate.
            There is a trade-off between learning_rate and n_estimators.
        n_estimators : int (default=100)
            The number of boosting stages to perform.
            Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
        subsample : float, optional (default=1.0)
            The fraction of samples to be used for fitting the individual base learners.
            If smaller than 1.0 this results in Stochastic Gradient Boosting.
            Subsample interacts with the parameter n_estimators.
            Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
        criterion : string, optional (default=”friedman_mse”)
            The function to measure the quality of a split.
            Supported criteria are “friedman_mse” for the mean squared error with improvement score by Friedman, “mse” for mean squared error, and “mae” for the mean absolute error.
            The default value of “friedman_mse” is generally the best as it can provide a better approximation in some cases.
        min_samples_split : int, float, optional (default=2)
            The minimum number of samples required to split an internal node:
                If int, then consider min_samples_split as the minimum number.
                If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
        min_samples_leaf : int, float, optional (default=1)
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches.
            This may have the effect of smoothing the model, especially in regression.
                If int, then consider min_samples_leaf as the minimum number.
                If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
        max_depth : integer, optional (default=3)
            maximum depth of the individual regression estimators.
            The maximum depth limits the number of nodes in the tree.
            Tune this parameter for best performance; the best value depends on the interaction of the input variables.
        max_features : int, float, string or None, optional (default=None)
            The number of features to consider when looking for the best split:
                If int, then consider max_features features at each split.
                If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
                If “auto”, then max_features=sqrt(n_features).
                If “sqrt”, then max_features=sqrt(n_features).
                If “log2”, then max_features=log2(n_features).
                If None, then max_features=n_features.
            Choosing max_features < n_features leads to a reduction of variance and an increase in bias.
            Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features. 
        max_leaf_nodes : int or None, optional (default=None)
            Grow trees with max_leaf_nodes in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.
        presort : bool or ‘auto’, optional (default=’auto’)
            Whether to presort the data to speed up the finding of best splits in fitting.
            Auto mode by default will use presorting on dense data and default to normal sorting on sparse data.
            Setting presort to true on sparse data will raise an error.
        validation_fraction : float, optional, default 0.1
            The proportion of training data to set aside as validation set for early stopping.
            Must be between 0 and 1. Only used if n_iter_no_change is set to an integer.
        tol : float, optional, default 1e-4
            Tolerance for the early stopping.
            When the loss is not improving by at least tol for n_iter_no_change iterations (if set to a number), the training stops.
        Returns
        -------
        ClassificationModelAnalysis
            ClassificationModelAnalysis object to view results and analyze results
        Examples
        --------
        >>> model.GradientBoostingClassification()
        >>> model.GradientBoostingClassification(model_name='m1', n_estimators=100)
        >>> model.GradientBoostingClassification(cv_type='kfold')
        >>> model.GradientBoostingClassification(gridsearch={'n_estimators':[100, 200]}, cv_type='strat-kfold')
        >>> model.GradientBoostingClassification(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier

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