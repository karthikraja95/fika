import inspect
import multiprocessing as mp
import os
import pickle
import warnings
from functools import partial, wraps
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt


with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)

    import mlflow
    import mlflow.lightgbm
    import mlflow.sklearn
    import mlflow.xgboost

import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from yellowbrick.model_selection import CVScores, LearningCurve

from nyx.config import EXP_DIR, DEFAULT_MODEL_DIR, IMAGE_DIR, cfg
from nyx.config.config import _global_config
from nyx.util import _make_dir
from pickle import dump

def add_to_queue(model_function):
    @wraps(model_function)
    def wrapper(self, *args, **kwargs):
        default_kwargs = get_default_args(model_function)

        kwargs["run"] = kwargs.get("run", default_kwargs["run"])
        kwargs["model_name"] = kwargs.get("model_name", default_kwargs["model_name"])
        cv = kwargs.get("cv", False)

        if not _validate_model_name(self, kwargs["model_name"]):
            raise AttributeError("Invalid model name. Please choose another one.")

        if kwargs["run"]:
            return model_function(self, *args, **kwargs)
        else:
            if "XGBoost" in model_function.__name__:
                print(
                    "XGBoost is not comptabile to be run in parallel. Please run it normally or in run all models in a series."
                )
            else:
                kwargs["run"] = True
                self._queued_models[kwargs["model_name"]] = partial(
                    getattr(self, model_function.__name__), *args, **kwargs
                )

    return wrapper

def get_default_args(func):
    """
    Gets default arguments from a function.
    """

    signature = inspect.signature(func)

    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def run_gridsearch(model, gridsearch, cv=5, scoring="accuracy", **gridsearch_kwargs):
    """
    Runs Gridsearch on a model
    
    Parameters
    ----------
    model : Model
        Model to run gridsearch on
    gridsearch : dict
        Dict of params to test
    cv : int, Crossvalidation Generator, optional
        Cross validation method, by default 12
    
    scoring : str
        Scoring metric to use when evaluating models
    
    Returns
    -------
    Model
        Initialized Gridsearch model
    """

    if isinstance(gridsearch, dict):
        gridsearch_grid = gridsearch
        print(f"Gridsearching with the following parameters: {gridsearch_grid}")
    else:
        raise ValueError("Invalid Gridsearch input.")

    model = GridSearchCV(
        model, gridsearch_grid, cv=cv, scoring=scoring, **gridsearch_kwargs
    )

    return model
