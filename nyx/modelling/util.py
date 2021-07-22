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