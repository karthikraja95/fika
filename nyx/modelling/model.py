import os
import warnings
import pandas as pd
import sklearn
import numpy as np
import copy


from pathlib import Path
from IPython.display import display
from ipywidgets import widgets
from ipywidgets.widgets.widget_layout import Layout

from nyx.config import shell
from nyx.config.config import _global_config
from nyx.model_analysis.unsupervised_model_analysis import UnsupervisedModelAnalysis
from nyx.model_analysis.text_model_analysis import TextModelAnalysis
from nyx.modelling import text
from nyx.modelling.util import (
    _get_cv_type,
    _make_img_project_dir,
    _run_models_parallel,
    add_to_queue,
    run_crossvalidation,
    run_gridsearch,
    to_pickle,
    track_model,
)
from nyx.templates.template_generator import TemplateGenerator as tg
from nyx.util import _input_columns, split_data, _get_attr_, _get_item_

warnings.simplefilter("ignore", FutureWarning)