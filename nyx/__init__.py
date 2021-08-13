import warnings
import os
import shutil

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)

nyx_home = os.path.join(os.path.expanduser("~"), ".nyx")
config_home = os.path.join(nyx_home, "config.yml")
pkg_directory = os.path.dirname(__file__)

# Create the config file
if not os.path.exists(config_home):
    os.makedirs(nyx_home)
    shutil.copyfile(
        os.path.join(pkg_directory, "config", "config.yml"), os.path.join(config_home)
    )

import pandas as pd
from IPython import get_ipython
import plotly.io as pio

# let init-time option registration happen
import nyx.config.config_init
from nyx.config.config import (
    describe_option,
    get_option,
    options,
    reset_option,
    set_option,
)

from nyx.helpers import groupby_analysis

from nyx.analysis import Analysis
from nyx.modelling import Classification, Regression, Unsupervised
from nyx.model_analysis import (
    ClassificationModelAnalysis,
    RegressionModelAnalysis,
    UnsupervisedModelAnalysis,
    TextModelAnalysis,
)

pd.options.mode.chained_assignment = None
pio.templates.default = "plotly_white"

__all__ = [
    "Analysis",
    "Classification",
    "Regression",
    "Unsupervised",
    "ClassificationModelAnalsysis",
    "RegressionModelAnalysis",
    "UnsupervisedModelAnalysis",
    "TextModelAnalysis",
]

shell = get_ipython().__class__.__name__

if shell == "ZMQInteractiveShell":
    import shap
    from plotly.offline import init_notebook_mode

    init_notebook_mode(connected=True)
    shap.initjs()