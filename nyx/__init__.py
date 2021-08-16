import warnings
import os
import shutil

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)

nija_home = os.path.join(os.path.expanduser("~"), ".nija")
config_home = os.path.join(nija_home, "config.yml")
pkg_directory = os.path.dirname(__file__)

# Create the config file
if not os.path.exists(config_home):
    os.makedirs(nija_home)
    shutil.copyfile(
        os.path.join(pkg_directory, "config", "config.yml"), os.path.join(config_home)
    )

import pandas as pd
from IPython import get_ipython
import plotly.io as pio

# let init-time option registration happen
import nija.config.config_init
from nija.config.config import (
    describe_option,
    get_option,
    options,
    reset_option,
    set_option,
)

from nija.helpers import groupby_analysis

from nija.analysis import Analysis
from nija.modelling import Classification, Regression, Unsupervised
from nija.model_analysis import (
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