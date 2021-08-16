import warnings
import os
import shutil

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)

fika_home = os.path.join(os.path.expanduser("~"), ".fika")
config_home = os.path.join(fika_home, "config.yml")
pkg_directory = os.path.dirname(__file__)

# Create the config file
if not os.path.exists(config_home):
    os.makedirs(fika_home)
    shutil.copyfile(
        os.path.join(pkg_directory, "config", "config.yml"), os.path.join(config_home)
    )

import pandas as pd
from IPython import get_ipython
import plotly.io as pio

# let init-time option registration happen
import fika.config.config_init
from fika.config.config import (
    describe_option,
    get_option,
    options,
    reset_option,
    set_option,
)

from fika.helpers import groupby_analysis

from fika.analysis import Analysis
from fika.modelling import Classification, Regression, Unsupervised
from fika.model_analysis import (
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