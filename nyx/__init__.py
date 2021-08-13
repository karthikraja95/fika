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