import copy
import os
import re

import ipywidgets as widgets
import numpy as np
import pandas as pd

from nyx.config import shell
from nyx.stats.stats import Stats
from nyx.util import (
    CLEANING_CHECKLIST,
    DATA_CHECKLIST,
    ISSUES_CHECKLIST,
    MULTI_ANALYSIS_CHECKLIST,
    PREPARATION_CHECKLIST,
    UNI_ANALYSIS_CHECKLIST,
    _get_columns,
    _get_attr_,
    _get_item_,
    _interpret_data,
    label_encoder,
)
from nyx.visualizations.visualizations import Visualizations
from IPython import get_ipython
from IPython.display import HTML, display
from ipywidgets import Layout
