import os 
import yaml
from Ipython import get_ipython
from nyx.util import _make_dir

pkg_directory = os.path.dirname(__file__)

with open(
    os.path.join(os.path.expanduser("~"), ".nyx", "config.yml"), "r"
) as ymlfile:
    cfg = yaml.safe_load(ymlfile)

shell = get_ipython().__class__.__name__