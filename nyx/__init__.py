import warnings
import os
import shutil

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)

nyx_home = os.path.join(os.path.expanduser("~"), ".nyx")
config_home = os.path.join(nyx_home, "config.yml")
pkg_directory = os.path.dirname(__file__)

