import os

import click
import subprocess
from nyx.config.user_config import EXP_DIR

@click.group()
def main():
    pass


@main.command()
def create():
    """
    Creates a Data Science folder structure and files.
    """
    os.system("cookiecutter https://github.com/drivendata/cookiecutter-data-science")


@main.command()
def enable_extensions():
    """
    Enables jupyter extensions such as qgrid.
    """
    os.system("jupyter nbextension enable --py --sys-prefix widgetsnbextension")
    os.system("jupyter nbextension enable --py --sys-prefix qgrid")