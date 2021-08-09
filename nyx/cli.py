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

@main.command()
def install_corpora():
    """
    Installs the necessary corpora from spaCy and NLTK for NLP analysis.
    """

    py_exe = ""

    try:
        if "3." in subprocess.check_output(
            "python --version", stderr=subprocess.STDOUT, shell=True
        ).decode("utf-8"):
            py_exe = "python"
    except subprocess.CalledProcessError as e:
        pass

    try:
        subprocess.check_output(
            "python3 --version", stderr=subprocess.STDOUT, shell=True
        ).decode("utf-8")
        py_exe = "python3"
    except subprocess.CalledProcessError as e:
        pass

    if not py_exe:
        raise EnvironmentError("Python is not in your path, please add it your path.")

    os.system(f"{py_exe} -m textblob.download_corpora")
    os.system(f"{py_exe} -c 'import nltk; nltk.download(\"stopwords\")'")
    os.system(f"{py_exe} -m spacy download en")
