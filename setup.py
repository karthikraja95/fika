import os
import sys
from subprocess import check_call

from setuptools import find_packages, setup
from setuptools.command.install import install

VERSION = "1.0.0"

pkgs = [
    "numpy==1.18.2",
    "pandas==1.0.3",
    "scikit-learn==0.24",
    "textblob==0.15.3",
    "matplotlib==3.2.1",
    "pandas_summary",
    "ptitprince==0.2.5",
    "nltk==3.5",
    "ipython==7.13.0",
    "gensim==3.8.2",
    "pandas-profiling==2.8.0",
    "cookiecutter==1.6.0",
    "shap==0.35.0",
    "interpret==0.1.21",
    "yellowbrick==1.3",
    "spacy==2.2.4",
    "xgboost==0.90",
    "ipywidgets==7.5.1",
    "qgrid==1.3.1",
    "python-dateutil<2.8.1",
    "itables==0.2.1",
    "selenium==3.141.0",
    "graphviz==0.13.2",
    "pyldavis==2.1.2",
    "swifter==1.0.9",
    "lightgbm==2.3.1",
    "mlflow==1.7.2",
    "statsmodels==0.11.1",
    "ppscore==0.0.2",
    "autoviz==0.0.68",
]

extras = {"ptmodels": ["transformers==2.3.0", "tensorflow==2.1.0"]}


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="nyx",
    url="https://github.com/karthikraja95/NyX",
    packages=find_packages(),
    author="Karthik Bhaskar",
    author_email="karthikrajabk@gmail.com",
    install_requires=pkgs,
    extras_require=extras,
    version=VERSION,
    license="GPL-3.0",
    description="Automate Data Science Workflow from Exploratory Data Analysis to Model Deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="datascience, machinelearning, automation, analysis, automl, python, eda",
    python_requires=">= 3.6",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 4 - Beta",
    ],

    entry_points={"console_scripts": ["nyx=nyx.__main__:main"]},
)