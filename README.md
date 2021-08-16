# nija



*"A collection of tools for Data Scientists and ML Engineers to automate their workflow of performing analysis to deploying models and pipelines."*

nija is a library/platform that automates your data science and analytical tasks at any stage in the pipeline. nija is, at its core, a uniform API that helps automate analytical techniques from various libaries such as pandas, sci-kit learn, spacy, etc.

## My Vision for nija

nija is a platform (Web Application and API) that automates tasks of the ML pipeline, from Data Cleaning to Validation. nija is designed to aggregate machine learning techniques and models so that they are usable by everyone from Data Scientists and Machine Learning Engineers to Data Analysts and Business Professionals. It gives users full customizability and visibility into the techniques that are being used and also comes with an autoML feature (soon). Each technique has customizable parameters where applicable that can be passed in as arguments to help automate tasks. Every part of the auto-ml pipeline will be automated and users can start automating at any point (i.e. if the user already has cleaned their dataset, they can start automating from the feature engineering/extraction phase). All of this being done with the goal in mind that engineers, scientists, analysts and professionals alike spend less time on coding and worrying about how to do the analysis and instead worry about what analytic tools will best help them get insights from their data.

nija provides you with the code for each technique that was ran on your dataset to remove the "black box" of other autoML platforms. This allows users to learn, customize and tweak the code as they desire. The code provided will be production-ready so you don't have to waste time writing the code and then revising it to production standard. If any models were ran, the users will receive the trained models. As nija goes through the ML pipeline it records its actions and steps provides a detailed report of what was done, how it was done, where it was done, etc. allowing users to share their process with co-workers, colleagues, friends, etc.

It is Py-automls's goal that Data Scientists and Machine Learning Engineers will contribute the techniques they have used and that researchers will contribute with their code and paper so that everyone using the platform can apply the latest advancements and techniques in A.I. onto their dataset.

## Setup

**Python Requirements**: 3.6, 3.7

#### Install from GitHub

`pip install git+https://github.com/karthikraja95/nija.git`


## How to use nija

Take a look at this [nija.ipynb](https://github.com/karthikraja95/nija/blob/master/examples/data/nija.ipynb) notebook