import os
import warnings
import pandas as pd
import sklearn
import numpy as np
import copy


from pathlib import Path
from IPython.display import display
from ipywidgets import widgets
from ipywidgets.widgets.widget_layout import Layout

from nyx.config import shell
from nyx.config.config import _global_config
from nyx.model_analysis.unsupervised_model_analysis import UnsupervisedModelAnalysis
from nyx.model_analysis.text_model_analysis import TextModelAnalysis
from nyx.modelling import text
from nyx.modelling.util import (
    _get_cv_type,
    _make_img_project_dir,
    _run_models_parallel,
    add_to_queue,
    run_crossvalidation,
    run_gridsearch,
    to_pickle,
    track_model,
)
from nyx.templates.template_generator import TemplateGenerator as tg
from nyx.util import _input_columns, split_data, _get_attr_, _get_item_

warnings.simplefilter("ignore", FutureWarning)

class ModelBase(object):
    def __init__(
        self,
        x_train,
        target,
        x_test=None,
        test_split_percentage=0.2,
        exp_name="my-experiment",
    ):

        self._models = {}
        self._queued_models = {}
        self.exp_name = exp_name

        problem = "c" if type(self).__name__ == "Classification" else "r"

        self.x_train = x_train
        self.x_test = x_test
        self.target = target
        self.test_split_percentage = test_split_percentage
        self.target_mapping = None

        if self.x_test is None and not type(self).__name__ == "Unsupervised":
            # Generate train set and test set.
            self.x_train, self.x_test = split_data(
                self.x_train, test_split_percentage, self.target, problem
            )
            self.x_train = self.x_train.reset_index(drop=True)
            self.x_test = self.x_test.reset_index(drop=True)

    def __getitem__(self, key):

        return _get_item_(self, key)

    def __getattr__(self, key):

        # For when doing multi processing when pickle is reconstructing the object
        if key in {"__getstate__", "__setstate__"}:
            return object.__getattr__(self, key)

        if key in self._models:
            return self._models[key]

        return _get_attr_(self, key)

    def __setattr__(self, key, value):

        if key not in self.__dict__ or hasattr(self, key):
            # any normal attributes are handled normally
            dict.__setattr__(self, key, value)
        else:
            self.__setitem__(key, value)

    def __setitem__(self, key, value):

        if key in self.__dict__:
            dict.__setitem__(self.__dict__, key, value)

    def __repr__(self):

        return self.x_train.head().to_string()

    def _repr_html_(self):  # pragma: no cover

        if self.target:
            cols = self.features + [self.target]
        else:
            cols = self.features

        return self.x_train[cols].head()._repr_html_()


    def __deepcopy__(self, memo):

        x_test = self.x_test.copy() if self.x_test is not None else None

        new_inst = type(self)(
            x_train=self.x_train.copy(),
            target=self.target,
            x_test=x_test,
            test_split_percentage=self.test_split_percentage,
            exp_name=self.exp_name,
        )

        new_inst.target_mapping = self.target_mapping
        new_inst._models = self._models
        new_inst._queued_models = self._queued_models

        return new_inst

    @property
    def features(self):
        """Features for modelling"""

        cols = self.x_train.columns.tolist()

        if self.target:
            cols.remove(self.target)

        return cols

    @property
    def train_data(self):
        """Training data used for modelling"""

        return self.x_train[self.features]

    @train_data.setter
    def train_data(self, val):
        """Setting for train_data"""

        val[self.target] = self.y_train
        self.x_train = val

    @property
    def test_data(self):
        """Testing data used to evaluate models"""

        return self.x_test[self.features] if self.x_test is not None else None

    @test_data.setter
    def test_data(self, val):
        """Test data setter"""

        val[self.target] = self.y_test
        self.x_test = val

    @property
    def y_test(self):
        """
        Property function for the testing predictor variable
        """

        if self.x_test is not None:
            if self.target:
                return self.x_test[self.target]
            else:
                return None
        else:
            return None

    @y_test.setter
    def y_test(self, value):
        """
        Setter function for the testing predictor variable
        """

        if self.x_test is not None:
            if self.target:
                self.x_test[self.target] = value
            else:
                self.target = "label"
                self.x_test["label"] = value
                print('Added a target (predictor) field (column) named "label".')



    @property
    def columns(self):
        """
        Property to return columns in the dataset.
        """

        return self.x_train.columns.tolist()

    def copy(self):
        """
        Returns deep copy of object.
        
        Returns
        -------
        Object
            Deep copy of object
        """

        return copy.deepcopy(self)

    def help_debug(self):
        """
        Displays a tips for helping debugging model outputs and how to deal with over and underfitting.
        Credit: Andrew Ng's and his book Machine Learning Yearning
        Examples
        --------
        >>> model.help_debug()
        """

        from nyx.model_analysis.constants import DEBUG_OVERFIT, DEBUG_UNDERFIT

        overfit_labels = [
            widgets.Checkbox(description=item, layout=Layout(width="100%"))
            for item in DEBUG_OVERFIT
        ]
        underfit_labels = [
            widgets.Checkbox(description=item, layout=Layout(width="100%"))
            for item in DEBUG_UNDERFIT
        ]

        overfit_box = widgets.VBox(overfit_labels)
        underfit_box = widgets.VBox(underfit_labels)

        tab_list = [overfit_box, underfit_box]

        tab = widgets.Tab()
        tab.children = tab_list
        tab.set_title(0, "Overfit")
        tab.set_title(1, "Underfit")

        display(tab)

    def run_models(self, method="parallel"):
        """
        Runs all queued models.
        The models can either be run one after the other ('series') or at the same time in parallel.
        Parameters
        ----------
        method : str, optional
            How to run models, can either be in 'series' or in 'parallel', by default 'parallel'
        Examples
        --------
        >>> model.run_models()
        >>> model.run_models(method='series')
        """

        models = []

        if method == "parallel":
            models = _run_models_parallel(self)
        elif method == "series":
            for model in self._queued_models:
                models.append(self._queued_models[model]())
        else:
            raise ValueError(
                'Invalid run method, accepted run methods are either "parallel" or "series".'
            )

        return models

    def list_models(self):
        """
        Prints out all queued and ran models.
        Examples
        --------
        >>> model.list_models()
        """

        print("######## QUEUED MODELS ########")
        if self._queued_models:
            for key in self._queued_models:
                print(key)
        else:
            print("No queued models.")

        print()

        print("######### RAN MODELS ##########")
        if self._models:
            for key in self._models:
                print(key)
        else:
            print("No ran models.")

    def delete_model(self, name):
        """
        Deletes a model, specified by it's name - can be viewed by calling list_models.
        Will look in both queued and ran models and delete where it's found.
        Parameters
        ----------
        name : str
            Name of the model
        Examples
        --------
        >>> model.delete_model('model1')
        """

        if name in self._queued_models:
            del self._queued_models[name]
        elif name in self._models:
            del self._models[name]
        else:
            raise ValueError(f"Model {name} does not exist")

        self.list_models()

    def compare_models(self):
        """
        Compare different models across every known metric for that model.
        
        Returns
        -------
        Dataframe
            Dataframe of every model and metrics associated for that model
        
        Examples
        --------
        >>> model.compare_models()
        """

        results = []

        for model in self._models:
            results.append(self._models[model].metrics())

        results_table = pd.concat(results, axis=1, join="inner")
        results_table = results_table.loc[:, ~results_table.columns.duplicated()]

        # Move descriptions column to end of dataframe.
        descriptions = results_table.pop("Description")
        results_table["Description"] = descriptions

        return results_table

    def to_pickle(self, name: str):
        """
        Writes model to a pickle file.
        
        Parameters
        ----------
        name : str
            Name of the model
        Examples
        --------
        >>> m = Model(df)
        >>> m.LogisticRegression()
        >>> m.to_pickle('log_reg')
        """

        model_obj = self._models[name]

        to_pickle(model_obj.model, model_obj.model_name)

    def to_service(self, model_name: str, project_name: str):
        """
        Creates an app.py, requirements.txt and Dockerfile in `~/.aethos/projects` and the necessary folder structure
        to run the model as a microservice.
        
        Parameters
        ----------
        model_name : str
            Name of the model to create a microservice of.
        project_name : str
            Name of the project that you want to create.
        Examples
        --------
        >>> m = Model(df)
        >>> m.LogisticRegression()
        >>> m.to_service('log_reg', 'your_proj_name')
        """

        model_obj = self._models[model_name]

        to_pickle(
            model_obj.model,
            model_obj.model_name,
            project=True,
            project_name=project_name,
        )
        tg.generate_service(
            project_name, f"{model_obj.model_name}.pkl", model_obj.model
        )

        print("To run:")
        print("\tdocker build -t `image_name` ./")
        print("\tdocker run -d --name `container_name` -p `port_num`:80 `image_name`")


    ################### TEXT MODELS ########################

    @add_to_queue
    def summarize_gensim(
        self,
        *list_args,
        list_of_cols=[],
        new_col_name="_summarized",
        model_name="model_summarize_gensim",
        run=True,
        **summarizer_kwargs,
    ):
        # region
        """
        Summarize bodies of text using Gensim's Text Rank algorithm. Note that it uses a Text Rank variant as stated here:
        https://radimrehurek.com/gensim/summarization/summariser.html
        The output summary will consist of the most representative sentences and will be returned as a string, divided by newlines.
        
        Parameters
        ----------
        list_of_cols : list, optional
            Column name(s) of text data that you want to summarize
        new_col_name : str, optional
            New column name to be created when applying this technique, by default `_extracted_keywords`
        model_name : str, optional
            Name for this model, default to `model_summarize_gensim`
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        ratio : float, optional
            Number between 0 and 1 that determines the proportion of the number of sentences of the original text to be chosen for the summary.
        word_count : int or None, optional
            Determines how many words will the output contain. If both parameters are provided, the ratio will be ignored.
        split : bool, optional
            If True, list of sentences will be returned. Otherwise joined strings will be returned.
        Returns
        -------
        TextModelAnalysis
            Resulting model
        Examples
        --------
        >>> model.summarize_gensim('col1')
        >>> model.summarize_gensim('col1', run=False) # Add model to the queue
        """
        # endregion

        list_of_cols = _input_columns(list_args, list_of_cols)

        (self.x_train, self.x_test,) = text.gensim_textrank_summarizer(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            new_col_name=new_col_name,
            **summarizer_kwargs,
        )

        self._models[model_name] = TextModelAnalysis(None, self.x_train, model_name)

        return self._models[model_name]

    @add_to_queue
    def extract_keywords_gensim(
        self,
        *list_args,
        list_of_cols=[],
        new_col_name="_extracted_keywords",
        model_name="model_extracted_keywords_gensim",
        run=True,
        **keyword_kwargs,
    ):
        # region
        """
        Extracts keywords using Gensim's implementation of the Text Rank algorithm. 
        Get most ranked words of provided text and/or its combinations.
        
        Parameters
        ----------
        list_of_cols : list, optional
            Column name(s) of text data that you want to summarize
        new_col_name : str, optional
            New column name to be created when applying this technique, by default `_extracted_keywords`
        model_name : str, optional
            Name for this model, default to `model_extract_keywords_gensim`
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        ratio : float, optional
            Number between 0 and 1 that determines the proportion of the number of sentences of the original text to be chosen for the summary.
        words : int, optional
            Number of returned words.
        split : bool, optional
            If True, list of sentences will be returned. Otherwise joined strings will be returned.
        scores : bool, optional
            Whether score of keyword.
        pos_filter : tuple, optional
            Part of speech filters.
        lemmatize : bool, optional 
            If True - lemmatize words.
        deacc : bool, optional
            If True - remove accentuation.
        
        Returns
        -------
        TextModelAnalysis
            Resulting model
        
        Examples
        --------
        >>> model.extract_keywords_gensim('col1')
        >>> model.extract_keywords_gensim('col1', run=False) # Add model to the queue
        """
        # endregion

        list_of_cols = _input_columns(list_args, list_of_cols)

        self.x_train, self.x_test = text.gensim_textrank_keywords(
            x_train=self.x_train,
            x_test=self.x_test,
            list_of_cols=list_of_cols,
            new_col_name=new_col_name,
            **keyword_kwargs,
        )

        self._models[model_name] = TextModelAnalysis(None, self.x_train, model_name)

        return self._models[model_name]


