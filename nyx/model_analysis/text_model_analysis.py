from IPython.display import HTML, display

from .model_analysis import ModelAnalysisBase

class TextModelAnalysis(ModelAnalysisBase):
    def __init__(self, model, data, model_name, **kwargs):
        """
        Class to analyze Text models through metrics and visualizations.
        Parameters
        ----------
        model : str or Model Object
            Model object or .pkl file of the objects.
        data : pd.DataFrame
            Training Data used for the model.
        model_name : str
            Name of the model for saving images and model tracking purposes
        corpus : list, optional
            Gensim LDA corpus variable. NOTE: Only for Gensim LDA
        id2word : list, optional
            Gensim LDA corpus variable. NOTE: Only for Gensim LDA
        """

        self.model = model
        self.x_train = data
        self.model_name = model_name

        # LDA dependant variables
        self.corpus = kwargs.pop("corpus", None)
        self.id2word = kwargs.pop("id2word", None)

    def view(self, original_text, model_output):
        """
        View the original text and the model output in a more user friendly format
        
        Parameters
        ----------
        original_text : str
            Column name of the original text
        model_output : str
            Column name of the model text
        Examples
        --------
        >>> m = model.LDA()
        >>> m.view('original_text_col_name', 'model_output_col_name')
        """

        results = self.x_train[[original_text, model_output]]

        display(HTML(results.to_html()))

    def view_topics(self, num_topics=10, **kwargs):
        """
        View topics from topic modelling model.
        
        Parameters
        ----------
        num_topics : int, optional
            Number of topics to view, by default 10
        Returns
        --------
        str
            String representation of topics and probabilities
        Examples
        --------
        >>> m = model.LDA()
        >>> m.view_topics()
        """

        return self.model.show_topics(num_topics=num_topics, **kwargs)

    def view_topic(self, topic_num: int, **kwargs):
        """
        View a specific topic from topic modelling model.
        
        Parameters
        ----------
        topic_num : int
        Returns
        --------
        str
            String representation of topic and probabilities
        Examples
        --------
        >>> m = model.LDA()
        >>> m.view_topic(1)
        """

        return self.model.show_topic(topicid=topic_num, **kwargs)
