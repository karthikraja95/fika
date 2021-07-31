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