import pandas as pd

from nyx.modelling.model import ModelBase
from nyx.config import shell
from nyx.model_analysis.unsupervised_model_analysis import UnsupervisedModelAnalysis
from nyx.analysis import Analysis
from nyx.cleaning.clean import Clean
from nyx.preprocessing.preprocess import Preprocess
from nyx.feature_engineering.feature import Feature
from nyx.visualizations.visualizations import Visualizations
from nyx.stats.stats import Stats
from nyx.modelling.util import add_to_queue

class Unsupervised(
    ModelBase, Analysis, Clean, Preprocess, Feature, Visualizations, Stats
):
    def __init__(
        self, x_train, exp_name="my-experiment",
    ):
        """
        Class to run analysis, transform your data and run Unsupervised algorithms.
        Parameters
        -----------
        x_train: pd.DataFrame
            Training data or aethos data object
        exp_name : str
            Experiment name to be tracked in MLFlow.
        """

        super().__init__(
            x_train, "", x_test=None, test_split_percentage=0.2, exp_name=exp_name,
        )

    @add_to_queue
    def KMeans(
        self, model_name="km", run=True, verbose=1, **kwargs,
    ):
        # region
        """
        NOTE: If 'n_clusters' is not provided, k will automatically be determined using an elbow plot using distortion as the mteric to find the optimal number of clusters.
        K-means clustering is one of the simplest and popular unsupervised machine learning algorithms.
        The objective of K-means is simple: group similar data points together and discover underlying patterns.
        To achieve this objective, K-means looks for a fixed number (k) of clusters in a dataset.
        In other words, the K-means algorithm identifies k number of centroids,
        and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible.
        For a list of all possible options for K Means clustering please visit: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        Parameters
        ----------
        model_name : str, optional
            Name for this model, by default "kmeans"
        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False
        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1
        n_clusters : int, optional, default: 8
            The number of clusters to form as well as the number of centroids to generate.
        init : {‘k-means++’, ‘random’ or an ndarray}
            Method for initialization, defaults to ‘k-means++’:
                ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.
                ‘random’: choose k observations (rows) at random from data for the initial centroids.
            If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
        n_init : int, default: 10
            Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
        max_iter : int, default: 300
            Maximum number of iterations of the k-means algorithm for a single run.
        random_state : int, RandomState instance or None (default)
            Determines random number generation for centroid initialization. Use an int to make the randomness deterministic. See Glossary.
        algorithm : “auto”, “full” or “elkan”, default=”auto”
            K-means algorithm to use.
            The classical EM-style algorithm is “full”. The “elkan” variation is more efficient by using the triangle inequality, but currently doesn’t support sparse data. 
            “auto” chooses “elkan” for dense data and “full” for sparse data.
                    
        Returns
        -------
        UnsupervisedModelAnalysis
            UnsupervisedModelAnalysis object to view results and further analysis
        Examples
        --------
        >>> model.KMeans()
        >>> model.KMeans(model_name='kmean_1, n_cluster=5)
        >>> model.KMeans(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.cluster import KMeans

        def find_optk():

            from yellowbrick.cluster import KElbowVisualizer

            model = KMeans(**kwargs)

            visualizer = KElbowVisualizer(model, k=(4, 12))
            visualizer.fit(self.train_data)
            visualizer.show()

            print(f"Optimal number of clusters is {visualizer.elbow_value_}.")

            return visualizer.elbow_value_

        n_clusters = kwargs.pop("n_clusters", None)

        if not n_clusters:
            n_clusters = find_optk()

        model = KMeans

        model = self._run_unsupervised_model(
            model,
            model_name,
            run=run,
            verbose=verbose,
            n_clusters=n_clusters,
            **kwargs,
        )

        return model
