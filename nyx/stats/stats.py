import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
import scipy as sc

from scipy.stats.stats import ks_2samp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from collections import Counter
from typing import Union
from nyx.stats.util import run_2sample_ttest


class Stats(object):


    def predict_data_sample(self):

        """
        Identifies how similar the train and test set distribution are by trying to predict whether each sample belongs
        to the train or test set using Random Forest, 10 Fold Stratified Cross Validation.
        The lower the F1 score, the more similar the distributions are as it's harder to predict which sample belongs to which distribution.
        Credit: https://www.kaggle.com/nanomathias/distribution-of-test-vs-training-data#1.-t-SNE-Distribution-Overview
        Returns
        -------
        Data:
            Returns a deep copy of the Data object.
        Examples
        --------
        >>> data.predict_data_sample()
        """

        if self.x_test is None or not self.target:
            raise ValueError(
                "Test data or target field must be set. They can be set by assigning values to the `target` or the `x_test` variable."
            )

        x_train = self.x_train.drop(self.target, axis=1)
        x_test = self.x_test.drop(self.target, axis=1)

        x_train["label"] = 1
        x_test["label"] = 0

        data = pd.concat([x_train, x_test], axis=0)
        label = data["label"].tolist()

        predictions = cross_val_predict(
            ExtraTreesClassifier(n_estimators=100),
            data.drop(columns=["label"]),
            label,
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        )

        print(classification_report(data["label"].tolist(), predictions))

        return self

    def ks_feature_distribution(self, threshold=0.1, show_plots=True):

        """
        Uses the Kolomogorov-Smirnov test see if the distribution in the training and test sets are similar.
        
        Credit: https://www.kaggle.com/nanomathias/distribution-of-test-vs-training-data#1.-t-SNE-Distribution-Overview
        Parameters
        ----------
        threshold : float, optional
            KS statistic threshold, by default 0.1
        show_plots : bool, optional
            True to show histograms of feature distributions, by default True
        Returns
        -------
        DataFrame
            Columns that are significantly different in the train and test set.
        Examples
        --------
        >>> data.ks_feature_distribution()
        >>> data.ks_feature_distribution(threshold=0.2)
        """

        import swifter
        from tqdm import tqdm


        if self.x_test is None:
            raise ValueError(
                "Data must be split into train and test set. Please set the `x_test` variable."
            )

        diff_data = []
        diff_df = None

        for col in tqdm(self.x_train.columns):
            statistic, pvalue = ks_2samp(
                self.x_train[col].values, self.x_test[col].values
            )

            if pvalue <= 0.05 and np.abs(statistic) > threshold:
                diff_data.append(
                    {
                        "feature": col,
                        "p": np.round(pvalue, 5),
                        "statistic": np.round(np.abs(statistic), 2),
                    }
                )

        if diff_data:
            diff_df = pd.DataFrame(diff_data).sort_values(
                by=["statistic"], ascending=False
            )

            if show_plots:
                n_cols = 4
                n_rows = int(len(diff_df) / n_cols) + 1

                _, ax = plt.subplots(n_rows, n_cols, figsize=(40, 8 * n_rows))

                for i, (_, row) in enumerate(diff_df.iterrows()):
                    if i >= len(ax):
                        break

                    extreme = np.max(
                        np.abs(
                            self.x_train[row.feature].tolist()
                            + self.x_test[row.feature].tolist()
                        )
                    )
                    self.x_train.loc[:, row.feature].swifter.apply(np.log1p).hist(
                        ax=ax[i],
                        alpha=0.6,
                        label="Train",
                        density=True,
                        bins=np.arange(-extreme, extreme, 0.25),
                    )

                    self.x_test.loc[:, row.feature].swifter.apply(np.log1p).hist(
                        ax=ax[i],
                        alpha=0.6,
                        label="Train",
                        density=True,
                        bins=np.arange(-extreme, extreme, 0.25),
                    )

                    ax[i].set_title(f"Statistic = {row.statistic}, p = {row.p}")
                    ax[i].set_xlabel(f"Log({row.feature})")
                    ax[i].legend()

                plt.tight_layout()
                plt.show()

        return diff_df

    def most_common(
        self, col: str, n=15, plot=False, use_test=False, output_file="", **plot_kwargs
    ):

        """
        Analyzes the most common values in the column and either prints them or displays a bar chart.
        
        Parameters
        ----------
        col : str
            Column to analyze
        n : int, optional
            Number of top most common values to display, by default 15
        plot : bool, optional
            True to plot a bar chart, by default False
        use_test : bool, optional
            True to analyze the test set, by default False
        output_file : str,
            File name to save plot as, IF plot=True
        Examples
        --------
        >>> data.most_common('col1', plot=True)
        >>> data.most_common('col1', n=50, plot=True)
        >>> data.most_common('col1', n=50)
        """

        if use_test:
            data = self.x_test[col].tolist()
        else:
            data = self.x_train[col].tolist()

        test_sample = data[0]

        if isinstance(test_sample, list):
            data = itertools.chain(*map(list, data))
        elif isinstance(test_sample, str):
            data = map(str.split, data)
            data = itertools.chain(*data)

        counter = Counter(data)
        most_common = dict(counter.most_common(n))

        if plot:
            df = pd.DataFrame(list(most_common.items()), columns=["Word", "Count"])

            fig = self._viz.barplot(
                x="Word", y="Count", data=df, output_file=output_file, **plot_kwargs
            )

            return fig
        else:
            for k, v in most_common.items():
                print(f"{k}: {v}")

            return most_common

    def ind_ttest(self, group1: str, group2: str, equal_var=True, output_file=None):

        results = run_2sample_ttest(
            group1, group2, self.x_train, "ind", output_file, equal_var=equal_var
        )

        matrix = [
            ["", "Test Statistic", "p-value"],
            ["Sample Data", results[0], results[1]],
        ]

        self._viz.create_table(matrix, True, output_file)

        return results

            

       
     

        
                        
