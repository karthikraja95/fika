import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from nyx.config import IMAGE_DIR, cfg
from nyx.util import _make_dir

class VizCreator(object):

    def raincloud(self, col:str, target_col: str,
        data: pd.DataFrame, output_file="", **params):

        """
        Visualizes 2 columns using raincloud.
        
        Parameters
        ----------
        col : str
            Column name of general data
        target_col : str
            Column name of measurable data, numerical
        data : Dataframe
            Dataframe of the data
        params: dict
            Parameters for the RainCloud visualization
        ouput_file : str
            Output file name for the image including extension (.jpg, .png, etc.)
        """

        import ptitprince as pt

        fig, ax = plt.subplots(figsize=(12,8))

        if not params:

            params = {
                "pointplot": True,
                "width_viol": 0.8,
                "width_box": 0.4,
                "orient": "h",
                "move": 0.0,
                "ax":ax,
            }

        ax = pt.RainCloud(x=col, y=target_col, data=data.infer_objects(), **params)

        if output_file:
            fig.savefig(os.path(IMAGE_DIR, output_file))

        return ax

    def barplot(self, x:str, y:str, data:pd.DataFrame,
        method=None, asc=None, output_file="", **barplot_kwargs):

        """
        Visualizes a bar plot.
        
        Parameters
        ----------
        x : str
            Column name for the x axis.
        y : str
            Columns for the y axis
        data : Dataframe
            Dataset
        method : str
            Method to aggregate groupy data
            Examples: min, max, mean, etc., optional
            by default None
        asc : bool
            To sort values in ascending order, False for descending
        """

        import ploty.express as px

        orient = barplot_kwargs.get("orientation", None)

        if method:

            if orient == "h":
                data = data.groupby(y, as_index=False)
            else:
                data = data.groupby(x, as_index=False)

            data = getattr(data, method)()

            if not y:
                y = data.iloc[:, 1].name

        if asc in not None:

            data[x] = data[x].astype(str)
            data = data.sort_values(y, ascending = asc)

        fig = px.bar(data, x=x, y=y, **barplot_kwargs)

        if as in not None:
            fig.update_layout(xaxis_type="category")

        if output_file:
            fig.write_image(os.path.join(IMAGE_DIR, output_file))

        return fig

    def scatterplot(
        self,
        x: str,
        y: str,
        z=None,
        data=None,
        color=None,
        title="Scatter Plot",
        output_file="",
        **scatterplot_kwargs,
    ):

       if color:
            data[color] = data[color].astype(str)

        if z is None:
            fig = px.scatter(
                data, x=x, y=y, color=color, title=title, **scatterplot_kwargs
            )

        else:
            fig = px.scatter_3d(
                data, x=x, y=y, z=z, color=color, title=title, **scatterplot_kwargs
            )

        if output_file:  # pragma: no cover
            fig.write_image(os.path.join(IMAGE_DIR, output_file))

        return fig 