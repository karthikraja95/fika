from nyx.visualizations.visualize import VizCreator
import numpy as np
import pandas as pd

class Visualizations(object):
    @property
    def _viz(self):
        return VizCreator()

    @property
    def plot_colors(self):  # pragma: no cover
        """
        Displays all plot colour names
        """

        from IPython.display import IFrame

        IFrame(
            "https://python-graph-gallery.com/wp-content/uploads/100_Color_names_python.png"
        )