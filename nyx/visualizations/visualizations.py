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

    @property
    def plot_colorpalettes(self):  # pragma: no cover
        """
        Displays color palette configuration guide.
        """

        from IPython.display import IFrame

        IFrame("https://seaborn.pydata.org/tutorial/color_palettes.html")

    def raincloud(self, x=None, y=None, output_file="", **params):
        """
        Combines the box plot, scatter plot and split violin plot into one data visualization.
        This is used to offer eyeballed statistical inference, assessment of data distributions (useful to check assumptions),
        and the raw data itself showing outliers and underlying patterns.
        A raincloud is made of:
        1) "Cloud", kernel desity estimate, the half of a violinplot.
        2) "Rain", a stripplot below the cloud
        3) "Umberella", a boxplot
        4) "Thunder", a pointplot connecting the mean of the different categories (if `pointplot` is `True`)
        Useful parameter documentation
        ------------------------------
        https://seaborn.pydata.org/generated/seaborn.boxplot.html
        https://seaborn.pydata.org/generated/seaborn.violinplot.html
        https://seaborn.pydata.org/generated/seaborn.stripplot.html
        Parameters
        ----------
        x : str
            X axis data, reference by column name, any data
        y : str
            Y axis data, reference by column name, measurable data (numeric)
            by default target
        hue : Iterable, np.array, or dataframe column name if 'data' is specified
            Second categorical data. Use it to obtain different clouds and rainpoints
        output_file : str, optional
            Output file name for image with extension (i.e. jpeg, png, etc.)
        orient : str                  
            vertical if "v" (default), horizontal if "h"
        width_viol : float            
            width of the cloud
        width_box : float             
            width of the boxplot
        palette : list or dict        
            Colours to use for the different levels of categorical variables
        bw : str or float
            Either the name of a reference rule or the scale factor to use when computing the kernel bandwidth,
            by default "scott"
        linewidth : float             
            width of the lines
        cut : float
            Distance, in units of bandwidth size, to extend the density past the extreme datapoints.
            Set to 0 to limit the violin range within the range of the observed data,
            by default 2
        scale : str
            The method used to scale the width of each violin.
            If area, each violin will have the same area.
            If count, the width of the violins will be scaled by the number of observations in that bin.
            If width, each violin will have the same width.
            By default "area"
        jitter : float, True/1
            Amount of jitter (only along the categorical axis) to apply.
            This can be useful when you have many points and they overlap,
            so that it is easier to see the distribution. You can specify the amount of jitter (half the width of the uniform random variable support),
            or just use True for a good default.
        move : float                  
            adjust rain position to the x-axis (default value 0.)
        offset : float                
            adjust cloud position to the x-axis
        color : matplotlib color
            Color for all of the elements, or seed for a gradient palette.
        ax : matplotlib axes
            Axes object to draw the plot onto, otherwise uses the current Axes.
        figsize : (int, int)    
            size of the visualization, ex (12, 5)
        pointplot : bool   
            line that connects the means of all categories, by default False
        dodge : bool 
            When hue nesting is used, whether elements should be shifted along the categorical axis.
        Source: https://micahallen.org/2018/03/15/introducing-raincloud-plots/
        
        Examples
        --------
        >>> data.raincloud('col1') # Will plot col1 values on the x axis and your target variable values on the y axis
        >>> data.raincloud('col1', 'col2') # Will plot col1 on the x and col2 on the y axis
        >>> data.raincloud('col1', 'col2', output_file='raincloud.png')
        """

        if y is None:
            y = self.target

        fig = self._viz.raincloud(y, x, self.x_train, output_file=output_file, **params)

        return fig