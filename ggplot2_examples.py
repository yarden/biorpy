##
## Examples using ggplot2 from Rpy2
##

import os
import sys
import time

import scipy
import numpy as np

from r_plot import *

# Data directory
data_dir = "./data"

# Load up the iris dataset
iris_df = pandas.read_table(os.path.join(data_dir, "iris.csv"),
                            sep=",")
print iris_df
# Make Species column to make it more similar to R iris dataset
iris_df["Species"] = iris_df["Name"]

# Make plots dir
plots_dir = "./plots"
if not os.path.isdir(plots_dir):
    os.makedirs(plots_dir)

# Make non-melted dataframe example
def get_nonmelted_df():
    df = pandas.DataFrame({"gene": np.arange(200),
                           "sample1_expr": scipy.randn(200),
                           "sample2_expr": scipy.randn(200),
                           "binding": np.array(["no"]*50+["yes"]*50+["maybe"]*100),
                           "peak_height": scipy.randn(200)*5})
    return df

    
def iris_plots():
    """
    Make some plots with iris
    """
    # Convert iris to R dataframe before plotting
    r_df = conversion_pydataframe(iris_df)
    plot_fname = os.path.join(plots_dir, "iris1.pdf")
    r.pdf(plot_fname)
    # Simple scatter plot
    p = ggplot2.ggplot(r_df) + \
        ggplot2.geom_point(aes_string(x="SepalWidth", y="SepalLength",
                                      colour="Species"))
    p.plot()
    plot_fname = os.path.join(plots_dir, "iris2.pdf")
    r.pdf(plot_fname, width=9, height=5)
    # Separate panel for each species
    p = ggplot2.ggplot(r_df) + \
        ggplot2.geom_point(aes_string(x="SepalWidth", y="SepalLength",
                                      colour="Species")) + \
        ggplot2.facet_grid(Formula("~ Species"))
    p.plot()
    # Horizontal boxplots
    plot_fname = os.path.join(plots_dir, "iris3.pdf")
    r.pdf(plot_fname, width=9, height=5)
    p = ggplot2.ggplot(r_df) + \
        ggplot2.geom_boxplot(aes_string(x="Species", y="SepalWidth", fill="Species")) + \
        ggplot2.coord_flip()
    p.plot()


def melting_dfs():
    """
    Take unmelted df and melt it.
    """
    df = get_nonmelted_df()
    plot_fname = os.path.join(plots_dir, "melt1.pdf")
    r.pdf(plot_fname, width=5, height=5)
    # First, no melting required for many things
    r_df = conversion_pydataframe(df)
    p = ggplot2.ggplot(r_df) + \
        ggplot2.geom_point(aes_string(x="sample1_expr", y="sample2_expr",
                                      size="peak_height",
                                      colour="factor(binding)")) + \
        ggplot2.facet_grid(Formula("binding ~ ."))
    p.plot()
    # Histogram sample1's expression for each binding state
    plot_fname = os.path.join(plots_dir, "melt2.pdf")
    r.pdf(plot_fname, width=9, height=5)
    p = ggplot2.ggplot(r_df) + \
        ggplot2.geom_histogram(aes_string(x="sample1_expr", fill="binding"),
                               color="white") + \
        ggplot2.facet_grid(Formula("~ binding"))
    p.plot()
    # In the original dataframe, we have a column for each
    # sample's expression values: sample1_expr, sample2_expr, etc.
    # If we want to carve up the data for plotting according to
    # which sample's gene expression is being used (i.e. we want to
    # treat the sample name as a variable) then we need to melt
    # the dataframe.  This can be easily done with pandas like this:
    ## Melt the dataframe so that the only variable is whether we're looking
    ## at sample1 or sample2
    melted_df = pandas.melt(df, id_vars=["gene", "peak_height", "binding"])
    # The melted dataframe is now twice as big, since two distinct columns got merged
    # into long form ("melted form") as rows
    print "Melted df has %d elements." %(len(melted_df))
    # Make it an R dataframe
    r_melted = conversion_pydataframe(melted_df)
    # Plot the expression for each sample across different bindings
    plot_fname = os.path.join(plots_dir, "melt3.pdf")
    r.pdf(plot_fname)
    p = ggplot2.ggplot(r_melted) + \
        ggplot2.geom_histogram(aes_string(x="value", colour="binding"), fill="white") + \
        ggplot2.facet_grid(Formula("variable ~ binding"))
    p.plot()

    
    
    

def main():
    iris_plots()
    melting_dfs()


if __name__ == "__main__":
    main()
