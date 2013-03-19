##
## R plotting utilities
##
import sys
import pandas
import numpy as np
##
## Wrapper to Rpy2.
##
## Transparently convert Pandas objects (DataFrames, Series) into
## corresponding R objects.  Also handle ordinary numpy arrays/matrices.
##
## 

import scipy
import rpy2

from collections import OrderedDict

from rpy2.robjects import r
import rpy2.robjects as robj
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.lib import grid
from rpy2.robjects.lib import ggplot2
lattice = importr("lattice")
# Conversion utilities
import pandas.rpy.common as com
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.lib import grid
from rpy2.robjects import r, Formula
py2ri_orig = rpy2.robjects.conversion.py2ri

from rpy2.robjects.lib.ggplot2 import ggplot, \
                               aes_string, \
                               geom_histogram, \
                               element_blank, \
                               theme_line, \
                               theme_blank, \
                               theme_bw, \
                               theme

def get_nogrid_theme():
    """
    Get no grid theme for ggplot2.
    """
    nogrid_x_theme = theme(**{'panel.grid.major.x': element_blank(),
                              'panel.grid.minor.x': element_blank(),
                              'panel.grid.major.y': element_blank(),
                              'panel.grid.minor.y': element_blank()})
    return nogrid_x_theme


def r_grid(nrows, ncols):
    lt = grid.layout(nrows, ncols)
    vp = grid.viewport(layout = lt)
    vp.push()
    return vp


def conversion_pydataframe(obj):
    if isinstance(obj, pandas.core.frame.DataFrame):
        od = OrderedDict()
        for name, values in obj.iteritems():
            if values.dtype.kind == 'O':
                od[name] = rpy2.robjects.vectors.StrVector(values)
            else:
                od[name] = rpy2.robjects.conversion.py2ri(values)
        return rpy2.robjects.vectors.DataFrame(od)
    else:
        return py2ri_orig(obj)

rpy2.robjects.conversion.py2ri = conversion_pydataframe
#py2ri = rpy2.robjects.conversion.py2ri

def convert_pandas_to_r(myobj):
    """
    Convert Pandas/Numpy objects to R objects.

    If the input object is a Pandas DataFrame, convert it to
    an R DataFrame.  If it's a Series, treat it like a vector/numpy
    array. 
    """
    if isinstance(myobj, pandas.core.frame.DataFrame):
        return pandas_data_frame_to_rpy2_data_frame(myobj)
    elif isinstance(myobj, pandas.Series):
        return py2ri_orig(myobj)
    return myobj

def pandas_data_frame_to_rpy2_data_frame(pDataframe):
    """
    Convert a pandas DataFrame to an Rpy2 DataFrame.
    """
    orderedDict = OrderedDict()

    for columnName in pDataframe.columns:
        columnValues = pDataframe[columnName].values
        filteredValues = \
            [value if pandas.notnull(value) else robj.NA_Real \
             for value in columnValues]
        try:
            orderedDict[columnName] = robj.FloatVector(filteredValues)
        except ValueError:
            orderedDict[columnName] = robj.StrVector(filteredValues)

    rDataFrame = robj.DataFrame(orderedDict)
    rDataFrame.rownames = robj.StrVector(pDataframe.index)

    return rDataFrame

#py2ri = pandas_data_frame_to_rpy2_data_frame
py2ri = convert_pandas_to_r

def py2mat(myobj):
    """
    Convert Python series to R matrix.
    """
    if isinstance(myobj, pandas.Series):
        mat = r.matrix(myobj,
                       rownames=myobj.index,
                       dimnames=myobj.name)
    else:
        mat = r.matrix(myobj)
    return mat


def convert_to_r_matrix(df, strings_as_factors=False):

    """
    Convert a pandas DataFrame to a R matrix.

    Parameters
    ----------
    df: The DataFrame being converted
    strings_as_factors: Whether to turn strings into R factors (default: False)

    Returns
    -------
    A R matrix
    """
    if isinstance(df, pandas.Series):
        # If it's a Series, cast it to a DataFrame
        df = pandas.DataFrame(df)
    r_dataframe = pandas_data_frame_to_rpy2_data_frame(df)
    as_matrix = robj.baseenv.get("as.matrix")
    r_matrix = as_matrix(r_dataframe)
    return r_matrix


def plot_qc_reads(qc_df):
    """
    Plot number of reads part of a pipeline QC file.
    """
    # Record NA values as 0
    qc_df = qc_df.fillna(0)#.set_index("sample")
    cols = ["sample",
            "num_reads",
            "num_mapped",
            "num_unique_mapped",
            "num_junctions"]
    qc_df = qc_df[cols]
    melted_qc = pandas.melt(qc_df, id_vars=["sample"])
    qc_r = conversion_pydataframe(melted_qc)
    labels = tuple(["num_reads",
                    "num_mapped",
                    "num_unique_mapped",
                    "num_junctions"])
    labels = robj.StrVector(labels)
    variable_i = qc_r.names.index('variable')
    qc_r[variable_i] = robj.FactorVector(qc_r[variable_i],
                                         levels = labels)
    ggplot2.theme_set(ggplot2.theme_bw(12))
    scales = importr("scales")
    r_opts = r.options(scipen=4)
    p = ggplot2.ggplot(qc_r) + \
        ggplot2.geom_point(aes_string(x="sample", y="value")) + \
        ggplot2.scale_y_continuous(trans=scales.log10_trans(),
                                   breaks=scales.trans_breaks("log10",
                                                              robj.r('function(x) 10^x')),
                                   labels=scales.trans_format("log10",
                                                              robj.r('math_format(10^.x)'))) + \
        r.xlab("CLIP-Seq samples") + \
        r.ylab("No. reads") + \
        ggplot2.coord_flip() + \
        ggplot2.facet_wrap(Formula("~ variable"), ncol=1) + \
        theme(**{"panel.grid.major.x": element_blank(),
                 "panel.grid.minor.x": element_blank(),
                 "panel.grid.major.y": theme_line(size=0.5,colour="grey66",linetype=3)})
    p.plot()

    return
    r.par(mfrow=np.array([1,2]))
    num_samples = len(qc_df.num_reads)
    r.par(bty="n", lwd=1.7, lty=2)
    r_opts = r.options(scipen=4)
    r.options(r_opts)
    r.dotchart(convert_to_r_matrix(qc_df[["num_reads",
                                          "num_mapped",
                                          "num_unique_mapped"]]),
               xlab="No. reads",
               lcolor="black",
               pch=19,
               gcolor="darkblue",
               cex=0.8)
    r.par(bty="n")
    r.dotchart(convert_to_r_matrix(qc_df[["num_ribosub_mapped",
                                          "num_ribo",
                                          "num_junctions"]]),
               xlab="No. reads",
               lcolor="black",
               pch=19,
               gcolor="darkblue",
               cex=0.8)


def plot_qc_percents(qc_df):
    """
    Plot percentage parts of pipeline QC file.
    """
    # Record NA values as 0
    qc_df = qc_df.fillna(0).set_index("sample")
    r.par(mfrow=np.array([1,2]))
    num_samples = len(qc_df.num_reads)
    r_opts = r.options(scipen=10)
    r.options(r_opts)
    r.par(bty="n", lwd=1.7, lty=2)
    r.dotchart(convert_to_r_matrix(qc_df[["percent_mapped",
                                          "percent_unique",
                                          "percent_ribo"]]),
               xlab="Percent reads",
               lcolor="black",
               pch=19,
               gcolor="darkblue",
               cex=0.8)
    r.par(bty="n")
    r.dotchart(convert_to_r_matrix(qc_df[["percent_exons",
                                          "percent_cds",
                                          "percent_3p_utr",
                                          "percent_5p_utr",                                          
                                          "percent_introns"]]),
               xlab="Percent reads",
               lcolor="black",
               pch=19,
               gcolor="darkblue",
               cex=0.8)


def r_set_df_factor(r_df, variable_name, values):
    """
    Set the DataFrame variable_name as values, assuming
    that it's a factor.
    """
    labels = tuple(values)
    labels = robj.StrVector(labels)
    variable_i = r_df.names.index(variable_name)
    r_df[variable_i] = robj.FactorVector(r_df[variable_i],
                                         levels=labels)
    return r_df
    

    
#    r.dotchart(convert_to_r_matrix(qc_df.percent_exons),
#               cex=.7,
#               color=colors)
#    r.dotchart(convert_to_r_matrix(qc_df.percent_introns),
#               cex=.7,
#               color=colors)

#    x <- mtcars[order(mtcars$mpg),] # sort by mpg
#    x$cyl <- factor(x$cyl) # it must be a factor
#    x$color[x$cyl==4] <- "red"
#    x$color[x$cyl==6] <- "blue"
#    x$color[x$cyl==8] <- "darkgreen"
#    dotchart(x$mpg,labels=row.names(x),cex=.7,groups= x$cyl,
#             main="Gas Milage for Car Models\ngrouped by cylinder",
#             xlab="Miles Per Gallon", gcolor="black", color=x$color)    
#    r.axis(1, at=np.arange(1, num_samples+1),
#           las=2)
    #r.barplot(py2ri(qc_df.num_reads))
#    r.plot(py2ri(qc_df))


if __name__ == "__main__":
    print "Plotting"

    from rpy2.robjects.lib import ggplot2
    from rpy2.robjects import r
    from rpy2.robjects.packages import importr

    scales = importr('scales')

    iris = r('iris')

    r.pdf("/home/yarden/jaen/Musashi/rtest.pdf")

    iris_py = pandas.read_csv("/home/yarden/iris.csv")
    iris_py = iris_py.rename(columns={"Name": "Species"})
    corrs = []
    from scipy.stats import spearmanr
    for species in set(iris_py.Species):
        entries = iris_py[iris_py["Species"] == species]
        c = spearmanr(entries["SepalLength"], entries["SepalWidth"])
        print "c: ", c

    # compute r.cor(x, y) and divide up by Species
    # Assume we get a vector of length Species saying what the
    # correlation is for each Species' Petal Length/Width
    p = ggplot2.ggplot(iris) + \
        ggplot2.geom_point(ggplot2.aes_string(x="Sepal.Length", y="Sepal.Width")) + \
        ggplot2.facet_wrap(Formula("~Species")) 
    p.plot()
    r["dev.off"]()    

    sys.exit(1)
    grdevices = importr('grDevices')
    ggplot2.theme_set(ggplot2.theme_bw(12))

    p = ggplot2.ggplot(iris) + \
        ggplot2.geom_point(ggplot2.aes_string(x="Sepal.Length", y="Sepal.Width")) + \
        ggplot2.facet_wrap(Formula('~ Species'), ncol=2, nrow = 2) + \
        ggplot2.geom_text(aes_string(x="Sepal.Length", y="Sepal.Width"), label="t") + \
        ggplot2.GBaseObject(r('ggplot2::coord_fixed')()) # aspect ratio
    p.plot()

