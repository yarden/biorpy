##
## biorpy plotting wrappers
##
## nspies in ze house
##
import numpy
import pandas
import rpy2

#from rpy2.robjects import r
from biorpy.betteR import BetteR
import rpy2.robjects.numpy2ri
from rpy2 import robjects as robj
from rpy2.rlike.container import TaggedList
import pandas.rpy.common as com
rpy2.robjects.numpy2ri.activate()

r = BetteR()

def plot(x, y, **kwargs):
    """
    Wrapper to robjects.r

    Handles DataFrames and numpy vectors intelligently.
    """
    # Set empty x/y axis labels to avoid
    # overly verbose labeling when passing in
    # pandas DataFrames 
    if "xlab" not in kwargs:
        kwargs["xlab"] = ""
    if "ylab" not in kwargs:
        kwargs["ylab"] = ""
    # Call r.plot here with supplied keyword arguments...

def plotMulti(xs, ys, names, colors=None, legendWhere="bottomright", **kwdargs):
    assert len(ys) == len(names)
    if len(xs) != len(ys):
        xs = [xs for i in range(len(names))]
    assert len(xs) == len(ys)

    if colors is None:
        colors = ["red", "blue", "green", "orange", "brown", "purple", "black"]

    ylim = [min(min(y) for y in ys), max(max(y) for y in ys)]
    xlim = [min(min(x) for x in xs), max(max(x) for x in xs)]

    plotargs = {"xlab":"", "ylab":"", "xlim":xlim, "ylim":ylim}
    plotargs.update(kwdargs)

    for i in range(len(xs)):
        if i == 0:
            r.plot(xs[0], ys[0], col=colors[0], type="l", **plotargs)
        else:
            r.lines(xs[i], ys[i], col=colors[i%len(colors)])

    r.legend(legendWhere, legend=names, lty=1, lwd=2, col=colors, bg="white")


def plotWithCor(x, y, method="spearman", main="", **kwdargs):
    cor = r.cor(x, y, method=method)[0]
        
    r.plot(x, y, main="{} rs = {}".format(main, cor), **kwdargs)

def plotWithFit(x, y, main="", fitkwdargs={}, **plotkwdargs):
    import scipy.stats

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

    r.plot(x, y, main="{} r={:.2g} p={:.2g}".format(main, r_value, p_value), **plotkwdargs)
    r.abline(a=intercept, b=slope, **fitkwdargs)

def errbars(x=None, y=None, x_lower=None, x_upper=None, y_lower=None, y_upper=None, length=0.08, *args, **kwdargs):
    if y is not None and  x_lower is not None  and x_upper is not None:
        r.arrows(x_lower, y, x_upper, y, angle = 90, code = 3, length = length, *args, **kwdargs)
    elif x is not None and y_lower is not None and y_upper is not None:
        r.arrows(x, y_lower, x, y_upper, angle = 90, code = 3, length = length, *args, **kwdargs)
    else:
        raise Exception("must define either (y, x_lower, x_upper) or (x, y_lower, y_upper)")
            
    

def ecdf(vectors, labels, colors=["red", "blue", "orange", "violet", "green", "brown"],
         xlab="", ylab="cumulative fraction", main="", legendWhere="topleft", 
         lty=1, lwd=1, **ecdfKwdArgs):
    """ Take a list of lists, convert them to vectors, and plots them sequentially on a CDF """

    #print "MEANS:", main
    #for vector, label in zip(convertToVectors, labels):
    #    print label, numpy.mean(vector)
    
    def _expand(item):
        try:
            iter(item)
            return item
        except TypeError:
            return [item] * len(vectors)
            
    
    lty = _expand(lty)
    lwd = _expand(lwd)

    ecdfKwdArgs.update({"verticals":True, "do.points":False, "col.hor":colors[0], "col.vert":colors[0], "lty":lty[0], "lwd":lwd[0]})

    if not "xlim" in ecdfKwdArgs:
        xlim = [min(min(vector) for vector in vectors),
                max(max(vector) for vector in vectors)]
        ecdfKwdArgs["xlim"] = xlim

    r.plot(r.ecdf(vectors[0]), main=main, xlab=xlab, ylab=ylab, **ecdfKwdArgs)

    for i, vector in enumerate(vectors[1:]):
        r.plot(r.ecdf(vector), add=True,
                    **{"verticals":True, "do.points":False, "col.hor":colors[i+1], "col.vert":colors[i+1],
                       "lty":lty[i+1], "lwd":lwd[i+1]})

    labelsWithN = []
    for i, label in enumerate(labels):
        labelsWithN.append(label+" (n=%d)"%len(vectors[i]))
    r.legend(legendWhere, legend=labelsWithN, lty=lty, lwd=[lwdi*2 for lwdi in lwd], col=colors, cex=0.7, bg="white")



def boxPlot(dict_, keysInOrder=None, *args, **kwdargs):
    if not keysInOrder:
        keysInOrder = sorted(dict_.keys())
        
    t = TaggedList([])
    for key in keysInOrder:
        t.append(robj.FloatVector(dict_[key]), "X:"+str(key))

    x = r.boxplot(t, names=keysInOrder,*args, **kwdargs)
    return x

def barPlot(dict_, keysInOrder=None, printCounts=True, *args, **kwdargs):
    if not keysInOrder:
        keysInOrder = sorted(dict_.keys())
    
    heights = [dict_[key] for key in keysInOrder]

    kwdargs["names.arg"] = keysInOrder

    if printCounts:
        ylim = [0, max(heights)*1.1]
    else:
        ylim = [0, max(heights)]

    x = r.barplot(heights, ylim=ylim, *args, **kwdargs)

    if printCounts:
        r.text(x, heights, heights, pos=3)
    return x



def scatterplotMatrix(dataFrame, main="", **kwdargs):
    """ Plots a scatterplot matrix, with scatterplots in the upper left and correlation
    values in the lower right. Input is a pandas DataFrame.
    """
    robj.r.library("lattice")

    taggedList = TaggedList(map(robj.FloatVector, [dataFrame[col] for col in dataFrame.columns]), dataFrame.columns)

    #print taggedList
    #df = robj.r['data.frame'](**datapointsDict)
    #df = robj.r['data.frame'](taggedList)
    df = robj.DataFrame(taggedList)
    #print df
    #robj.r.splom(df)
    #robj.r.pairs(df)

    robj.r("""panel.cor <- function(x, y, digits=2, prefix="", cex.cor)
    {
        usr <- par("usr"); on.exit(par(usr))
        par(usr = c(0, 1, 0, 1))
        r <- cor(x, y, method="spearman")
        scale = abs(r)*0.8+0.2
        txt <- format(c(r, 0.123456789), digits=digits)[1]
        txt <- paste(prefix, txt, sep="")
        if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
        text(0.5, 0.5, txt, cex = cex.cor * scale+0.2)
    }
    """)
    robj.r("""panel.hist <- function(x, ...)
    {
        usr <- par("usr"); on.exit(par(usr))
        par(usr = c(usr[1:2], 0, 1.5) )
        h <- hist(x, plot = FALSE)
        breaks <- h$breaks; nB <- length(breaks)
        y <- h$counts; y <- y/max(y)
        rect(breaks[-nB], 0, breaks[-1], y, col="lightgrey", ...)
    }""")
                                        

    additionalParams = {"upper.panel": robj.r["panel.smooth"], "lower.panel": robj.r["panel.cor"], "diag.panel":robj.r["panel.hist"]}
    additionalParams.update(kwdargs)
    robj.r["pairs"](df, main=main, **additionalParams)


def plotWithSolidErrbars(x, y, upper, lower, add=False, errbarcol="lightgray", plotargs={}, polygonargs={}):
    x = numpy.asarray(x)

    errbarx = numpy.concatenate([x, x[::-1]])
    errbary = numpy.concatenate([upper, lower[::-1]])

    if not add:
        r.plot(x, y, type="n", **plotargs)

    polygondefaults = {"border":"NA"}
    polygonargs.update(polygondefaults)

    r.polygon(errbarx, errbary, col=errbarcol, **polygonargs)

    r.lines(x, y, **plotargs)

    return x, y, upper, lower, errbarx, errbary