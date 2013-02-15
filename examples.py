import numpy

from rpy2 import robjects as robj
from rpy2.rlike.container import TaggedList
import rpy2.robjects.numpy2ri


def boxPlot(dict_, keysInOrder=None, *args, **kwdargs):
    # NEEDS A LITTLE WORK TO ACTUALLY WORK, BUT THIS IS THE GENERAL OUTLINE FOR GETTING
    # A BOXPLOT FROM RPY2

    if not keysInOrder:
        keysInOrder = dict_.keys()
        
    t = TaggedList([])
    for key in keysInOrder:
        t.append(robj.FloatVector(dict_[key]), "X:"+str(key))
        #print key, mean(dict_[key]), median(dict_[key])

    x = robj.r.boxplot(t, names=robj.StrVector(keysInOrder),*args, **kwdargs)
    return x

def superpose (x, above, below, length = 0.08, *args, **keywordargs):
    robj.r.arrows(x, above, x, below, angle = 90, code = 3, length = length, *args, **keywordargs)
    
def barPlotWithErrBars(heights, aboves, belows, labels, *args, **kwdargs):
    kwdargs["names.arg"] = labels
    if not "ylim" in kwdargs:
        kwdargs["ylim"] = robj.FloatVector((min([min(heights), min(aboves), min(belows),0]),
                                            max([max(heights), max(aboves), max(belows)])))

    heights = robj.FloatVector(heights)
    aboves = robj.FloatVector(aboves)
    belows = robj.FloatVector(belows)
    labels = robj.StrVector(labels)

    x = robj.r.barplot(heights, *args, **kwdargs)
    superpose(x, aboves, belows)


def scatterplotMatrix(taggedList, main="", **kwdargs):
    """ Plots a scatterplot matrix, with scatterplots in the upper left and correlation
    values in the lower right.

    >>> t = TaggedList(map(robj.IntVector, [(1,2,3,4,5), (1,3,4,6,6), (6,4,3,2,1)]), ("first", "second", "third"))
    >>> scatterplotMatrix(t)
    """
    robj.r.library("lattice")

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


def linearRegression(independent, dependent):
    fmla = robj.Formula("dep ~ indep")
    env = fmla.environment
    env["dep"] = dependent # y
    env["indep"] = independent # x

    #robj.r.summary(fit).rx("r.squared")
    
    return robj.r.lm(fmla)


def plotWithCor(x, y, xlab="", ylab="", main="", **kwdargs):
    x = robj.FloatVector(x)
    y = robj.FloatVector(y)

    cor = robj.r.cor(x, y, method="spearman")[0]
    
    robj.r.plot(x, y, xlab=xlab, ylab=ylab, main="%s %.3g,n=%d"%(main, cor,len(x)), **kwdargs)
    

def plotSeries(positions, seriesDict, keys, type_="l", **options):
    """ Plots various series on the same axes. Optionally include various plot/line/points
    as lists, one for each series, or as individual items repeated for all the series.

    >>> robj.r.pdf("/tmp/tmp.pdf")
    <...>
    >>> data = {"a":range(0,20, 2)[::-1], "b":range(10,20)}
    >>> plotSeries(range(10), data, data.keys(), type_="p", col=["red", "blue"])
    >>> robj.r("dev.off()") != None
    True
    """

    def _getSeriesOptions(i, options):
        seriesOptions = {}
        for key in options:
            if type(options[key]) == type([]) or type(options[key]) == type(()):
                seriesOptions[key] = options[key][i] # should be some sort of mod len(options[key])
            else:
                seriesOptions[key] = options[key]

        return seriesOptions
    
    robj.r.plot(positions, robj.FloatVector(seriesDict[keys[0]]), type=type_, **_getSeriesOptions(0, options))

    for i, key in enumerate(keys[1:]):
        if type_ == "l":
            fn = robj.r.lines
        elif type_ == "p":
            fn = robj.r.points
        fn(positions, robj.FloatVector(seriesDict[key]), **_getSeriesOptions(i+1, options))


def ecdf(convertToVectors, labels, colors=("red", "blue", "orange", "violet", "green", "brown"),
         xlab="", ylab="cumulative fraction", main="", legendWhere="topleft", **ecdfKwdArgs):
    """ Take a list of lists, convert them to vectors, and plots them sequentially on a CDF """

    #print "MEANS:", main
    #for vector, label in zip(convertToVectors, labels):
    #    print label, numpy.mean(vector)
        
    

    ecdfKwdArgs.update({"verticals":True, "do.points":False, "col.hor":colors[0], "col.vert":colors[0]})

    if not "xlim" in ecdfKwdArgs:
        xlim = robj.FloatVector((min(min(vector) for vector in convertToVectors),
                                 max(max(vector) for vector in convertToVectors)))
        ecdfKwdArgs["xlim"] = xlim

    vectors = [robj.FloatVector(x) for x in convertToVectors]
    robj.r.plot(robj.r.ecdf(vectors[0]), main=main, xlab=xlab, ylab=ylab, **ecdfKwdArgs)

    for i, vector in enumerate(vectors[1:]):
        robj.r.plot(robj.r.ecdf(vector), add=True,
                    **{"verticals":True, "do.points":False, "col.hor":colors[i+1], "col.vert":colors[i+1]})

    labelsWithN = []
    for i, label in enumerate(labels):
        labelsWithN.append(label+" (n=%d)"%len(convertToVectors[i]))
    robj.r.legend(legendWhere, legend=robj.StrVector(labelsWithN), lty=1, lwd=2, col=robj.StrVector(colors), cex=0.7, bg="white")



class GenomicLociPlot(object):
    def __init__(self, width):
        self.width = width
        self.rows = []
        
        # scaling...

    def addRow(self, row):
        assert len(row) == self.width
        self.rows.append(numpy.array(row))

    def plot(self):
        matrix = numpy.array(self.rows)
        matrix = numpy.fliplr(numpy.transpose(matrix)) # eh not sure why these transformations are necessary...
        print matrix
        robj.r.image(matrix)
        
if __name__ == "__main__":
    import doctest
    import random
    random.seed(10)
    #doctest.testmod(optionflags=doctest.ELLIPSIS)

    glp = GenomicLociPlot(100)
    glp.addRow([random.randint(0, 10) for i in range(100)])
    glp.addRow([0]*25 + [random.randint(0, 10) for i in range(25)]+[0]*50)
    glp.addRow([0]*50 + [random.randint(5, 10) for i in range(40)]+[0]*10)

    robj.r.pdf("temp.pdf")
    glp.plot()
    robj.r("dev.off()")
    
#     #d= {"First": robj.IntVector([1,2,3,4,5]), "Second": robj.IntVector([2,5,6,7,8]), "Third": robj.IntVector([7,5,3,5,2])}
#     t = TaggedList([robj.IntVector([1,2,3,4,5]), robj.IntVector([2,5,6,7,8]), robj.IntVector([7,5,3,5,2])],
#                    tags=["First", "Second", "Third"])


#     robj.r.pdf("tmp.pdf")
#     scatterplotMatrix(t)
#     robj.r("dev.off()")
