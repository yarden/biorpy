##
## biorpy plotting wrappers
##
## nspies in ze house
##
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

def plotWithCor(x, y, method="spearman", **kwdargs):
    cor = r.cor(x, y, method=method)[0]

    if "main" in kwdargs:
        main = kwdargs.pop("main")
    else:
        main = ""
        
    r.plot(x, y, main="{} rs = {}".format(main, cor), **kwdargs)

def errbars(x=None, y=None, x_lower=None, x_upper=None, y_lower=None, y_upper=None, length=0.08, *args, **kwdargs):
    if y is not None and  x_lower is not None  and x_upper is not None:
        r.arrows(x_lower, y, x_upper, y, angle = 90, code = 3, length = length, *args, **kwdargs)
    elif x is not None and y_lower is not None and y_upper is not None:
        r.arrows(x, y_lower, x, y_upper, angle = 90, code = 3, length = length, *args, **kwdargs)
    else:
        raise Exception("must define either (y, x_lower, x_upper) or (x, y_lower, y_upper)")
            
    

def ecdf(vectors, labels, colors=["red", "blue", "orange", "violet", "green", "brown"],
         xlab="", ylab="cumulative fraction", main="", legendWhere="topleft", **ecdfKwdArgs):
    """ Take a list of lists, convert them to vectors, and plots them sequentially on a CDF """

    #print "MEANS:", main
    #for vector, label in zip(convertToVectors, labels):
    #    print label, numpy.mean(vector)
        
    

    ecdfKwdArgs.update({"verticals":True, "do.points":False, "col.hor":colors[0], "col.vert":colors[0]})

    if not "xlim" in ecdfKwdArgs:
        xlim = [min(min(vector) for vector in vectors),
                max(max(vector) for vector in vectors)]
        ecdfKwdArgs["xlim"] = xlim

    r.plot(r.ecdf(vectors[0]), main=main, xlab=xlab, ylab=ylab, **ecdfKwdArgs)

    for i, vector in enumerate(vectors[1:]):
        r.plot(r.ecdf(vector), add=True,
                    **{"verticals":True, "do.points":False, "col.hor":colors[i+1], "col.vert":colors[i+1]})

    labelsWithN = []
    for i, label in enumerate(labels):
        labelsWithN.append(label+" (n=%d)"%len(vectors[i]))
    r.legend(legendWhere, legend=labelsWithN, lty=1, lwd=2, col=colors, cex=0.7, bg="white")



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