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

def errbars(x=None, y=None, x_lower=None, x_upper=None, y_lower=None, y_upper=None, length=0.08, *args, **kwdargs):
    if y is not None and  x_lower is not None  and x_upper is not None:
        r.arrows(x_lower, y, x_upper, y, angle = 90, code = 3, length = length, *args, **kwdargs)
    elif x is not None and y_lower is not None and y_upper is not None:
        r.arrows(x, y_lower, x, y_upper, angle = 90, code = 3, length = length, *args, **kwdargs)
    else:
        raise Exception("must define either (y, x_lower, x_upper) or (x, y_lower, y_upper)")
            
    
