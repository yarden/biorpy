##
## biorpy plotting wrappers
##
## nspies in ze house
##
import pandas
import rpy2

from rpy2.robjects import r
import rpy2.robjects.numpy2ri
import pandas.rpy.common as com
rpy2.robjects.numpy2ri.activate()

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
    
