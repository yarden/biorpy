##
## R plotting utilities
##
import pandas
import numpy
##
## Wrapper to Rpy2.
##
## Transparently convert Pandas objects (DataFrames, Series) into
## corresponding R objects.  Also handle ordinary numpy arrays/matrices.
##
## 

import scipy
import rpy2

from rpy2.robjects import r
import rpy2.robjects as robj
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.lib import grid
from rpy2.robjects.lib import ggplot2
rpy2.robjects.numpy2ri.activate()
from collections import OrderedDict
py2ri_orig = rpy2.robjects.conversion.py2ri

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
    # Use the index to label the rows
    rDataFrame.rownames = robj.StrVector(pDataframe.index)

    return rDataFrame

#py2ri = pandas_data_frame_to_rpy2_data_frame
py2ri = convert_pandas_to_r

