from rpy2 import robjects
import operator

from rpy2.robjects import numpy2ri
import pandas


def rx(name):
    """ extracts a value by name from an R object """
    def f(x):
        return x.rx(name)
    return f

def item(i):
    """ Short name for operator.itemgetter() """
    return operator.itemgetter(i)


class Handler(object):
    def __init__(self, name, defaults=None, outputs=None):
        """
        :param name: name of the R function
        :param defaults: a dictionary of default arguments to the function
        :param outputs: a dictionary whose values are lists of functions used to 
        extract values from the return R value. For example: {"p.value":[rx("p.value"), item(0), item(0)]}
        """
        self.name = name

        if not defaults and not outputs:
            raise Exception("Need to define at least one of defaults or outputs")

        self.defaults = defaults if defaults else {}
        self.outputs = outputs if outputs else {}

        # may want some extra error checking here
        self._robject = robjects.r[self.name]

    def __call__(self, *args, **kwdargs):
        defaults = self.defaults.copy()
        defaults.update(kwdargs)

        print "calling:", self.name, args, defaults
        
        rval = robjects.r[self.name](*args, **defaults)

        result = {"rval":rval}

        for output in self.outputs:
            print self.outputs[output]
            result[output] = reduce(lambda x, f: f(x), self.outputs[output], rval)

        return result

class BetteR(object):
    # this in theory could also be a subclass of rpy2.robjects.R

    def __init__(self):
        self.aliases = {"devoff":"dev.off"}

        self._handlers = {}
        self.addHandler_(Handler("wilcox.test", 
            outputs={"p.value":[rx("p.value"), item(0), item(0)]}
            )
        )

        self.addHandler_(Handler("plot", 
            defaults={"xlab":"", "ylab":"", "main":"plot"}
            )
        )

    def addHandler_(self, handler):
        self._handlers[handler.name] = handler


    def __getattribute__(self, attr):
        # cribbed directly from rpy2.robjects.R

        try:
            return super(BetteR, self).__getattribute__(attr)
        except AttributeError as ae:
            orig_ae = ae

        try:
            return self.__getitem__(attr)
        except LookupError as le:
            raise orig_ae

    def __getitem__(self, attr):
        if attr in self.aliases:
            attr = self.aliases[attr]

        if attr in self._handlers:
            return self._handlers[attr]
        else:
            return robjects.r[attr]


def convertToR(obj):
    """
    Convert Pandas/Numpy objects to R objects.

    If the input object is a Pandas DataFrame, convert it to
    an R DataFrame.  If it's a Series, treat it like a vector/numpy
    array. 
    """
    if isinstance(obj, pandas.core.frame.DataFrame):
        return pandasDataFrameToRPy2DataFrame(obj)
    elif isinstance(obj, pandas.Series):
        return obj
    elif isinstance(obj, numpy.ndarray):
        return numpy2ri.numpy2ri(obj)
    elif isinstance(obj, )

    return obj

def pandasDataFrameToRPy2DataFrame(pDataframe):
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


if __name__ == '__main__':
    r = BetteR()

    result = r["wilcox.test"](robjects.FloatVector(range(5)), robjects.FloatVector([1,2,55,3,6]))
    print result["p.value"]

    r.plot(robjects.FloatVector(range(5)), robjects.FloatVector([1,2,55,3,6]))

    for i in range(100000):
        for j in range(1000):
            pass