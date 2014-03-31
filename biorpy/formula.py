from rpy2.robjects import Formula

def getSimpleFormula(x, y):
    formula = Formula("y ~ x")
    formula.environment["x"] = x
    formula.environment["y"] = y
    
    return formula