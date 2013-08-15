from rpy2.robjects import Formula

def getSimpleFormula(x, y):
    formula = Formula("y ~ x")
    formula.env["x"] = x
    formula.env["y"] = y
    
    return formula