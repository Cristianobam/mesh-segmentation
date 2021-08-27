def millimeter2inches(x:float) -> float:
    return x / 25.4

def kilogram2pound(x:float) -> float:
    return x * 2.205

def roundcap(x, base=1, prec=1):
    return round(base * round(float(x)/base),prec)