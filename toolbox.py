import sympy as smp
from IPython import display
from IPython.display import Math
print("toolbox.py loaded")

def display_smp_formula(formula, description = r"f"):
    l = r"$ " + description + " = " + smp.latex(formula)
    display(Math(l))