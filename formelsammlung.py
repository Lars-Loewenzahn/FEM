from sympy import symbols, Function, Integral, Derivative, Eq
from sympy.vector import CoordSys3D, Del

# Koordinatensystem und Variablen definieren
R = CoordSys3D('R')
t = symbols('t')
Phi = Function('Phi')(t, R.x, R.y, R.z)
pi = Function('pi')(t, R.x, R.y, R.z)
q_dot = Function('q_dot')(t, R.x, R.y, R.z)
v = Function('v')(t, R.x, R.y, R.z)
u = Function('u')(t, R.x, R.y, R.z)
n = symbols('n')  # Flächennormalenvektor

# Hauptgleichung
depsilon_dt = Eq(
    Derivative(Integral(Phi, (R.x, -oo, oo), (R.y, -oo, oo), (R.z, -oo, oo)), t),
    symbols('F_Kon') + symbols('F_Dif') + symbols('Pi')
)

# Flussdefinitionen
F_Kon = Eq(
    symbols('F_Kon'),
    -Integral(Phi * (v - u).dot(n), (R.x, -oo, oo), (R.y, -oo, oo), (R.z, -oo, oo))
)

F_Dif = Eq(
    symbols('F_Dif'),
    -Integral(q_dot.dot(n), (R.x, -oo, oo), (R.y, -oo, oo), (R.z, -oo, oo))
)

Pi = Eq(
    symbols('Pi'),
    Integral(pi, (R.x, -oo, oo), (R.y, -oo, oo), (R.z, -oo, oo))
)

# Kombinierte Transportgleichung
transport_eq = Eq(
    Derivative(Integral(Phi, (R.x, -oo, oo), (R.y, -oo, oo), (R.z, -oo, oo)), t),
    -Integral((Phi*(v-u) + q_dot).dot(n), (R.x, -oo, oo), (R.y, -oo, oo), (R.z, -oo, oo))
    + Integral(pi, (R.x, -oo, oo), (R.y, -oo, oo), (R.z, -oo, oo))
)
