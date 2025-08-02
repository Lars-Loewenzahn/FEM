import math
import numpy as np  # falls du es nutzen willst
import sympy as sp
from scipy.integrate import quad

# === Allgemeine Konstanten ===
g = 9.81  # [m/s²] Erdbeschleunigung

# === 1. Wellengleichung (Bernoulli und d'Alembert) ===
def welle_bernoulli_loesung(W_x, p_t):
    return W_x * p_t

def welle_dalembert(w0, v0, x, t, c):
    from scipy.integrate import quad
    integral, _ = quad(lambda ξ: v0(ξ), x - c*t, x + c*t)
    return 0.5 * (w0(x - c*t) + w0(x + c*t) + (1/c) * integral)

# === Rayleigh-Quotient - Euler Balken===
def rayleigh_quotient(W_tilde, W_tilde_pp, EI, mu, l):
    """
    W_tilde: Eigenform-Näherung W̃₁(x) (Array)
    W_tilde_pp: zweite Ableitung davon W̃₁''(x) (Array)
    EI: Biegesteifigkeit
    mu: Massenbelegung (konstant oder Feld)
    l: Länge des Balkens
    """
    dx = l / len(W_tilde)
    numerator = np.trapz(EI * W_tilde_pp**2, dx=dx)
    denominator = np.trapz(mu * W_tilde**2, dx=dx)
    return numerator / denominator  # ergibt ω̃₁²

# === 3. Hydromechanik ===
def bernoulli(p1, v1, z1, p2, v2, z2, rho):
    lhs = 0.5 * rho * v1**2 + rho * g * z1 + p1
    rhs = 0.5 * rho * v2**2 + rho * g * z2 + p2
    return lhs, rhs

def volumenstrom(v, A):
    return v * A

def impulsbilanz(rho, V_dot, v1, v2):
    return rho * V_dot * (v2 - v1)

# === 4. Prinzip von Hamilton (Beispiel: Längsschwingung) ===
def T_laengs(mu, u_dot, l):
    return 0.5 * np.trapz(mu * u_dot**2, dx=l/len(u_dot))

def U_laengs(EA, u_prime, l):
    return 0.5 * np.trapz(EA * u_prime**2, dx=l/len(u_prime))








#########################################################################
# Formelsammlung.py — Formelblatt 1
#########################################################################


# === Symbolische Variablen ===
x, t, c, ω = sp.symbols('x t c ω')
W = sp.Function('W')(x)
p = sp.Function('p')(t)

# === 1. Eindimensionale Wellengleichung ===
# ẅ(x, t) = c² w''(x, t)
def wellengleichung_symbolisch():
    w = W * p
    d2w_dt2 = sp.diff(w, t, 2)
    d2w_dx2 = sp.diff(w, x, 2)
    return sp.Eq(d2w_dt2, c**2 * d2w_dx2)

# === 2. Masse pro Länge ===
def massenbelegung(Ax, rho_x):
    return Ax * rho_x  # µ(x) = A(x) * ρ(x)

# === 3. Längsschwingung von Stäben ===
def laengsschwingung(mu, EA, u_x, q_xt):
    u = sp.Function('u')(x, t)
    return sp.Eq(mu * sp.diff(u, t, 2) - sp.diff(EA * sp.diff(u, x), x), q_xt)

# === 4. Torsionsschwingung ===
def torsionsschwingung(rho, G, Ip, mT_xt):
    theta = sp.Function('θ')(x, t)
    return sp.Eq(rho * Ip * sp.diff(theta, t, 2) - G * Ip * sp.diff(theta, x, 2), mT_xt)

# === 5. Querschwingung von Saiten ===
def saite_schwingung(mu, T, q_xt):
    w = sp.Function('w')(x, t)
    return sp.Eq(mu * sp.diff(w, t, 2) - T * sp.diff(w, x, 2), q_xt)

# === 6. Wellenausbreitungsgeschwindigkeiten ===
def c_laengs(E, rho): return np.sqrt(E / rho)
def c_torsion(G, rho): return np.sqrt(G / rho)
def c_saite(T, mu): return np.sqrt(T / mu)

# === 7. Produktansatz für Bernoullische Lösung ===
def bernoulli_loesung(Wx, pt): return Wx * pt

# === 8. Eigenfrequenzen aus RWP ===
def randwertproblem_symbolisch():
    Wpp = sp.diff(W, x, 2)
    eq = sp.Eq(Wpp + (ω/c)**2 * W, 0)
    return eq  # W'' + (ω/c)² W = 0

# === 9. Orthogonalität der Eigenfunktionen ===
def orthogonalitaet(Wi, Wj):
    return sp.integrate(Wi * Wj, (x, 0, sp.symbols('l')))








#########################################################################
# Formelsammlung.py — Formelblatt 2
#########################################################################


# === d’Alembertsche Lösung der Wellengleichung ===
# ẅ = c² w''  → Lösung: w(x, t) = f₁(x - ct) + f₂(x + ct)
def d_alembert_loesung(f1, f2, x, t, c):
    """Gibt die d'Alembertsche Lösung als Summe zweier Funktionsausbreitungen zurück"""
    return f1(x - c * t) + f2(x + c * t)

# === Anfangsbedingungen einarbeiten ===
# w(x, 0) = w₀(x),  ẇ(x, 0) = v₀(x)
def d_alembert_angepasst(w0, v0, x, t, c):
    """Lösung mit Anfangsbedingungen w0(x) und v0(x)"""
    integral, _ = quad(lambda ξ: v0(ξ), x - c*t, x + c*t)
    return 0.5 * (w0(x - c * t) + w0(x + c * t) + (1 / c) * integral)

# === Zwangsschwingung einer Saite ===
# ẅ - c²w'' = (1/μ)·Q(x)·sin(Ωt)
def zwangsschwingung_ansatz(Qx, mu, Omega, x, t):
    """Zwangsschwingung einer Saite unter harmonischer Streckenlast"""
    # Ansatz: w(x,t) = W(x)·sin(Ωt), führt zu: -Ω²μW = c²W''
    # → Separiertes Problem, das W(x) lösen muss
    Wx = Qx / (mu * (Omega**2))  # einfache Näherung ohne Randbedingungen
    return Wx * np.sin(Omega * t)








#########################################################################
# Formelsammlung.py — Formelblatt 3
#########################################################################
# === Symbolische Variablen für freie Schwingungen ===
x, t, ω, μ, EI = sp.symbols('x t ω μ EI')
W = sp.Function('W')(x)
p = sp.Function('p')(t)
w = W * p

# === Feldgleichung für konstante Balkenparameter ===
# μ·ẅ(x,t) + EI·w⁽⁴⁾(x,t) = 0
def feldgleichung_eb_balken():
    d2w_dt2 = sp.diff(w, t, 2)
    d4w_dx4 = sp.diff(w, x, 4)
    return sp.Eq(μ * d2w_dt2 + EI * d4w_dx4, 0)

# === Produktansatz einsetzen → Entkopplung in zwei Gleichungen ===
# p̈(t) + ω²·p(t) = 0
def zeitanteil_dgl():
    return sp.Eq(sp.diff(p, t, 2) + ω**2 * p, 0)

# === Raumanteil (Randwertproblem) ===
# W⁽⁴⁾(x) − (μω²/EI)·W(x) = 0
def raumanteil_dgl():
    return sp.Eq(sp.diff(W, x, 4) - (μ * ω**2 / EI) * W, 0)

# === Randbedingungen: Beispiele ===
def randbedingungen_konsole_frei():
    return {
        "w(0)": 0,
        "w'(0)": 0,
        "w''(l)": 0,
        "w'''(l)": 0
    }

def randbedingungen_frei_frei():
    return {
        "w''(0)": 0,
        "w''(l)": 0,
        "w'''(0)": 0,
        "w'''(l)": 0
    }

# === Gesamtlösung: w(x,t) = Σ Wᵢ(x)·pᵢ(t) mit Eigenwertproblem
def allgemeine_loesung():
    Wi = sp.Function('W')(x)
    pi = sp.Function('p')(t)
    return sp.Sum(Wi * pi, (sp.symbols('i'), 1, sp.oo))








#########################################################################
# Formelsammlung.py — Formelblatt 4
#########################################################################

# === Symbolische Definitionen ===
x, t, l = sp.symbols('x t l')
u = sp.Function('u')(x, t)
theta = sp.Function('theta')(x, t)
w = sp.Function('w')(x, t)

# === Prinzip von Hamilton ===
# ∫_{t0}^{t1} L dt + ∫_{t0}^{t1} δW dt = 0
# L = T - U
def prinzip_von_hamilton(L, delta_W):
    t0, t1 = sp.symbols('t0 t1')
    return sp.Eq(sp.integrate(L, (t, t0, t1)) + sp.integrate(delta_W, (t, t0, t1)), 0)

# === 1. Längsschwingungen (Stab) ===
def laengsschwingung_hamilton(mu_x, EA_x, F_t):
    T = (1/2) * sp.integrate(mu_x * sp.diff(u, t)**2, (x, 0, l))
    U = (1/2) * sp.integrate(EA_x * sp.diff(u, x)**2, (x, 0, l))
    delta_W = F_t * sp.Function('delta_u')(l, t)
    return T, U, delta_W

# === 2. Torsionsschwingungen (Stab) ===
def torsionsschwingung_hamilton(rho, Ip, G, F_t):
    T = (1/2) * sp.integrate(rho * Ip * sp.diff(theta, t)**2, (x, 0, l))
    U = (1/2) * sp.integrate(G * Ip * sp.diff(theta, x)**2, (x, 0, l))
    delta_W = F_t * sp.Function('delta_theta')(l, t)
    return T, U, delta_W

# === 3. Biegeschwingungen (Balken) ===
def biegeschwingung_hamilton(mu_x, EI_x, F_t, q_xt, c_x, d_x):
    T = (1/2) * sp.integrate(mu_x * sp.diff(w, t)**2, (x, 0, l))
    U = (1/2) * sp.integrate(
        EI_x * sp.diff(w, x, 2)**2 +
        F_t * sp.diff(w, x)**2 +
        c_x * w**2,
        (x, 0, l)
    )
    delta_w = sp.Function('delta_w')(x, t)
    delta_W = sp.integrate(q_xt * delta_w - d_x * sp.diff(w, t) * delta_w, (x, 0, l))
    return T, U, delta_W









#########################################################################
# Formelsammlung.py — Formelblatt 5
#########################################################################

# === Harmonisch erzwungene Schwingung (cos Ωt) ===
def erzwungene_schwingung_ansatz(Wx, Omega, t):
    """
    Lösung der Form: w(x,t) = W(x) · cos(Ω·t)
    """
    return Wx * np.cos(Omega * t)

# === Randwertproblem resultiert aus Einsetzen in PDE ===
# Diese Funktion gibt dir symbolisch die RWP-Gleichung zurück
import sympy as sp
def rwg_erzwungene_schwingung(mu, EI, W, Omega):
    x = sp.symbols('x')
    return sp.Eq(EI * sp.diff(W, x, 4) - mu * Omega**2 * W, 0)







#########################################################################
# Formelsammlung.py — Formelblatt 6
#########################################################################

# === Bernoulli-Gleichung (stationär, ideale Flüssigkeit) ===
def bernoulli(p1, v1, z1, p2, v2, z2, rho, g=9.81):
    """
    Bernoulli-Gleichung zwischen zwei Punkten entlang eines Stromfadens.
    Rückgabe: True, wenn Gleichung erfüllt ist
    """
    lhs = 0.5 * rho * v1**2 + rho * g * z1 + p1
    rhs = 0.5 * rho * v2**2 + rho * g * z2 + p2
    return np.isclose(lhs, rhs), lhs, rhs

# === Kontinuitätsgleichung ===
def volumenstrom_konstant(v1, A1, v2, A2):
    """
    Kontinuitätsgleichung: V̇ = v₁·A₁ = v₂·A₂
    Rückgabe: True, wenn erfüllt
    """
    V1 = v1 * A1
    V2 = v2 * A2
    return np.isclose(V1, V2), V1, V2

# === Impulssatz für stationäre Strömung ===
def impulssatz_stationaer(rho, V_dot, v1_vec, v2_vec):
    """
    Impulssatz: F = ρ·V̇·(v₂ − v₁), Vektoren als Arrays übergeben
    """
    v1 = np.array(v1_vec)
    v2 = np.array(v2_vec)
    F = rho * V_dot * (v2 - v1)
    return F


def bernoulli_symbolisch():
    p1, v1, z1, p2, v2, z2, rho, g = sp.symbols('p1 v1 z1 p2 v2 z2 ρ g')
    eq = sp.Eq(0.5*rho*v1**2 + rho*g*z1 + p1, 0.5*rho*v2**2 + rho*g*z2 + p2)
    return eq