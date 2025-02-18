import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation

# Number of nodes
nodes = 2

t, g = smp.symbols('t g')
masses = [smp.symbols(f'm{i}') for i in range(1, nodes + 1)]
lengths = [smp.symbols(f'L{i}') for i in range(1, nodes + 1)]
thetas = [smp.Function(f'theta{i}')(t) for i in range(1, nodes + 1)]

thetas_d = [smp.diff(theta, t) for theta in thetas]
thetas_dd = [smp.diff(theta_d, t) for theta_d in thetas_d]

# Kinetic and potential energy
T = sum(1/2 * m * (smp.diff(length * smp.sin(theta), t)**2 + smp.diff(length * smp.cos(theta), t)**2) 
          for m, length, theta in zip(masses, lengths, thetas))
V = sum(m * g * length * (1 - smp.cos(theta)) for m, length, theta in zip(masses, lengths, thetas))

# Lagrangian
L = T - V

# Lagrange's equations
LE = [smp.diff(L, theta) - smp.diff(smp.diff(L, smp.diff(theta, t)), t) for theta in thetas]
sols = smp.solve(LE, thetas_dd, simplify=False, rational=False)

# Numerical integration setup
initial_conditions = [np.pi / 4, 0] * nodes  # Initial angles and angular velocities

def dSdt(S, t, g, masses, lengths):
    thetas_vals = S[:nodes]
    thetas_d_vals = S[nodes:]
    derivatives = []
    for i in range(nodes):
        derivatives.append(thetas_d_vals[i])
        dz = sols[thetas_dd[i]].subs({
            **{thetas[j]: thetas_vals[j] for j in range(nodes)},
            **{thetas_d[j]: thetas_d_vals[j] for j in range(nodes)},
            **{masses[j]: masses[j] for j in range(nodes)},
            **{lengths[j]: lengths[j] for j in range(nodes)},
            g: 9.81
        })
        dz_numeric = dz.evalf()  # Evaluate the expression to a numerical value
        derivatives.append(float(dz_numeric))
    return derivatives

# Time array for simulation
end_time = 10
FPS = 30
t = np.linspace(0, end_time, end_time * FPS)

# Integrate the equations of motion
ans = odeint(dSdt, y0=initial_conditions, t=t, args=(9.81, [1]*nodes, [1]*nodes))

# Visualization
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ln, = plt.plot([], [], 'ro-', lw=2)

# Animation function
positions = []
for i in range(nodes):
    x = np.cumsum([length * np.sin(ans[:, j]) for j, length in enumerate([1]*nodes)], axis=0)
    y = np.cumsum([-length * np.cos(ans[:, j]) for j, length in enumerate([1]*nodes)], axis=0)
    positions.append((x, y))

x_data, y_data = zip(*positions)

def init():
    ln.set_data([], [])
    return ln,

def update(frame):
    ln.set_data(x_data[0][frame], y_data[0][frame])
    return ln,

ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True)
plt.show()