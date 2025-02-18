import numpy as np

# Definiere Parameter
L = 1.0       # Länge des Stabs
A = 1.0       # Querschnittsfläche
E = 100.0     # Elastizitätsmodul
n_elements = 4

# Diskretisierung
n_nodes = n_elements + 1
node_coords = np.linspace(0, L, n_nodes)

# Elementsteifigkeitsmatrix
k_local = (A * E / (L / n_elements)) * np.array([[1, -1], [-1, 1]])

# Globale Steifigkeitsmatrix
K_global = np.zeros((n_nodes, n_nodes))

# Montage der globalen Steifigkeitsmatrix
for i in range(n_elements):
    K_global[i:i+2, i:i+2] += k_local

# Lastvektor
F = np.zeros(n_nodes)
F[-1] = 10.0  # Last am letzten Knoten

# Randbedingungen anwenden (z.B. fester Anfangspunkt)
K_global[0, :] = 0
K_global[:, 0] = 0
K_global[0, 0] = 1
F[0] = 0

# Lösung des Gleichungssystems
displacements = np.linalg.solve(K_global, F)

# Ergebnisse anzeigen
print("Knotenverschiebungen:", displacements)
