import numpy as np

class TrussElement:
    def __init__(self, E, A, startpoint, endpoint):
        self.E = E
        self.A = A
        self.x1 = startpoint[0]
        self.y1 = startpoint[1]
        self.x2 = endpoint[0]
        self.y2 = endpoint[1]
        self.L = np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
        self.theta = np.arctan2((self.y2 - self.y1), (self.x2 - self.x1))
        
def element_stiffness_matrix(E, A, L):
    # Lokale Steifigkeitsmatrix für ein 2D-Stabelement
    k_local = (E * A / L) * np.array([
        [1, 0, -1, 0],
        [0, 0, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 0]
    ])
    return k_local

def transformation_matrix(theta):
    # Transformationsmatrix für den Winkel theta
    c = np.cos(theta)
    s = np.sin(theta)
    T = np.array([
        [c, s, 0, 0],
        [-s, c, 0, 0],
        [0, 0, c, s],
        [0, 0, -s, c]
    ])
    return T

def global_stiffness_matrix(E, A, trusses):
    # Bestimme die Anzahl der Knoten
    all_nodes = set()
    for truss in trusses:
        all_nodes.add((truss.x1, truss.y1))
        all_nodes.add((truss.x2, truss.y2))
    num_nodes = len(all_nodes)
    
    # Erstelle eine globale Steifigkeitsmatrix der richtigen Größe
    K_global = np.zeros((num_nodes * 2, num_nodes * 2))
    
    # Mapping von Knotenkordinaten zu Indizes
    nodes = list(all_nodes)
    node_map = {node: i for i, node in enumerate(nodes)}
    
    for truss in trusses:
        # Berechnung der Länge des Elements
        L = truss.L
        # Berechnung des Winkels des Elements
        theta = truss.theta
        
        # Lokale Steifigkeitsmatrix
        k_local = element_stiffness_matrix(truss.E, truss.A, L)
        
        # Transformationsmatrix
        T = transformation_matrix(theta)
        
        # Transformation der lokalen Steifigkeitsmatrix in die globalen Koordinaten
        k_global = T.T @ k_local @ T
        
        # Finde die Indizes der Knoten in der globalen Matrix
        node1 = (truss.x1, truss.y1)
        node2 = (truss.x2, truss.y2)
        index1 = node_map[node1] * 2
        index2 = node_map[node2] * 2
        
        # Addiere die lokalen Steifigkeitswerte zur globalen Matrix
        indices = [index1, index1 + 1, index2, index2 + 1]
        for i in range(4):
            for j in range(4):
                K_global[indices[i], indices[j]] += k_global[i, j]
    
    return K_global

def apply_boundary_conditions(K_global, F_global, fixed_nodes):
    # Wende die Randbedingungen an, indem die entsprechenden Zeilen und Spalten null gesetzt werden
    for node in fixed_nodes:
        index = node * 2
        # Setze die Zeile und Spalte für Verschiebung in x-Richtung
        K_global[index, :] = 0
        K_global[:, index] = 0
        K_global[index, index] = 1
        F_global[index] = 0
        
        # Setze die Zeile und Spalte für Verschiebung in y-Richtung
        index += 1
        K_global[index, :] = 0
        K_global[:, index] = 0
        K_global[index, index] = 1
        F_global[index] = 0
    
    return K_global, F_global

def create_force_vector(num_nodes, forces):
    # Erstelle den Kraftvektor mit den angegebenen Kräften an den entsprechenden Knoten
    F_global = np.zeros(num_nodes * 2)
    for (node, fx, fy) in forces:
        index = node * 2
        F_global[index] = fx
        F_global[index + 1] = fy
    return F_global

# Beispielparameter
E = 210e9  # Elastizitätsmodul in N/m^2 (Stahl)
A = 0.01   # Querschnittsfläche in m^2
k1 = (0, 0)  # Koordinaten des ersten Knotens
k2 = (0, 2)  # Koordinaten des zweiten Knotens
k3 = (3, 1)  # Koordinaten des dritten Knotens

truss1 = TrussElement(E, A, k2, k3)
truss2 = TrussElement(E, A, k1, k3)
trusses = [truss1, truss2]

# Globale Steifigkeitsmatrix berechnen
K_global = global_stiffness_matrix(E, A, trusses)

# Erstelle den Kraftvektor
forces = [(1, 1000, 0)]  # 1000 N in x-Richtung am Knoten 1
num_nodes = 3  # Anzahl der Knoten
oF_global = create_force_vector(num_nodes, forces)

# Wende Randbedingungen an (angenommen, Knoten 0 ist fest)
fixed_nodes = [0]
K_global, F_global = apply_boundary_conditions(K_global, F_global, fixed_nodes)

print("Globale Steifigkeitsmatrix mit Randbedingungen:\n", K_global)
print("Kraftvektor:\n", F_global)
