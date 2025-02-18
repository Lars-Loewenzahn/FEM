import math
import pygame

class Node:
    def __init__(self, x, y, fixed=False):
        self.x = x
        self.y = y
        self.force_x = 0.0  # Kraft in x-Richtung
        self.force_y = 0.0  # Kraft in y-Richtung
        self.displacement_x = 0.0  # Verschiebung in x-Richtung
        self.displacement_y = 0.0  # Verschiebung in y-Richtung
        self.fixed = fixed  # Gibt an, ob der Knoten fixiert ist

    def draw(self, screen):
        color = (0, 0, 0) if self.fixed else (0, 255, 0)
        pygame.draw.circle(screen, color, (self.x, self.y), 5)

    def __repr__(self):
        return f"Node(x={self.x}, y={self.y})"


class Truss:
    def __init__(self, node1, node2, e_modulus, area, is_road=False):
        self.node1 = node1
        self.node2 = node2
        self.e_modulus = e_modulus  # E-Modul (Elastizitätsmodul)
        self.area = area  # Querschnittsfläche
        self.length = self.calculate_length()
        self.angle = self.calculate_angle()
        self.is_road = is_road  # Gibt an, ob der Truss als Straße fungiert
        self.stiffness_matrix = self.calculate_stiffness_matrix()

    def calculate_length(self):
        dx = self.node2.x - self.node1.x
        dy = self.node2.y - self.node1.y
        return math.sqrt(dx**2 + dy**2)

    def calculate_angle(self):
        dx = self.node2.x - self.node1.x
        dy = self.node2.y - self.node1.y
        return math.atan2(dy, dx)

    def calculate_stiffness_matrix(self):
        # Berechnung der Steifigkeitsmatrix für das Trusselement
        c = math.cos(self.angle)
        s = math.sin(self.angle)
        length = self.length
        e_a_l = (self.e_modulus * self.area) / length
        return [
            [c * c * e_a_l, c * s * e_a_l, -c * c * e_a_l, -c * s * e_a_l],
            [c * s * e_a_l, s * s * e_a_l, -c * s * e_a_l, -s * s * e_a_l],
            [-c * c * e_a_l, -c * s * e_a_l, c * c * e_a_l, c * s * e_a_l],
            [-c * s * e_a_l, -s * s * e_a_l, c * s * e_a_l, s * s * e_a_l]
        ]

    def draw(self, screen):
        color = (20, 20, 20) if self.is_road else (100, 100, 100)
        pygame.draw.line(screen, color, (self.node1.x, self.node1.y), (self.node2.x, self.node2.y), 5)

    def __repr__(self):
        return f"Truss(Node1={self.node1}, Node2={self.node2}, Length={self.length:.2f}, IsRoad={self.is_road})"

# Beispiel zur Verwendung
nodes = [
    Node(100, 300, fixed=True),
    Node(700, 300, fixed=True),
]
trusses = []

# Pygame Darstellung
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Truss Simulation")
clock = pygame.time.Clock()

drawing = False
current_node = None
is_road = False

running = True
while running:
    screen.fill((100, 100, 255))
    
    # Nodes zeichnen
    for node in nodes:
        node.draw(screen)
    
    # Trusses zeichnen
    for truss in trusses:
        truss.draw(screen)
    
    pygame.display.flip()
    clock.tick(60)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if not drawing:
                # Start eines neuen Trusses
                for node in nodes:
                    if math.hypot(node.x - x, node.y - y) < 10:
                        current_node = node
                        drawing = True
                        break
            else:
                # Ende eines Trusses
                for node in nodes:
                    if math.hypot(node.x - x, node.y - y) < 10:
                        trusses.append(Truss(current_node, node, e_modulus=210e9, area=0.01, is_road=is_road))
                        break
                else:
                    new_node = Node(x, y)
                    nodes.append(new_node)
                    trusses.append(Truss(current_node, new_node, e_modulus=210e9, area=0.01, is_road=is_road))
                current_node = None
                drawing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                # Wechsel zwischen Straße oder nicht
                is_road = not is_road

pygame.quit()
