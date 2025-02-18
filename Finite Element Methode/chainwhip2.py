import pygame
import Box2D
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)

# Initialize Pygame
pygame.init()

# Constants
PPM = 20.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480

# Pygame setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('Chain Simulation')
clock = pygame.time.Clock()

# Box2D world setup
world = world(gravity=(0, -10), doSleep=True)

# Create ground body
ground_body = world.CreateStaticBody(position=(0, 1))
ground_box = ground_body.CreatePolygonFixture(box=(50, 1), density=0, friction=0.3)

# Create chain
chain_links = []
prev_body = ground_body
for i in range(10):
    body = world.CreateDynamicBody(position=(5, 25 - i))
    box = body.CreatePolygonFixture(box=(0.5, 0.125), density=1, friction=0.3)
    joint = world.CreateRevoluteJoint(bodyA=prev_body, bodyB=body,
                                      anchor=(5, 25 - i))
    chain_links.append(body)
    prev_body = body

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear screen
    screen.fill((0, 0, 0))

    # Draw chain
    for body in chain_links:
        for fixture in body.fixtures:
            shape = fixture.shape
            vertices = [(body.transform * v) * PPM for v in shape.vertices]
            vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
            pygame.draw.polygon(screen, (255, 255, 255), vertices)

    # Update the screen
    pygame.display.flip()

    # Step the physics world
    world.Step(TIME_STEP, 10, 10)

    # Limit FPS
    clock.tick(TARGET_FPS)

pygame.quit()