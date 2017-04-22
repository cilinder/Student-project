#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import pygame
from pygame.locals import *

import Box2D
from Box2D import *


# --- constants ---
# Box2D deals with meters, but we want to display pixels,
# so define a conversion factor:
PPM = 20.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
SCREEN_COLOR = (0,0,255,0)
BLACK = (0,0,0,0)



# --- pygame setup ---
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('Simple pygame example')
clock = pygame.time.Clock()


world = b2World(gravity=(0,0), doSleep=True)  

boundingBox = world.CreateStaticBody(position=(0, 20))
boundingBox.CreateEdgeChain(
    [(5, 5),
     (7, 5),
     (7, 7),
     (5, 7),
     (5, 5)]
)
"""boundingBox.CreateEdgeChain(
    [(-20, -20),
     (-20, 20),
     (20, 20),
     (20, -20),
     (-20, -20)]
)
"""



# Create a dynamic body at (0, 4)
boat = world.CreateDynamicBody(position=(10, 10))

# And add a box fixture onto it (with a nonzero density, so it will move)
box = boat.CreatePolygonFixture(box=(2, 1), density=1, friction=0.3)


# Prepare for simulation. Typically we use a time step of 1/60 of a second
# (60Hz) and 6 velocity/2 position iterations. This provides a high quality
# simulation in most game scenarios.
timeStep = 1.0 / 60
vel_iters, pos_iters = 6, 2


colors = {
        'boat' : BLACK,
}

running = True
while running:

    # Check the event que
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            # The user closed the window or pressed escape
            running = False

    # This is how you can check for continious key pressing
    keys = pygame.key.get_pressed()
    if keys[K_w]:
        print('w', end='')
        sys.stdout.flush()
        f = boat.GetWorldVector(localVector=(0,20))
        p = boat.GetWorldPoint(localPoint=(1,0))
        boat.ApplyForce(f, p, True)


    # Draw the world
    screen.fill(SCREEN_COLOR)

    pygame.draw.circle(screen, BLACK, (200,200), 2) 
    pygame.draw.line(screen, BLACK, (100, 100), (150, 100))


    # First draw the bounding box
    # It consists of pairs of coordinates representing the edges of the box
    # So we draw a line for each of the pairs
    # But first we have to transform the coordinates to on screen coordinates
    for fixture in boundingBox.fixtures:
        shape = fixture.shape
        v0 = shape.vertices[0]
        v1 = shape.vertices[1]
        #print(v0, v1)
        v0, v1 = [(boundingBox.transform * v) * PPM for v in (v0,v1)]
        #print(v0, v1)
        v0[1] = SCREEN_HEIGHT - v0[1]
        v1[1] = SCREEN_HEIGHT - v1[1]
        #print(v0, v1)
        
        pygame.draw.line(screen, colors['boat'], v0, v1, 2)

    #print('oi!')

    # Next draw the boat
    vertices = [(boat.transform * v) * PPM for v in boat.fixtures[0].shape.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]

    pygame.draw.polygon(screen, colors['boat'], vertices)

    world.Step(TIME_STEP, 10, 10)

    world.ClearForces()

    # Update the (whole) display
    pygame.display.update()
    clock.tick(TARGET_FPS)


pygame.quit()
print('Done!')





