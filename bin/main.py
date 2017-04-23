#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import math
import pygame
from pygame.locals import *

import Box2D
from Box2D import *

import Boat
from Boat import *


# --- constants ---
# Box2D deals with meters, but we want to display pixels,
# so define a conversion factor:
PPM = 30.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
SCREEN_COLOR = (0,0,255,0)
BLACK = (0,0,0,0)
RED = (255,0,0,255)

def handleKeyboardInput(keys):

    # This means the boat should accelerate
    if keys[K_w]:
        f = boat.GetWorldVector(localVector=(0,20))
        p = boat.GetWorldPoint(localPoint=(0,1))
        boat.ApplyForce(f, p, True)
    # This means decrease the angle of the rudder
    if keys[K_a]:
        boat.rudder_angle = max(-math.pi/2, boat.rudder_angle - 0.05)

    # This means increase the angle of the rudder
    if keys[K_d]:
        boat.rudder_angle = min(math.pi/2, boat.rudder_angle + 0.05)


def displayInformation(data):

    if data is not None:
        label = font.render(data, 1, (255,255,255))
        screen.blit(label, (0,0))


# --- pygame setup ---
pygame.init()
#pygame.font.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('Boat simulation enviroment')
clock = pygame.time.Clock()
font = None

try:
    font = pygame.font.Font(None, 20)
except IOError:
    try:
        font = pygame.font.Font("freesansbold.ttf", 20)
    except IOError:
        print("Unable to load default font or 'freesansbold.ttf'")
        print("Disabling text drawing.")
        self.Print = lambda *args: 0
        self.DrawStringAt = lambda *args: 0


world = b2World(gravity=(0,0), doSleep=True)  

# Create the bounding box that holds the testing area and the boat
boundingBox = world.CreateStaticBody(shapes=b2ChainShape(vertices=([(4,2), (20,2), (20,15), (4,15)])), position=(0, 0))


# Create the boat
# TODO: Create the Boat class to store the boat information
boat_object = Boat()
boat = world.CreateDynamicBody(position=(10,10), shapes=b2PolygonShape(vertices=([(0,0), (0.5,0), (0.5,1), (0.25,1.25), (0,1)])), userData=boat_object)
boat.power = 0
boat.rudder_angle = 0
boat.rudder_angle_offset = -(math.pi*1.5) # This setting is for drawing the rudder on the boat, since we want to work with angles [-pi/2, pi/2] we have to offset by -(pi*3)/2

# And add a box fixture onto it (with a nonzero density, so it will move)
box = boat.CreatePolygonFixture(box=(2, 1), density=1, friction=0.3)


# Prepare for simulation. Typically we use a time step of 1/60 of a second
# (60Hz) and 6 velocity/2 position iterations. This provides a high quality
# simulation in most game scenarios.
timeStep = 1.0 / 60
vel_iters, pos_iters = 10, 10


colors = {
        'boat' : RED,
}


# The main program loop
running = True
while running:

    # Check the event que
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            # The user closed the window or pressed escape
            running = False

    # This is how you can check for continious key pressing
    keys = pygame.key.get_pressed()
    handleKeyboardInput(keys)


    # Draw the world
    screen.fill(SCREEN_COLOR)

    display_string = ''
    display_string = display_string + str(boat.rudder_angle)

    displayInformation(display_string)

    #pygame.draw.circle(screen, BLACK, (200,200), 2) 
    #pygame.draw.line(screen, BLACK, (100, 100), (150, 100))


    # First draw the bounding box
    # It consists of pairs of coordinates representing the edges of the box
    # So we draw a line for each of the pairs
    # But first we have to transform the coordinates to on screen coordinates
    for fixture in boundingBox.fixtures:
        vertices = fixture.shape.vertices
        vertices = [(boundingBox.transform * v) * PPM for v in vertices]
        vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
        pygame.draw.lines(screen, BLACK, False, vertices)

    # Next draw the boat
    vertices = [(boat.transform * v) * PPM for v in boat.fixtures[0].shape.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    pygame.draw.polygon(screen, colors['boat'], vertices)

    # Now we want to draw the rudder on the boat
    

    # Advance the world by one step
    world.Step(TIME_STEP, 10, 10)
    # This is done to make sure everything behaves as it should
    world.ClearForces()

    # Update the (whole) display
    pygame.display.update()
    clock.tick(TARGET_FPS)


pygame.quit()
print('Done!')





