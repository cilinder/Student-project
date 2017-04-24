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

x_spacing = 2
y_spacing = 2
display_grid = False
default_string_display_offset = 0
offset = 0
strings_to_be_displayed = []


def handleKeyboardInput(keys):

    # This means the boat should accelerate
    if keys[K_w]:
        boat.power = min(10, boat.power + 0.5)
    if keys[K_d]:
        boat.power = max(0, boat.power - 0.5)
    # This means decrease the angle of the rudder
    if keys[K_a]:
        boat.rudder_angle = max(-math.pi/2, boat.rudder_angle - 0.05)

    # This means increase the angle of the rudder
    if keys[K_d]:
        boat.rudder_angle = min(math.pi/2, boat.rudder_angle + 0.05)



# Warning this is only accurate to a ceratain degree with no guarantees, use at your own risk!
def from_pybox2d_to_pygame_coordinates(point):
    return (int(round(point[0] * PPM)), int(round(SCREEN_HEIGHT - (point[1] * PPM))))

# Warning this is only accurate to a ceratain degree with no guarantees, use at your own risk!
def from_pygame_to_pybox2d_coordinates(point):
    return (point[0]/PPM, (SCREEN_HEIGHT - point[1])/PPM)


def displayString(string, location, color, fontSize):
    global default_string_display_offset

    if string is None:
        print("String to be displayed is empty, not displaying anything...")
        return

    if location is None:
        print("Please select a location to display string")
        return

    try:
        font = pygame.font.Font(None, fontSize)
    except IOError:
        try:
            print("Error loading default font, trying backup")
            font = pygame.font.Font("freesansbold.ttf", fontSize)
        except IOError:
            print("Unable to load default font or 'freesansbold.ttf'")
            print("Disabling text drawing.")

    label = font.render(string, 1, color)
    screen.blit(label, location)


def addStringToDisplay(string, location=None, color=(0,0,0), fontSize=15):
    global default_string_display_offset

    if string is None:
        print("String to be displayed is empty, not displaying anything...")
        return

    if location is None:
        location = (2, default_string_display_offset)
        default_string_display_offset += fontSize

    strings_to_be_displayed.append((string, location, color, fontSize))


def displayAllStrings():
    global default_string_display_offset
    for display in strings_to_be_displayed:
        displayString(display[0],display[1],display[2],display[3])

    # Clean up after displaying the stiring list
    default_string_display_offset = 0


def displayGrid():
    x_0 = int(round(x_spacing * PPM))
    y_0 = int(round(y_spacing * PPM))

    for x in range(x_0, SCREEN_WIDTH, x_0):
        pygame.draw.line(screen, BLACK, (x,0), (x,SCREEN_HEIGHT))
        addStringToDisplay(str(x/PPM), (x+2, SCREEN_HEIGHT - 10), fontSize=16)

    y_max = 0
    for y in range(y_0, SCREEN_HEIGHT, y_0):
        pygame.draw.line(screen, BLACK, (0,SCREEN_HEIGHT - y), (SCREEN_WIDTH, SCREEN_HEIGHT - y))
        addStringToDisplay(str(y/PPM), (0, SCREEN_HEIGHT - y+1), fontSize=16)



def drawRudder():

    # A "middle point" of the boat (may not be exactly the center of mass
    point = boat.GetWorldPoint(localPoint=(0.25,0.5))
    pygame_point = from_pybox2d_to_pygame_coordinates(point)
    #addStringToDisplay(str(point), pygame_point)
    pygame.draw.circle(screen, BLACK, pygame_point, 2)

    theta = boat.angle
    alpha = boat.rudder_angle

    addStringToDisplay(str(alpha), fontSize=20)

    l = 0.4
    #p = (l*math.cos(theta)+point[0], l*math.sin(theta)+point[1])
    #pygame.draw.line(screen, BLACK, pygame_point,from_pybox2d_to_pygame_coordinates(p))

    end_point = (l * math.cos(theta + math.pi*1.5 + alpha) + point[0], l * math.sin(theta + math.pi*1.5 + alpha) + point[1])
    end_point = from_pybox2d_to_pygame_coordinates(end_point)

    pygame.draw.line(screen, BLACK, pygame_point, end_point)


def applyAllForces():

    f = 



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
boat.angle = 0
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
        if event.type == KEYDOWN:
            if event.key == K_g:
                display_grid = not display_grid
            if event.key == K_PLUS:
                m = min(SCREEN_HEIGHT,SCREEN_WIDTH) / PPM
                x_spacing = min(m, x_spacing + 0.5)
                y_spacing = min(m, y_spacing + 0.5)
            if event.key == K_MINUS:
                x_spacing = max(0.5, x_spacing - 0.5)
                y_spacing = max(0.5, y_spacing - 0.5)

    # Clear the display
    strings_to_be_displayed = []

    # This is how you can check for continious key pressing
    keys = pygame.key.get_pressed()
    handleKeyboardInput(keys)


    # Draw the world
    screen.fill(SCREEN_COLOR)

    display_string = ''
    display_string = display_string + str(boat.rudder_angle)
    display_string = display_string + str(boat.angle)

    point = boat.GetWorldPoint(localPoint=(0.5,0.25))

    addStringToDisplay("sample")

    # Whether we should display the cartesian grid (mainly used for debugging)
    # Use 'g' to toggle grid on/off, use '+','-' to increase/decrease grid size
    if display_grid:
        displayGrid()

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
        pygame.draw.lines(screen, BLACK, False, vertices, 2)

    # Next draw the boat
    vertices = [(boat.transform * v) * PPM for v in boat.fixtures[0].shape.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    pygame.draw.polygon(screen, colors['boat'], vertices)

    # Now we want to draw the rudder on the boat
    drawRudder()

    # Display all the strings we have collected in the loop
    displayAllStrings()

    # Apply all the forces acting on the boat
    applyAllForces()

    # Advance the world by one step
    world.Step(TIME_STEP, 10, 10)
    # This is done to make sure everything behaves as it should
    world.ClearForces()

    # Update the (whole) display
    pygame.display.update()
    clock.tick(TARGET_FPS)


pygame.quit()
print('Done!')





