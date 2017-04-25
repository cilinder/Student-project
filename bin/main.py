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

import framework

# --- constants ---
# Box2D deals with meters, but we want to display pixels,
# so define a conversion factor:
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
SCREEN_COLOR = (0,0,255,0)
BLACK = (0,0,0,0)
WHITE = (255,255,255,0)
RED = (255,0,0,255)
BLUE = (0,0,255,255)
GREEN = (0,255,0,255)

x_spacing = 2
y_spacing = 2
display_grid = False
default_string_display_offset = 0
offset = 0
strings_to_be_displayed = []

def testingFunction():

    vertices = [test_body.transform * v for v in test_body.fixtures[0].shape.vertices]
    drawer.DrawPolygon(vertices, WHITE)

    center = test_body.localCenter
    global_center = test_body.GetWorldPoint(center)
    #print(global_center)
    drawer.DrawCircle(global_center, 0.2)

    drawer.DrawCircle(boat.GetWorldPoint(boat.localCenter), 0.2, color=GREEN)


def checkEventQue():
    global display_grid
    global running
    global x_spacing, y_spacing
    global pause
    global display_hud

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
            if event.key == K_p:
                pause = not pause
            if event.key == K_h:
                display_hud = not display_hud
            if event.key == K_r:
                resetGame()


def handleKeyboardInput(keys):

    # This means the boat should accelerate
    if keys[K_w]:
        boat.power = min(boat.max_power, boat.power + 0.01)
    if keys[K_s]:
        boat.power = max(0, boat.power - 0.05)
    # This means decrease the angle of the rudder
    # This r is so you can readjust the angle to 0
    increase_factor = 0.05
    r = (math.pi/2) % increase_factor
    if keys[K_a]:
        boat.rudder_angle = max(-math.pi/2+r, boat.rudder_angle - 0.05)
    # This means increase the angle of the rudder
    if keys[K_d]:
        boat.rudder_angle = min(math.pi/2-r, boat.rudder_angle + 0.05)



# Warning this is only accurate to a ceratain degree with no guarantees, use at your own risk!
def from_pybox2d_to_pygame_coordinates(point):
    return (int(round(point[0] * PPM)), int(round(SCREEN_HEIGHT - (point[1] * PPM))))

# Warning this is only accurate to a ceratain degree with no guarantees, use at your own risk!
def from_pygame_to_pybox2d_coordinates(point):
    return (point[0]/PPM, (SCREEN_HEIGHT - point[1])/PPM)


def addStringToDisplay(string, location=None, color=(0,0,0), fontSize=20):
    global default_string_display_offset

    if string is None:
        print("String to be displayed is empty, not displaying anything...")
        return

    if location is None:
        location = from_pygame_to_pybox2d_coordinates((2, default_string_display_offset))
        default_string_display_offset += fontSize

    strings_to_be_displayed.append((string, location, color, fontSize))


def drawAllStrings():
    global default_string_display_offset

    for display in strings_to_be_displayed:
        drawer.DrawString(display[0],display[1],display[2],display[3])

    # Clean up after displaying the string list
    default_string_display_offset = 0 


def displayGrid():
    drawer.DisplayGrid(x_spacing, y_spacing)


def drawRudder():

    # A "middle point" of the boat (may not be exactly the center of mass
    point = boat.GetWorldPoint(localPoint=(0.25,0.5))
    pygame_point = from_pybox2d_to_pygame_coordinates(point)
    #drawer.DrawCircle(point, 2, BLACK)

    theta = boat.angle
    alpha = boat.rudder_angle

    #addStringToDisplay(str(alpha), fontSize=20)

    l = 0.4

    end_point = (l * math.cos(theta + math.pi*1.5 + alpha) + point[0], l * math.sin(theta + math.pi*1.5 + alpha) + point[1])
    drawer.DrawLine(point, end_point)

def displayHud():

    velocity = boat.linearVelocity
    loc_vel = boat.GetLocalVector(velocity).tuple
    alpha = boat.rudder_angle

    display_vel = "({0:.2f}".format(loc_vel[0]) + ", " + "{0:.2f})".format(loc_vel[1])

    # Draw the power display
    t = boat.power/boat.max_power
    boat.max_power*0.5 - t
    max_powerbar_length = 3
    drawer.DrawLine((0.5,0.5), (0.5 + max_powerbar_length, 0.5), color=WHITE, width=5)
    if ( t > 0 ): 
        drawer.DrawLine((0.5,0.5), ((1-t)*0.5 + t*(max_powerbar_length+0.5), 0.5), width=5)

    # Draw the steering wheel
    r = 1.5
    alpha = -boat.rudder_angle
    beta = alpha - math.pi/2
    drawer.DrawCircle((2,3), r, width=2)

    r = 1.8
    point = (-r*math.sin(alpha) + 2, r*math.cos(alpha) + 3)
    cross_point = (r*math.sin(alpha) + 2, -r*math.cos(alpha) + 3)
    drawer.DrawLine(point, cross_point, width=2)
    point2 = (-r*math.sin(beta) + 2, r*math.cos(beta) + 3)
    cross_point2 = (r*math.sin(beta) + 2, -r*math.cos(beta) + 3)
    drawer.DrawLine(point2, cross_point2, width=2)
    r = 1.5
    point = (-r*math.sin(alpha) + 2, r*math.cos(alpha) + 3)
    drawer.DrawCircle(point , 0.1, color=RED)

    addStringToDisplay("velocity: " + display_vel, fontSize=20)
    addStringToDisplay("rudder angle: "  + str(alpha), fontSize=20)

    drawAllStrings()


def applyAllForces():

    # Helper functions for transforming local to global coordinates
    point = lambda p: boat.GetWorldPoint(localPoint=p)
    vector = lambda v: boat.GetWorldVector(localVector=v)

    # This is the factor which makes the forces act in accordance with the realistic forces on a boat
    # This means gamma is a function of the size, shape of the boat, as well as location of rudder, etc.
    # With a fixed boat and idealised water conditions this parameter is a constant
    # We should fix this constant so the boat behaves realistically 
    gamma = 0.1
    delta = 1
    alpha = boat.rudder_angle

    # First we need the boat velocity
    velocity = boat.linearVelocity
    local_velocity = boat.GetLocalVector(velocity)

    # This is the of the boat swaying to the side
    sway_vel = local_velocity[0]
    # This is the "forward moving" velocity component
    surge_vel = local_velocity[1]

    # Apply the force from the propeler which is just forward with respect to the boat and with a magnitude proportional to boat.power
    f = boat.GetWorldVector(localVector=(0, boat.power))
    p = boat.GetWorldPoint(localPoint=(boat.localCenter))
    
    drawer.DrawCircle(p, radius=0.07, color=GREEN)
    # Draw the motor force arrow
    #drawer.DrawArrow(p, p+f)
    boat.ApplyForce(f, p, True)

    # The magnitude of the sway force vector to be applied
    F_s = -(surge_vel)**2 * alpha * gamma
    addStringToDisplay("F_s: " + str(F_s), fontSize=20)

    addStringToDisplay("Angular vel: " + str(boat.angularVelocity))

    # Apply the sway force based on the rudder angle and boat velocity
    f = boat.GetWorldVector(localVector=(F_s,0))
    p = boat.GetWorldPoint(localPoint=(0.25, 0))

    boat.ApplyForce(f, p, True)
    drawer.DrawArrow(p,p+f)
    
    # Now we need to apply the counter force acting on the boat by the water
    surge_velocity = (0, velocity[1])
    sway_velocity = (velocity[0], 0)

    p = point(boat.localCenter)
    
    drawer.DrawArrow(p, p + velocity)
    drawer.DrawArrow(p, p + surge_velocity)
    drawer.DrawArrow(p, p + sway_velocity)

    boat.GetWorldVector(surge_velocity)


    



def resetGame():
    print("Reset")
    boat.position = boat.userData.initialPosition
    boat.linearVelocity = (0,0)
    boat.angularVelocity = 0
    boat.angle = boat.userData.initialAngle
    boat.power = 0
    boat.rudder_angle = 0



def displayPause():

    loc = from_pygame_to_pybox2d_coordinates(((SCREEN_WIDTH/2 - 50), (SCREEN_HEIGHT/2 - 20)))
    drawer.DrawString("PAUSED", loc, color=RED, fontSize=35)


# Initialize the drawing object used to put everything on the screen
drawer = framework.Framework()
PPM = drawer.getPPM()
SCREEN_HEIGHT = drawer.getHeight()
SCREEN_WIDTH = drawer.getWidth()

world = b2World(gravity=(0,0), doSleep=True)  

# Create the bounding box that holds the testing area and the boat
boundingBox = world.CreateStaticBody(shapes=b2ChainShape(vertices=([(4,2), (20,2), (20,15), (4,15)])), position=(0, 0))


# Create the boat
# TODO: Create the Boat class to store the boat information
boat_object = Boat(position=(10,5), vertices=[(0,0), (0.5,0), (0.5,1), (0.25,1.25), (0,1)], angle=0)
#boat = world.CreateDynamicBody(position=(10,5), shapes=b2PolygonShape(vertices=([(0,0), (0.5,0), (0.5,1), (0.25,1.25), (0,1)])), userData=boat_object)
boat = world.CreateDynamicBody(position=(10,5), userData=boat_object)
boat_fixture = boat.CreatePolygonFixture(vertices=[(0,0), (0.5,0), (0.5,1), (0.25,1.25), (0,1)], friction=0.2, density=1)
boat.power = 0
boat.max_power = 2
boat.angle = boat_object.initialAngle
boat.rudder_angle = 0
boat.rudder_angle_offset = -(math.pi*1.5) # This setting is for drawing the rudder on the boat, since we want to work with angles [-pi/2, pi/2] we have to offset by -(pi*3)/2

"""
test_body = world.CreateDynamicBody(position=(7,7))
test_body_fixture = test_body.CreatePolygonFixture(box=(1,1), friction=0.2, density=1)
test_body.ResetMassData()
"""


# THIS WAS ISSUE
# And add a box fixture onto it (with a nonzero density, so it will move)
#box = boat.CreatePolygonFixture(box=(2, 1), density=1, friction=0.3)


# Prepare for simulation. Typically we use a time step of 1/60 of a second
# (60Hz) and 6 velocity/2 position iterations. This provides a high quality
# simulation in most game scenarios.
timeStep = 1.0 / 60
vel_iters, pos_iters = 10, 10


# The main program loop
running = True
pause = False
display_hud = True
while running:

    # Take care of active events like the quitting of the program
    checkEventQue()

    # Clear the display
    strings_to_be_displayed = []

    # This is how you can check for continious key pressing
    keys = pygame.key.get_pressed()
    handleKeyboardInput(keys)

    # Draw the world
    #screen.fill(SCREEN_COLOR)
    drawer.DrawWorld(SCREEN_COLOR)

    # Whether we should display the cartesian grid (mainly used for debugging)
    # Use 'g' to toggle grid on/off, use '+','-' to increase/decrease grid size
    if display_grid:
        displayGrid()

    # First draw the bounding box
    # It consists of pairs of coordinates representing the edges of the box
    # So we draw a line for each of the pairs
    # But first we have to transform the coordinates to on screen coordinates
    for fixture in boundingBox.fixtures:
        vertices = fixture.shape.vertices
        vertices = [boundingBox.transform * v for v in vertices]
        drawer.DrawLines(vertices, width=2)

    # Next draw the boat
    vertices = [boat.transform * v for v in boat.fixtures[0].shape.vertices]
    drawer.DrawPolygon(vertices, RED)

    # This is where all the temporary tests go
    #testingFunction()


    # Now we want to draw the rudder on the boat
    drawRudder()


    # Advance the world by one step
    if not pause:
        # Apply all the forces acting on the boat
        applyAllForces()
        world.Step(TIME_STEP, 10, 10)
        # This is done to make sure everything behaves as it should
        world.ClearForces()

        # Update the (whole) display
    else:
        displayPause()

    # Display all the strings we have collected in the loop
    
    if display_hud:
        displayHud()

    #drawAllStrings()
    drawer.update()


pygame.quit()
print('Done!')





