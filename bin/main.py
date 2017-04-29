#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import math
import pygame
from pygame.locals import *

import tensorflow as tf
import numpy as np

import Box2D
from Box2D import *

import objectData
from objectData import *

import matplotlib.pyplot as plt

import datetime

import framework

#########################################################
# Initializing all the variables that will be used later
# Should probably move this to an init or main method
######################################################

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
frame_by_frame_mode = False
generate_next_frame = False





#######################################
#
#   Methods
#
#####################################


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
    global frame_by_frame_mode
    global generate_next_frame

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
            if event.key == K_f:
                frame_by_frame_mode = not frame_by_frame_mode
            if frame_by_frame_mode:
                if event.key == K_n:
                    generate_next_frame = True



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


def addStringToDisplay(string, position=None, color=(0,0,0), fontSize=20):
    global default_string_display_offset

    if string is None:
        print("String to be displayed is empty, not displaying anything...")
        return

    if position is None:
        position = from_pygame_to_pybox2d_coordinates((2, default_string_display_offset))
        default_string_display_offset += fontSize

    strings_to_be_displayed.append((string, position, color, fontSize))


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

    # Display the current game timer
    timeElapsed = drawer.StopwatchTime(stopwatch) / 1000.0
    addStringToDisplay("time: " + str(timeElapsed) + " s", fontSize=22)

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
    # Delta contains the information of the inertial force acting on the boat when moving forward by the water and air in the form or water and air-resistance.
    # So it is actualy a function of the shape of the boat, the density and viscosity of the water (medium) its moving through.
    # It should be calibrated to get accurate sumilation enviroment.
    # The surge and sway parameters can (probably are) different, but we could use the same function in principle, so we can name them similarly
    # May need to vary the parameters with respect to the speed of the boat
    delta_surge = 1
    delta_sway = 2
    alpha = boat.rudder_angle

    # First we need the boat velocity
    velocity = boat.linearVelocity


    local_velocity = boat.GetLocalVector(velocity)

    # This is the of the boat swaying to the side
    sway_vel_magnitude = local_velocity[0]
    # This is the "forward moving" velocity component
    surge_vel_magnitude = local_velocity[1]

    # Apply the force from the propeler which is just forward with respect to the boat and with a magnitude proportional to boat.power
    f = boat.GetWorldVector(localVector=(0, boat.power))
    p = boat.GetWorldPoint(localPoint=(boat.localCenter))
    
    drawer.DrawCircle(p, radius=0.07, color=GREEN)
    # Draw the motor force arrow
    #drawer.DrawArrow(p, p+f)
    boat.ApplyForce(f, p, True)

    # The magnitude of the sway force vector to be applied based on the angle of the rudder
    # Should probably also take into account the force of the water acting on the rudder because of the propeler
    F_s = -surge_vel_magnitude**2 * math.sin(alpha) * gamma
    # Apply the sway force based on the rudder angle and boat velocity
    f = vector((F_s,0))
    p = point((0.25, 0))
    boat.ApplyForce(f, p, True)

    drawer.DrawArrow(p,p+f, color=GREEN)
    addStringToDisplay("F_s: " + str(F_s), fontSize=20)
    addStringToDisplay("Angular vel: " + str(boat.angularVelocity))
    addStringToDisplay("Boat angle: " + str(boat.angle))

    # Now we need to apply the counter force acting on the boat by the water
    surge_velocity = (0, velocity[1])
    sway_velocity = (velocity[0], 0)

    p = point(boat.localCenter)

    drawer.DrawArrow(p, p + velocity)
    drawer.DrawArrow(p, p + surge_velocity)
    drawer.DrawArrow(p, p + sway_velocity)

    # Apply the inertial surge force acting against the surge velocity
    F_inertial_surge = -vector((0, np.sign(surge_vel_magnitude)*surge_vel_magnitude**2 * delta_surge))
    p = point(boat.localCenter)
    boat.ApplyForce(F_inertial_surge, p, True)
    drawer.DrawArrow(p, p+F_inertial_surge, color=WHITE)

    # Apply the inertial sway force acting against the sway velocity
    # May need to also account for the angular velocity not just the sway velocity
    # as the water would try to work against rotation
    F_s = -vector((np.sign(sway_vel_magnitude)*sway_vel_magnitude**2 * delta_sway, 0))
    boat.ApplyForce(F_s, point(boat.localCenter), True)

    drawer.DrawArrow(point(boat.localCenter), point(boat.localCenter) + F_s, color=WHITE)



def resetGame():
    global frame_by_frame_mode 
    global goal_reached
    global stopwatch

    print("Reset")

    boat.position = boat.userData.initialPosition
    boat.linearVelocity = (0,0)
    boat.angularVelocity = 0
    boat.angle = boat.userData.initialAngle
    boat.power = 0
    boat.rudder_angle = 0

    frame_by_frame_mode = False
    goal_reached = False
    boat_object.goalReached = False
    boat_object.time = 0
    drawer.ResetStopwatch(stopwatch)


def displayPause():

    loc = from_pygame_to_pybox2d_coordinates(((SCREEN_WIDTH/2 - 50), (SCREEN_HEIGHT/2 - 20)))
    drawer.DrawString("PAUSED", loc, color=RED, fontSize=35)


def drawWorld():

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
    for fixture in boundingBox.body.fixtures:
        vertices = fixture.shape.vertices
        vertices = [boundingBox.body.transform * v for v in vertices]
        drawer.DrawLines(vertices, width=2)

    # Next draw the boat
    vertices = [boat.transform * v for v in boat.fixtures[0].shape.vertices]
    drawer.DrawPolygon(vertices, RED)

    # Draw the obstacle, we want this to be done automatically for all obstacles
    for buoy in obstacles:
        center = buoy.body.worldCenter
        radius = buoy.radius
        drawer.DrawCircle(center, radius)

    # Now we want to draw the rudder on the boat
    drawRudder()

    # Draw the goal separately as it looks different from other objects
    drawer.DrawGoal(goal)




def plotData():

    xaxis = np.linspace(0, len(boat.saved_angular_velocities)*TIME_STEP, len(boat.saved_angular_velocities))
    plt.figure(1)
    plt.plot(xaxis, boat.saved_angular_velocities)
    plt.xlabel('time(seconds)')
    plt.ylabel('angular velocity')
    plt.savefig('/home/ros/Student_project/data/angular_vel.png', bbox_inches='tight')

    plt.figure(2)
    v = np.asarray(boat.saved_linear_velocities)
    a = np.sqrt( v[:,0]**2 + v[:,1]**2 )
    plt.plot(xaxis, a)
    plt.xlabel('time(seconds)')
    plt.ylabel('velocity')

    plt.show()
    


def recordData():

    boat.saved_positions.append(boat.position)
    boat.saved_linear_velocities.append(boat.linearVelocity.tuple)
    boat.saved_angular_velocities.append(boat.angularVelocity)


def saveData():

    timestamp = datetime.datetime.now().isoformat()
    np.savez("/home/ros/Student_project/data/saved_run_"+str(timestamp), np.asarray(boat.saved_positions), np.asarray(boat.saved_linear_velocities), np.asarray(boat.saved_angular_velocities))





###############################################################
#
# World initialization and such, should move this to a main method of sorts
#
###############################################################



# Initialize the drawing object used to put everything on the screen
drawer = framework.Framework()
PPM = drawer.getPPM()
TIME_STEP = drawer.getTimeStep()
SCREEN_HEIGHT = drawer.getHeight()
SCREEN_WIDTH = drawer.getWidth()

world = b2World(gravity=(0,0), contactListener=framework.SimulationContactListener(), doSleep=True)  


# Create the bounding box that holds the testing area and the boat
boundingBox = ObjectData(position=(0,0), name='box')
boundingBox.body = world.CreateStaticBody(shapes=b2ChainShape(vertices=([(4,2), (20,2), (20,15), (4,15)])), position=(0, 0), userData=boundingBox)

# Create the boat
# TODO: Create the Boat class to store the boat information
boat_object = Boat(position=(10,3), angle=0)
boat = world.CreateDynamicBody(position=boat_object.initialPosition, userData=boat_object)
boat_fixture = boat.CreatePolygonFixture(vertices=[(0,0), (0.5,0), (0.5,1), (0.25,1.25), (0,1)], friction=0.2, density=1)
boat_object.body = boat
boat.angularDamping = 0.5
boat.power = 0
boat.max_power = 2
boat.angle = boat_object.initialAngle
boat.rudder_angle = 0
boat.rudder_angle_offset = -(math.pi*1.5) # This setting is for drawing the rudder on the boat, since we want to work with angles [-pi/2, pi/2] we have to offset by -(pi*3)/2
boat.saved_positions = []
boat.saved_linear_velocities = []
boat.saved_angular_velocities = []
boat.fix_in_place = False


obstacles = []

buoy1 = Obstacle(position=(10,9), name='buoy1')
buoy1.radius = 0.7
buoy1.body = world.CreateStaticBody(position=buoy1.initialPosition, shapes=b2CircleShape(pos=buoy1.initialPosition, radius=buoy1.radius), userData=buoy1)
buoy1.body.CreateCircleFixture(radius=buoy1.radius)

buoy2 = Obstacle(position=(12,9), name='buoy2')
buoy2.radius = 0.7
buoy2.body = world.CreateStaticBody(position=buoy2.initialPosition, shapes=b2CircleShape(pos=buoy2.initialPosition, radius=buoy2.radius), userData=buoy2)
buoy2.body.CreateCircleFixture(radius=buoy2.radius)

obstacles.append(buoy1)
obstacles.append(buoy2)

goal = Goal(position=(11,12), width=2, height=1)
goal.body = world.CreateStaticBody(position=goal.initialPosition, shapes=b2PolygonShape(vertices=goal.vertices), userData=goal)
goal.body.fixtures[0].sensor = True


# Prepare for simulation. Typically we use a time step of 1/60 of a second
# (60Hz) and 6 velocity/2 position iterations. This provides a high quality
# simulation in most game scenarios.
timeStep = 1.0 / 60
vel_iters, pos_iters = 10, 10


# The main program loop
running = True
pause = False
display_hud = True
goal_reached = False
stopwatch = drawer.StartStopwatch()

# Turn on live plotting for matplotlib
#plt.ion()


while running:
    
    # Take care of active events like the quitting of the program
    checkEventQue()

    # Clear the display
    strings_to_be_displayed = []

    # We have reached a goal, waiting for reset
    if goal_reached:
        pos = ((SCREEN_WIDTH/2 - 150), (SCREEN_HEIGHT/2 - 20))
        #print(str(time))
        drawer.DrawString("GOAL REACHED, time was: " + str(boat_object.time) + " s", position=pos, color=RED, transform=False, fontSize=25)
        drawer.DrawString("press r to reset", position=(pos[0], pos[1]+20), color=RED, transform=False)
        drawer.update()
        continue


    if frame_by_frame_mode:
        if generate_next_frame:
            generate_next_frame = False
        else:
            drawer.DrawString("Frame by frame mode activated, press 'n' to generate next frame", position=(3.8, 1), color=RED, fontSize=25)
            drawer.update()
            continue

    drawWorld()

    # This is how you can check for continious key pressing
    keys = pygame.key.get_pressed()
    handleKeyboardInput(keys)

    # This is where all the temporary tests go
    #testingFunction()

    # Advance the world by one step
    if not pause:
        # Apply all the forces acting on the boat
        applyAllForces()

        world.Step(TIME_STEP, 10, 10)
        # This is done to make sure everything behaves as it should

        if boat.fix_in_place:
            boat.position = boat.userData.initialPosition
            boat.angle = boat_object.initialAngle
            #boat.angularVelocity = 0

        world.ClearForces()

        # Update the (whole) display
    else:
        displayPause()

    if display_hud:
        displayHud()
    
    drawer.update()

    recordData()


    # Now lets check if we have reached the goal
    if boat_object.goalReached == True:
        goal_reached = True
        boat_object.time = drawer.StopwatchTime(stopwatch) / 1000.0




pygame.quit()

#saveData()

#plotData()

print('Done!')





