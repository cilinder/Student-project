#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
#hl


import sys
import getopt
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

from levels import Level


#########################################################
# Initializing all the variables that will be used later
# Should probably move this to an init or main method
######################################################

SCREEN_COLOR = (0,0,255,0)
BLACK = (0,0,0,0)
WHITE = (255,255,255,0)
RED = (255,0,0,255)
BLUE = (0,0,255,255)
GREEN = (0,255,0,255)

DO_NOTHING = np.array([1, 0, 0, 0, 0])
INCREASE_POWER = np.array([0, 1, 0, 0, 0])
DECREASE_POWER = np.array([0, 0, 1, 0, 0])
STEER_LEFT = np.array([0, 0, 0, 1, 0])
STEER_RIGHT = np.array([0, 0, 0, 0, 1])

# May want to move this to the learner class, because we don't want to have each instance of the game to have new key press listeners
def checkEventQue():

    # Check the event que
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            # The user closed the window or pressed escape
            self.running = False
        if event.type == KEYDOWN:
            if event.key == K_g:
                self.display_grid = not self.display_grid
            if event.key == K_PLUS:
                m = min(self.SCREEN_HEIGHT,self.SCREEN_WIDTH) / self.PPM
                self.x_spacing = min(m, self.x_spacing + 0.5)
                self.y_spacing = min(m, self.y_spacing + 0.5)
            if event.key == K_MINUS:
                self.x_spacing = max(0.5, self.x_spacing - 0.5)
                self.y_spacing = max(0.5, self.y_spacing - 0.5)
            if event.key == K_p:
                self.pause = not self.pause
            if event.key == K_h:
                self.display_hud = not self.display_hud
            if event.key == K_r:
                self.resetGame()
            if event.key == K_o:
                self.display_highscores = not self.display_highscores
            if event.key == K_b:
                self.boat_vision_on = not self.boat_vision_on




class Enviroment:

#####################################
#                                   #
#   Methods                         #
#                                   #
#####################################



    # Also want to move this to the learner class
    def handleKeyboardInput(self, keys):

        # This means the boat should accelerate
        if keys[K_w]:
            self.level.boat.power = min(self.level.boat.max_power, self.level.boat.power + 0.01)
        if keys[K_s]:
            self.level.boat.power = max(0, self.level.boat.power - 0.05)
        # This means decrease the angle of the rudder
        # This r is so you can readjust the angle to 0
        increase_factor = 0.05
        r = (math.pi/2) % increase_factor
        if keys[K_a]:
            self.level.boat.rudder_angle = max(-math.pi/2+r, self.level.boat.rudder_angle - 0.02)
        # This means increase the angle of the rudder
        if keys[K_d]:
            self.level.boat.rudder_angle = min(math.pi/2-r, self.level.boat.rudder_angle + 0.02)



    # We want this to be a class method, so all the instances can use a common calculator
    # Warning this is only accurate to a ceratain degree with no guarantees, use at your own risk!
    def from_pybox2d_to_pygame_coordinates(self, point):
        return (int(round(point[0] * self.PPM)), int(round(self.SCREEN_HEIGHT - (point[1] * self.PPM))))

    # Warning this is only accurate to a ceratain degree with no guarantees, use at your own risk!
    def from_pygame_to_pybox2d_coordinates(self, point):
        return (point[0]/self.PPM, (self.SCREEN_HEIGHT - point[1])/self.PPM)


    def addStringToDisplay(self, string, position=None, color=(0,0,0), fontSize=20):

        if string is None:
            print("String to be displayed is empty, not displaying anything...")
            return

        if position is None:
            position = (2, self.default_string_display_offset)
            self.default_string_display_offset += fontSize

        self.strings_to_be_displayed.append((string, position, color, fontSize))


    def drawAllStrings(self):

        for display in self.strings_to_be_displayed:
            self.drawer.DrawString(display[0],display[1],display[2],display[3], transform=False)

        # Clean up after displaying the string list
        self.default_string_display_offset = 0 


    def displayGrid(self):
        self.drawer.DisplayGrid(self.x_spacing, self.y_spacing)


    def drawRudder(self):

        # A "middle point" of the boat (may not be exactly the center of mass
        point = self.level.boat.body.GetWorldPoint(localPoint=(0.25,0.5))
        pygame_point = self.from_pybox2d_to_pygame_coordinates(point)

        theta = self.level.boat.angle
        alpha = self.level.boat.rudder_angle

        #addStringToDisplay(str(alpha), fontSize=20)

        l = 0.4

        end_point = (l * math.cos(theta + math.pi*1.5 + alpha) + point[0], l * math.sin(theta + math.pi*1.5 + alpha) + point[1])
        self.drawer.DrawLine(point, end_point)

    def displayHud(self):

        velocity = self.level.boat.body.linearVelocity
        loc_vel = self.level.boat.body.GetLocalVector(velocity).tuple
        alpha = self.level.boat.rudder_angle

        display_vel = "({0:.2f}".format(loc_vel[0]) + ", " + "{0:.2f})".format(loc_vel[1])

        # Draw the power display
        t = self.level.boat.power/self.level.boat.max_power
        self.level.boat.max_power*0.5 - t
        max_powerbar_length = 3
        self.drawer.DrawLine((0.5,0.5), (0.5 + max_powerbar_length, 0.5), color=WHITE, width=5)
        if ( t > 0 ): 
            self.drawer.DrawLine((0.5,0.5), ((1-t)*0.5 + t*(max_powerbar_length+0.5), 0.5), width=5)

        # Draw the steering wheel
        r = 1.5
        alpha = -self.level.boat.rudder_angle
        beta = alpha - math.pi/2
        self.drawer.DrawCircle((2,3), r, width=2)

        r = 1.8
        point = (-r*math.sin(alpha) + 2, r*math.cos(alpha) + 3)
        cross_point = (r*math.sin(alpha) + 2, -r*math.cos(alpha) + 3)
        self.drawer.DrawLine(point, cross_point, width=2)
        point2 = (-r*math.sin(beta) + 2, r*math.cos(beta) + 3)
        cross_point2 = (r*math.sin(beta) + 2, -r*math.cos(beta) + 3)
        self.drawer.DrawLine(point2, cross_point2, width=2)
        r = 1.5
        point = (-r*math.sin(alpha) + 2, r*math.cos(alpha) + 3)
        self.drawer.DrawCircle(point , 0.1, color=RED)

        self.addStringToDisplay("FPS: " + str(self.drawer.getFps()))
        self.addStringToDisplay("velocity: " + display_vel, fontSize=20)
        self.addStringToDisplay("rudder angle: "  + str(alpha), fontSize=20)

        # Display the current game timer
        timeElapsed = self.drawer.StopwatchTime(self.stopwatch) / 1000.0
        self.addStringToDisplay("time: " + str(timeElapsed) + " s", fontSize=22)

        self.drawAllStrings()


    def applyAllForces(self):

        # Helper functions for transforming local to global coordinates
        point = lambda p: self.level.boat.body.GetWorldPoint(localPoint=p)
        vector = lambda v: self.level.boat.body.GetWorldVector(localVector=v)

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
        alpha = self.level.boat.rudder_angle

        # First we need the boat velocity
        velocity = self.level.boat.body.linearVelocity


        local_velocity = self.level.boat.body.GetLocalVector(velocity)

        # This is the of the boat swaying to the side
        sway_vel_magnitude = local_velocity[0]
        # This is the "forward moving" velocity component
        surge_vel_magnitude = local_velocity[1]

        # Apply the force from the propeler which is just forward with respect to the boat and with a magnitude proportional to boat.power
        f = self.level.boat.body.GetWorldVector(localVector=(0, self.level.boat.power))
        p = self.level.boat.body.GetWorldPoint(localPoint=(self.level.boat.body.localCenter))
        
        self.drawer.DrawCircle(p, radius=0.07, color=GREEN)
        # Draw the motor force arrow
        #self.drawer.DrawArrow(p, p+f)
        self.level.boat.body.ApplyForce(f, p, True)

        # The magnitude of the sway force vector to be applied based on the angle of the rudder
        # Should probably also take into account the force of the water acting on the rudder because of the propeler
        F_s = -surge_vel_magnitude**2 * math.sin(alpha) * gamma
        # Apply the sway force based on the rudder angle and boat velocity
        f = vector((F_s,0))
        p = point((0.25, 0))
        self.level.boat.body.ApplyForce(f, p, True)

        self.drawer.DrawArrow(p,p+f, color=GREEN)
        self.addStringToDisplay("F_s: " + str(F_s), fontSize=20)
        self.addStringToDisplay("Angular vel: " + str(self.level.boat.body.angularVelocity))
        self.addStringToDisplay("Boat angle: " + str(self.level.boat.angle))

        # Now we need to apply the counter force acting on the boat by the water
        surge_velocity = (0, velocity[1])
        sway_velocity = (velocity[0], 0)

        p = point(self.level.boat.body.localCenter)

        self.drawer.DrawArrow(p, p + velocity)
        self.drawer.DrawArrow(p, p + surge_velocity)
        self.drawer.DrawArrow(p, p + sway_velocity)

        # Apply the inertial surge force acting against the surge velocity
        F_inertial_surge = -vector((0, np.sign(surge_vel_magnitude)*surge_vel_magnitude**2 * delta_surge))
        p = point(self.level.boat.body.localCenter)
        self.level.boat.body.ApplyForce(F_inertial_surge, p, True)
        self.drawer.DrawArrow(p, p+F_inertial_surge, color=WHITE)

        # Apply the inertial sway force acting against the sway velocity
        # May need to also account for the angular velocity not just the sway velocity
        # as the water would try to work against rotation
        F_s = -vector((np.sign(sway_vel_magnitude)*sway_vel_magnitude**2 * delta_sway, 0))
        self.level.boat.body.ApplyForce(F_s, point(self.level.boat.body.localCenter), True)

        self.drawer.DrawArrow(point(self.level.boat.body.localCenter), point(self.level.boat.body.localCenter) + F_s, color=WHITE)


    def resetGame(self):

        print("Reset")

        self.level.boat.body.position = self.level.boat.initialPosition
        self.level.boat.body.linearVelocity = (0,0)
        self.level.boat.body.angularVelocity = 0
        self.level.boat.angle = self.level.boat.initialAngle
        self.level.boat.power = 0
        self.level.boat.rudder_angle = 0

        self.goal_reached = False
        self.level.boat.goal_reached = False
        self.level.boat.hit_obstacle = False
        self.level.boat.time = 0
        self.drawer.ResetStopwatch(self.stopwatch)


    def displayPause(self):

        loc = self.from_pygame_to_pybox2d_coordinates(((self.SCREEN_WIDTH/2 - 50), (self.SCREEN_HEIGHT/2 - 20)))
        self.drawer.DrawString("PAUSED", loc, color=RED, fontSize=35)


    def drawWorld(self):

        # Draw the world
        #screen.fill(SCREEN_COLOR)
        self.drawer.DrawWorld(SCREEN_COLOR)

        # Whether we should display the cartesian grid (mainly used for debugging)
        # Use 'g' to toggle grid on/off, use '+','-' to increase/decrease grid size
        if self.display_grid:
            self.displayGrid()

        # Next draw the boat
        vertices = [self.level.boat.body.transform * v for v in self.level.boat.body.fixtures[0].shape.vertices]
        self.drawer.DrawPolygon(vertices, RED)

        # Now we want to draw the rudder on the boat
        self.drawRudder()

        if not self.boat_vision_on:
            # First draw the bounding box
            # It consists of pairs of coordinates representing the edges of the box
            # So we draw a line for each of the pairs
            # But first we have to transform the coordinates to on screen coordinates
            for fixture in self.level.bounding_box.body.fixtures:
                vertices = fixture.shape.vertices
                vertices = [self.level.bounding_box.body.transform * v for v in vertices]
                self.drawer.DrawLines(vertices, width=2)

            # Draw the obstacle, we want this to be done automatically for all obstacles
            for buoy in self.level.obstacles:
                center = buoy.body.worldCenter
                radius = buoy.body.fixtures[0].shape.radius
                self.drawer.DrawCircle(center, radius)
        elif self.boat_vision_on:
            # Draw only what the boat sees
            zero = np.array([0,0])
            for point in (point for point in self.level.boat.vision if not np.array_equal(point, zero)):
                self.drawer.DrawCircle(point, 0.02, color=GREEN)


        # Draw the goal separately as it looks different from other objects
        for goal in self.level.goals:
            self.drawer.DrawGoal(goal)


    def plotData(self):

        xaxis = np.linspace(0, len(self.level.boat.saved_angular_velocities)*self.TIME_STEP, len(self.level.boat.saved_angular_velocities))
        plt.figure(1)
        plt.plot(xaxis, self.level.boat.saved_angular_velocities)
        plt.xlabel('time(seconds)')
        plt.ylabel('angular velocity')
        plt.savefig('/home/ros/Student_project/data/angular_vel.png', bbox_inches='tight')

        plt.figure(2)
        v = np.asarray(self.level.boat.saved_linear_velocities)
        a = np.sqrt( v[:,0]**2 + v[:,1]**2 )
        plt.plot(xaxis, a)
        plt.xlabel('time(seconds)')
        plt.ylabel('velocity')

        plt.show()


    def recordData(self):

        self.level.boat.saved_positions.append(self.level.boat.body.position)
        self.level.boat.saved_linear_velocities.append(self.level.boat.body.linearVelocity.tuple)
        self.level.boat.saved_angular_velocities.append(self.level.boat.body.angularVelocity)


    def saveData(self):

        timestamp = datetime.datetime.now().isoformat()
        np.savez("/home/ros/Student_project/data/saved_run_"+str(timestamp), np.asarray(self.level.boat.saved_positions), np.asarray(self.level.boat.saved_linear_velocities), np.asarray(self.level.boat.saved_angular_velocities))


    def scanFOV(self):

        theta = self.level.boat.FOV_angle

        phi = self.level.boat.angle

        d = self.level.boat.view_distance
        n = self.level.boat.number_of_rays
        angles = np.linspace(-theta/2, theta/2, n)

        # Where to cast the ray from, currently the tip of the boat, can move it to another location where the camera is on the boat
        P = self.level.boat.body.GetWorldPoint(self.level.boat.position_of_camera)

        f = lambda alpha: np.array([-np.sin(alpha), np.cos(alpha)])

        # Rotation by phi ?
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

        i = 0
        for alpha in angles:
            end_point = np.asarray(P) + d * R.dot(f(alpha))
            self.level.boat.vision_array[i] = end_point

            self.drawer.DrawCircle(end_point, 0.02)
            if i == 0 or i == len(angles) - 1:
                self.drawer.DrawLine(P, end_point)

            i += 1

        for point in self.level.boat.vision_array:
            self.level.world.RayCast(self.callback, P, point)


        # Get only the points where the rays hit some object
        ray_intersections = self.callback.hits[0:self.callback.num_hits]

        # Now we want to calculate the angle they hit at so we can replace the points the ray was headed to
        # but first the need to move the points to the origin

        # First move them, then rotate so the angle will fall between -theta/2 and theta/2
        moved_to_origin = ray_intersections - self.level.boat.body.GetWorldPoint(self.level.boat.position_of_camera)

        # Rotate the point so the boat would have an angle of 0
        # Lets call the angle alpha
        alpha = self.level.boat.body.angle
        R = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])

        rotated = np.dot(moved_to_origin, R.T)
        R0 = np.array([[0, 1], [-1, 0]])

        rotated = np.dot(rotated, R0.T)

        # Then rotate by -pi/2 to get the angles to between -theta/2, theta/2 since the boat is at an angle of pi/2

        # Now find the angle by applying arctan(y/x) with the atan2 function
        measured_angles = np.arctan2(rotated[:,1], rotated[:,0])

        indices = np.around((n-1) * (measured_angles/float(theta)  + 0.5))

        #indices = (measured_angles * n) / theta
        indices = [int(ind) for ind in indices]

        self.level.boat.vision_array[indices] = ray_intersections

        for point in self.level.boat.vision_array:
            self.drawer.DrawCircle(point, 0.04, color=GREEN)

        hits = self.callback.hits[self.callback.hits != [0,0]]
        hits = hits.reshape(hits.size/2, 2)
        P = np.tile(np.asarray(P), (self.callback.num_hits,1))

        # The random parameter to account for the inperfection of the measurement
        mu = self.level.boat.camera_mu
        sigma = self.level.boat.camera_sigma
        gauss = np.random.normal(mu, sigma, self.callback.num_hits)
        gauss = gauss[:,np.newaxis]

        rand_hits = gauss*(P-hits) + hits

        self.level.boat.vision = rand_hits
        
        #print(self.callback.dist)
        self.callback.num_hits = 0
        self.callback.hits.fill(0)



#####################################################################################
#                                                                                   #
# World initialization and such                                                     #
#                                                                                   #
#####################################################################################


    def __init__(self, argv, displayWorld=True):


        # Parse the command line arguments which include: the player name, the level to be loaded
        try:
            opts, args = getopt.getopt(argv, "", ["player=","level="])
        except getopt.GetoptError:
            print("Format: python main.py --player=\'name\' --level=\'1\'")
            sys.exit(2)

        # Default values for the player and level
        self.player_name = "Jure"
        self.level_id = "1"

        for opt, arg in opts:
            if opt == "--player":
                self.player_name = arg
            if opt == "--level":
                print('here')
                self.level_id = arg


        # Initialize the drawing object used to put everything on the screen
        # Parameters are (width, height, PPM, fps)
        # maybe rename self.drawer to framework ?
        self.drawer = framework.Framework(640, 580, 30, 60, displayWorld)
        self.PPM = self.drawer.getPPM()
        self.TIME_STEP = self.drawer.getTimeStep()
        self.SCREEN_HEIGHT = self.drawer.getHeight()
        self.SCREEN_WIDTH = self.drawer.getWidth()

        # Here is where all the physical objects as well as the world that contains them reside (this includes highscores)
        # Should maybe rename this to world
        self.level = Level(self.level_id)

        # Set the number of ticks this simulation has run (how many states we have seen)
        self.timeTicks = 0

        self.displayWorld = displayWorld
        self.running = True
        self.pause = False
        self.display_hud = True
        self.goal_reached = False
        self.hit_obstacle = False
        self.display_highscores = False
        self.boat_vision_on = False
        self.stopwatch = self.drawer.StartStopwatch()
        self.x_spacing = 2
        self.y_spacing = 2
        self.display_grid = False
        self.default_string_display_offset = 0
        self.offset = 0
        self.strings_to_be_displayed = []
        self.manual_control = False

        # Turn on live plotting for matplotlib
        #plt.ion()

        # The callback object for handling scanning the field of vision
        self.callback = framework.RayCastCallback(level=self.level)


#################################
#                               #
#   The main program loop       #
#                               #
#################################

    
    # This method is how you interact with the enviroment/advance the world one time step
    def nextState(self, action):

        self.applyAction(action)

        # Clear the display
        self.strings_to_be_displayed = []

        # This is where all the temporary tests go
        # testingFunction()

        # Advance the world by one step
        if not self.pause:

            self.scanFOV()

            # Apply all the forces acting on the boat
            self.applyAllForces()

            self.level.world.Step(self.TIME_STEP, 10, 10)
            # This is done to make sure everything behaves as it should

            if self.level.boat.fix_in_place:
                self.level.boat.body.position = self.level.boat.initialPosition
                self.level.boat.angle = self.level.boat.initialAngle
                #boat.angularVelocity = 0

            self.level.world.ClearForces()

            # Update the (whole) display
        else:
            self.displayPause()

        if self.display_hud:
            self.displayHud()

        if self.display_highscores:
            self.drawer.DisplayHighscores(self.level.highscores)

        self.drawer.update()

        self.recordData()

        # Now lets check if we have reached the goal
        if self.level.boat.goal_reached == True:
            self.goal_reached = True
            self.level.boat.time = self.drawer.StopwatchTime(self.stopwatch) / 1000.0
            # Lets add the highscore to the highscore list
            self.level.add_highscore(self.level.boat.time, self.player_name)

        if self.displayWorld:
            self.drawWorld()

        # Advance the time by 1 to represent the number of time ticks
        self.timeTicks += 1


    def applyAction(self, action):

        increase_factor = 0.05
        r = (math.pi/2) % increase_factor

        if np.array_equal(action, DO_NOTHING):
            # Do nothing
            pass
        if np.array_equal(action, INCREASE_POWER):
            self.level.boat.power = min(self.level.boat.max_power, self.level.boat.power + 0.01)
        if np.array_equal(action, DECREASE_POWER):
            self.level.boat.power = max(0, self.level.boat.power - 0.01)
        if np.array_equal(action, STEER_LEFT):
            self.level.boat.rudder_angle = max(-math.pi/2+r, self.level.boat.rudder_angle - 0.02)
        if np.array_equal(action, STEER_RIGHT):
            self.level.boat.rudder_angle = min(math.pi/2-r, self.level.boat.rudder_angle + 0.02)


    def currentState(self):
        """
        Return the current state of the world defined as a feature vector
        """


        # Create a vector of distances to each point hit by the rays
        # TODO: decide how to handle rays that didn't hit anything, either just have it like they hit the maximum distance point, or have a special value (like -1)
        # to represent an infinite distance (does a neural network learn that ?).
        n = len(self.level.boat.vision_array)
        p_0 = self.level.boat.body.GetWorldPoint(self.level.boat.position_of_camera)

        D = (self.level.boat.vision_array - p_0)**2
        D = np.sum(D, axis=1)
        D = np.sqrt(D)

        # Add the x and y-distance to the goal, so the agent is able to effectively find it
        # This represents a gps location of the goal
        boat_pos = self.level.boat.body.worldCenter
        goal_pos = self.level.goals[0].body.worldCenter
        dist_x = goal_pos[0] - boat_pos[0]
        dist_y = goal_pos[1] - boat_pos[1]


        state = np.concatenate([[dist_x], [dist_y], D])

        state.shape = (1, len(state))

        return state


    def getReward(self):
        """
        Returns the reward for the current state
        """
        if self.level.boat.goal_reached:
            return 1000
        elif self.level.boat.hit_obstacle:
            return -1000
        else: 
            return -1


    def getTimeTicks(self):
        return self.timeTicks


    def gameHasEnded(self):
        if not self.running or self.level.boat.hit_obstacle or self.level.boat.goal_reached:
            return True
        else:
            return False
    



