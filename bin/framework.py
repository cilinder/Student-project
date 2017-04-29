
# This is the framework for drawing the world with the pygame backend
# The functions will recieve data in pybox2d types and will have to transform them to pygame coordinates
# We will not follow the pygame convention for ordering function parameters

from __future__ import print_function

import sys
import math
import pygame
from pygame.locals import *
from pygame import Color, Rect

import Box2D
from Box2D import *

import objectData


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


class Framework:


    # An important concert regarding point coordinates is whether they are in pixels or in pybox2d coordinates (meters).
    # We need to decide if we want to transform the coordinates or not
    # The current design decision is to add a transform=True/False parameter to all the functions, with the default being True

    def __init__(self):
        # --- pygame setup ---
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
        pygame.display.set_caption('Boat simulation enviroment')
        self.clock = pygame.time.Clock()
        self.timer_id = 0
        self.stopwatch_id = 0
        self.timers = {}
        self.stopwatches = {}
        font = None

    def update(self):
        pygame.display.update()
        self.clock.tick(TARGET_FPS)

    # Warning this is only accurate to a ceratain degree with no guarantees, use at your own risk!
    def from_pybox2d_to_pygame_coordinates(self, point):
        return (int(round(point[0] * PPM)), int(round(SCREEN_HEIGHT - (point[1] * PPM))))

    # Warning this is only accurate to a ceratain degree with no guarantees, use at your own risk!
    def from_pygame_to_pybox2d_coordinates(self, point):
        return (point[0]/PPM, (SCREEN_HEIGHT - point[1])/PPM)

    def getPPM(self):
        return PPM

    def getTimeStep(self):
        return TIME_STEP

    def getWidth(self):
        return SCREEN_WIDTH

    def getHeight(self):
        return SCREEN_HEIGHT


    def DrawString(self, string, position, color=BLACK, fontSize=20, transform=True):
        if string is None:
            print("String to be displayed is empty, not displaying anything...")
            return

        if position is None:
            print("Please select a location to display string")
            return

        try:
            font = pygame.font.Font(None, fontSize)
        except IOError:
            try:
                print("Error loading default font, trying backup")
                font = pygame.font.Font(None, fontSize)
            except IOError:
                print("Unable to load default font or 'freesansbold.ttf'")
                print("Disabling text drawing.")

        if transform:
            position = self.from_pybox2d_to_pygame_coordinates(position)
        label = font.render(string, 1, color)
        self.screen.blit(label, position)

    def DrawLine(self, startPos, endPos, color=BLACK, width=1, transform=True):
        if transform: 
            startPos = self.from_pybox2d_to_pygame_coordinates(startPos)
            endPos = self.from_pybox2d_to_pygame_coordinates(endPos)
        pygame.draw.line(self.screen, color, startPos, endPos, width)


    def DrawLines(self, pointList, closed=False, color=BLACK, width=1, transform=True):
        if transform:
            pointList = [self.from_pybox2d_to_pygame_coordinates(v) for v in pointList]
        pygame.draw.lines(self.screen, color, closed, pointList, width)

    def DrawCircle(self, position, radius, color=BLACK, width=0, transform=True):
        if transform:
            position = self.from_pybox2d_to_pygame_coordinates(position)
            radius = int(radius * PPM)
        pygame.draw.circle(self.screen, color, position, radius, width)


    def DrawPolygon(self, pointList, color=BLACK, width=0, transform=True):
        if transform:
            pointList = [self.from_pybox2d_to_pygame_coordinates(v) for v in pointList]
        pygame.draw.polygon(self.screen, color, pointList, width)


    def DrawWorld(self, color=BLACK):
        self.screen.fill(color)


    def DrawArrow(self, startPos, endPos, color=BLACK, width=1, transform=True):

        arrow_len = math.sqrt( (startPos[0]-endPos[0])**2 + (startPos[1]-endPos[1])**2)
        l = arrow_len

        (x_1, y_1) = startPos
        (x_2, y_2) = endPos

        alpha = 0
        if ( x_1 != x_2):
            #alpha = math.atan((y_2-y_1)/(x_2-x_1))
            alpha = math.atan2((y_2-y_1),(x_2-x_1))
        elif y_1 <= y_2:
            alpha = math.pi/2
        else: 
            alpha = -math.pi/2

        arrowhead_len = min(arrow_len * 0.3, 0.3)
        v = arrowhead_len * math.sqrt(2)/2

        arrowhead1_start = endPos
        arrowhead2_start = endPos
        
        cos = math.cos(alpha)
        sin = math.sin(alpha)
        arrowhead1_end = (cos*(l - v) + sin*v + x_1, sin*(l-v) - cos*v + y_1)
        arrowhead2_end = (cos*(l - v) + sin*(-v) + x_1, sin*(l-v) - cos*(-v) + y_1)
        #self.DrawLine(arrowhead1_start, arrowhead1_end)
        #self.DrawCircle(arrowhead1_end, 2)
        #self.DrawCircle(arrowhead2_end, 2)

        self.DrawLine(startPos,endPos, color, width, transform)
        self.DrawLine(endPos, arrowhead1_end, color, width, transform)
        self.DrawLine(endPos, arrowhead2_end, color, width, transform)


    def DrawGoal(self, goal, transform=True):
        vertices = [self.from_pybox2d_to_pygame_coordinates(goal.body.transform * v) for v in goal.vertices]
        self.DrawPolygon(vertices, BLACK, 2, False)
        position = (goal.initialPosition[0] + 0.3, goal.initialPosition[1] + goal.width / 2 - 0.3)
        self.DrawString("Goal", position, BLACK, fontSize=23)


    def DisplayGrid(self, x_spacing, y_spacing):

        x_0 = int(round(x_spacing * PPM))
        y_0 = int(round(y_spacing * PPM))

        for x in range(x_0, SCREEN_WIDTH, x_0):
            pygame.draw.line(self.screen, BLACK, (x,0), (x,SCREEN_HEIGHT))
            # First transform the coordinates back to pybox2d type because DrawString works with pybox2d coordinates 
            # A bit of a hack you could say, but as this is not a critical part of the program, the accuracy concerns and performance are not very relevant. 
            # This method is mostly used for debugging 
            location = self.from_pygame_to_pybox2d_coordinates((x+2, SCREEN_HEIGHT - 10))
            self.DrawString(str(x/PPM), location, fontSize=16)

        for y in range(y_0, SCREEN_HEIGHT, y_0):
            pygame.draw.line(self.screen, BLACK, (0,SCREEN_HEIGHT - y), (SCREEN_WIDTH, SCREEN_HEIGHT - y))
            location = self.from_pygame_to_pybox2d_coordinates((0, SCREEN_HEIGHT - y+1))
            self.DrawString(str(y/PPM), location, fontSize=16)

    def NextTimerId(self):
        self.timer_id += 1
        return (self.timer_id - 1)

    def ResetTimerId(self):
        self.timer_id = 0

    def StartTimer(self, endTime, timerId=-1):
        # Returns the id of the timer that was started (if one was started, else returns -1)
        startTime = pygame.time.get_ticks()

        if startTime > endTime:
            print("Unable to start timer at time: " + str(endTime) + ", timer not initiated")
            return -1

        if timerId < 0:
            timerId = self.NextTimerId()

        self.timers[timerId] = (startTime, endTime)
        return timerId

    def TimeRemaining(self, timerId):
        if self.timers[timerId] is not None:
            return self.timers[timerId][1] - pygame.time.get_ticks()
        else:
            print("Timer with timerId: " + str(timerId) + " not a valid timer")

    def TimerEndTime(self, timerId):
        if self.timers[timerId] is not None:
            return self.timers[timerId][1]
        else:
            print("Timer with timerId: " + str(timerId) + " not a valid timer")

    def TimerHasEnded(self, timerId):
        return self.timers[timerId][1] < pygame.time.get_ticks()

    def ResetAllTimers(self):
        self.timers = {}
        self.ResetTimerId()
        
    def StartStopwatch(self, stopwatchId=-1):
        # Returns the id of the stopwatch that was started (if one was started, else returns -1)
        startTime = pygame.time.get_ticks()

        if stopwatchId < 0:
            stopwatchId = self.NextStopwatchId()

        self.stopwatches[stopwatchId] = startTime
        return stopwatchId

    def StopwatchTime(self, stopwatchId):
        return pygame.time.get_ticks() - self.stopwatches[stopwatchId] 

    def NextStopwatchId(self):
        self.stopwatch_id += 1
        return (self.stopwatch_id - 1)

    def ResetStopwatchId(self):
        self.stopwatch_id = 0

    def ResetStopwatch(self, stopwatchId):
        self.StartStopwatch(stopwatchId)

    def ResetAllStopwatches(self):
        self.ResetStopwatchId()
        self.stopwatches = {}




class SimulationContactListener(b2ContactListener):

    def __init__(self):
        b2ContactListener.__init__(self)
    def BeginContact(self, contact):

        fixtureA = contact.fixtureA
        fixtureB = contact.fixtureB

        objectA = fixtureA.body.userData 
        objectB = fixtureB.body.userData 

        contact_location = contact.manifold.localPoint

        otherObject = None
        boat = None
        if objectA is not None and 'boat' in objectA.name:
            boat = objectA
            otherObject = objectB
            contact_location = fixtureA.body.transform * contact_location
        elif objectB is not None and 'boat' in objectB.name:
            boat = objectB
            otherObject = objectA
            contact_location = fixtureB.body.transform * contact_location
        else:
            p = contact.manifold.localPoint
            print("An unidentified collision has occured at location: ", p)
            return

        if otherObject.object_type == objectData.ObjectData.ObjectType.Goal:
            boat.goalReached = True

        print("The boat has hit something, the contact location is: ", contact_location.tuple)


    def EndContact(self, contact):
        pass
    def PreSolve(self, contact, oldManifold):
        pass
    def PostSolve(self, contact, impulse):
        pass



