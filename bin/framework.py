
# This is the framework for drawing the world with the pygame backend
# The functions will recieve data in pybox2d types and will have to transform them to pygame coordinates
# We will not follow the pygame convention for ordering function parameters

from __future__ import print_function

import sys
import math
import pygame
from pygame.locals import *

import Box2D
from Box2D import *


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

    def __init__(self):
        # --- pygame setup ---
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
        pygame.display.set_caption('Boat simulation enviroment')
        self.clock = pygame.time.Clock()
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

    def DrawString(self, string, position, color=BLACK, fontSize=15):
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
                font = pygame.font.Font("freesansbold.ttf", fontSize)
            except IOError:
                print("Unable to load default font or 'freesansbold.ttf'")
                print("Disabling text drawing.")

        position = self.from_pybox2d_to_pygame_coordinates(position)
        label = font.render(string, 1, color)
        self.screen.blit(label, position)

    def DrawLine(self, startPos, endPos, color=BLACK, width=1):
        startPos = self.from_pybox2d_to_pygame_coordinates(startPos)
        endPos = self.from_pybox2d_to_pygame_coordinates(endPos)
        pygame.draw.line(self.screen, color, startPos, endPos, width)


    def DrawLines(self, pointList, closed=False, color=BLACK, width=1):
        pointList = [self.from_pybox2d_to_pygame_coordinates(v) for v in pointList]
        pygame.draw.lines(self.screen, color, closed, pointList, width)

    def DrawCircle(self, position, radius, color=BLACK, width=0):
        position = self.from_pybox2d_to_pygame_coordinates(position)
        pygame.draw.circle(self.screen, color, position, radius, width)


    def DrawPolygon(self, pointList, color=BLACK, width=0):
        pointList = [self.from_pybox2d_to_pygame_coordinates(v) for v in pointList]
        pygame.draw.polygon(self.screen, color, pointList, width)


    def DrawWorld(self, color=BLACK):
        self.screen.fill(color)


    def DisplayGrid(self, x_spacing, y_spacing):

        x_0 = int(round(x_spacing * PPM))
        y_0 = int(round(y_spacing * PPM))

        for x in range(x_0, SCREEN_WIDTH, x_0):
            pygame.draw.line(self.screen, BLACK, (x,0), (x,SCREEN_HEIGHT))
            # First transform the coordinates back to pybox2d type because DrawString works with pybox2d coordinates
            # A bit of a hack you could say, but as this is not a critical part of the program, the accuracy concerns and performance are not very relevant. This method is mostly used for debugging 
            location = self.from_pygame_to_pybox2d_coordinates((x+2, SCREEN_HEIGHT - 10))
            self.DrawString(str(x/PPM), location, fontSize=16)

        for y in range(y_0, SCREEN_HEIGHT, y_0):
            pygame.draw.line(self.screen, BLACK, (0,SCREEN_HEIGHT - y), (SCREEN_WIDTH, SCREEN_HEIGHT - y))
            location = self.from_pygame_to_pybox2d_coordinates((0, SCREEN_HEIGHT - y+1))
            self.DrawString(str(y/PPM), location, fontSize=16)




