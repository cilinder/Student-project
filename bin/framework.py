
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
        clock = pygame.time.Clock()
        font = None

    # Warning this is only accurate to a ceratain degree with no guarantees, use at your own risk!
    def from_pybox2d_to_pygame_coordinates(point):
        return (int(round(point[0] * PPM)), int(round(SCREEN_HEIGHT - (point[1] * PPM))))

    # Warning this is only accurate to a ceratain degree with no guarantees, use at your own risk!
    def from_pygame_to_pybox2d_coordinates(point):
        return (point[0]/PPM, (SCREEN_HEIGHT - point[1])/PPM)

    def DrawString(string, position, color=BLACK, fontSize=15):
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
        self.screen.blit(label, position)

    def DrawLine(start_pos, end_pos, color=BLACK, width=1):
        pygame.draw.line(self.screen, color, start_pos, end_pos, width)


    def DrawCircle(position, radius, color=BLACK, width=0):
        pygame.draw.circle(self.screen, color, position, radius, width)


    def DrawPolygon(pointlist, color, width=0):
        pygame.draw.polygon(self.screen, color, pointlist, width)


    def DrawWorld(color=BLACK):
        self.screen.fill(color)




