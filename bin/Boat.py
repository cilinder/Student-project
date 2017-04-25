# This the the boat class used for storing parameters connected with the boat

import Box2D
from Box2D import *

class Boat:

    def __init__(self, position, vertices, angle=0):
        self.initialPosition = position
        self.vertices = vertices
        self.initialAngle = angle


