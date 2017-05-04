# This the the boat class used for storing parameters connected with the boat

import Box2D
from Box2D import *


class ObjectData(object):

    class ObjectType:
        Unknown, Boat, Obstacle, Goal = range(4)

    num_objects = 0

    def __init__(self, position, name, object_type=0, angle=0):
        self.initialPosition = position
        self.initialAngle = angle
        self.name = name
        self.object_type = object_type

        ObjectData.num_objects += 1

    def getNumberOfObjects():
        return num_objects


class Boat(ObjectData):

    def __init__(self, position, name='', angle=0):
        super(Boat, self).__init__(position, 'boat:' + name, object_type=ObjectData.ObjectType.Boat, angle=angle)
        self.goalReached = False
        self.time = 0


    def set_angle(self, angle):
        self._angle = angle
        self.body.angle = angle

    def get_angle(self):
        self._angle = self.body.angle
        return self._angle

    angle = property(get_angle, set_angle)


class Obstacle(ObjectData):

    num_obstacles = 0

    def __init__(self, position, name='', angle=0):
        super(Obstacle, self).__init__(position, 'obstacle:' + name, object_type=ObjectData.ObjectType.Obstacle, angle=angle)

        Obstacle.num_obstacles += 1

    def getNumberOfObstacles():
        return num_obstacles



class Goal(ObjectData):

    num_goals = 0


    def __init__(self, position, width=2, height=1, name='', angle=0):
        super(Goal, self).__init__(position, 'goal:' + name, object_type=ObjectData.ObjectType.Goal, angle=angle)
        self.width = width
        self.height = height

        x = position[0]
        y = position[1]
        self.vertices = [(0,0), (width,0), (width, height), (0, height)]

        Goal.num_goals += 1


    def getNumberOfGoals():
        return num_goals




