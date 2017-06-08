# Python standard libraries
import math
import json
import bisect

import Box2D
from Box2D import *

# My libraries
import objectData
from objectData import ObjectData, Boat, Obstacle, Goal

import numpy as np

import framework


class Level(object):


    def __init__(self, args="1"):
        super(Level, self).__init__()

        level_id = args

        self.filename = "/home/ros/Student_project/levels/level_" + level_id + ".json"

        json_data = open(self.filename)
        self.data = json.load(json_data)

        # Create the world
        self.world = b2World(gravity=(0,0), contactListener=framework.SimulationContactListener(), doSleep=True)  

        # Create the bounding box that holds the testing area and the boat
        bounding_box_json = self.data["bounding_box"]

        # Scale everything by a factor
        scale = 1

        #scaled_vertices = bounding_box_json["vertices"] * 2
        scaled_vertices = [[v[0]*scale,v[1]*scale] for v in bounding_box_json["vertices"]]

        self.bounding_box = ObjectData(position=(bounding_box_json["position"]), name='box')
        self.bounding_box.body = self.world.CreateStaticBody(
                shapes=b2ChainShape(vertices=scaled_vertices), 
                position=self.bounding_box.initialPosition, 
                userData=self.bounding_box)

        # Create the boat
        boat_json = self.data["boat"]

        scaled_position = [boat_json["position"][0] * scale, boat_json["position"][1] * scale]

        self.boat = Boat(position=scaled_position, angle=boat_json["angle"])

        
        self.boat.body = self.world.CreateDynamicBody(position=self.boat.initialPosition, userData=self.boat)
        self.boat.body.CreatePolygonFixture(vertices=boat_json["vertices"], friction=0.2, density=1)
        self.boat.body.angularDamping = 0.5
        self.boat.power = 0
        self.boat.max_power = 2
        self.boat.body.angle = self.boat.initialAngle
        self.boat.rudder_angle = 0
        self.boat.rudder_angle_offset = -(math.pi*1.5) # This setting is for drawing the rudder on the boat, since we want to work with angles [-pi/2, pi/2] we have to offset by -(pi*3)/2
        self.boat.saved_positions = []
        self.boat.saved_linear_velocities = []
        self.boat.saved_angular_velocities = []
        self.boat.fix_in_place = False
        self.boat.time = 0
        self.boat.goal_reached = False
        self.boat.hit_obstacle = False
        
        # The field of view angle, we can probably assume this is smaller than pi (180 degrees)
        self.boat.FOV_angle = math.pi * 0.5 # 0.75 
        self.boat.view_distance = 5
        self.boat.number_of_rays = 50
        self.boat.position_of_camera = (0.25,1.0)
        self.boat.vision = []
        self.boat.camera_mu = 0
        self.boat.camera_sigma = 0.05
        self.boat.vision_array = np.zeros((self.boat.number_of_rays, 2))

        self.obstacles = []

        try:
            obstacles = self.data["obstacles"]
        except KeyError:
            print("No obstacles in level")
            obstacles = []


        for obstacle in obstacles:

            scaled_position = [obstacle["position"][0] * scale, obstacle["position"][1] * scale] 

            buoy = Obstacle(position=scaled_position)

            scaled_radius = obstacle["radius"] * scale

            buoy.body = self.world.CreateStaticBody(position=buoy.initialPosition, shapes=b2CircleShape(pos=buoy.initialPosition, radius=scaled_radius), userData=buoy)
            buoy.body.CreateCircleFixture(radius=scaled_radius)
            self.obstacles.append(buoy)

        self.goals = []

        for goal in self.data["goals"]:

            scaled_position = [goal["position"][0] * scale, goal["position"][1] * scale]
            scaled_width = goal["width"] * scale
            scaled_height = goal["height"] * scale

            goal = Goal(position=scaled_position, width=scaled_width, height=scaled_height)
            goal.body = self.world.CreateStaticBody(position=goal.initialPosition, shapes=b2PolygonShape(vertices=goal.vertices), userData=goal)
            goal.body.fixtures[0].sensor = True

            self.goals.append(goal)


        try: 
            self.highscores = self.data["highscores"]
        except KeyError:
            print("Highscores not found, creating new table")
            self.highscores = []
            self.data["highscores"] = self.highscores


    def save_highscores(self):

        self.data["highscores"] = self.highscores[:10]
        print("saving highscores to file: " + self.filename)

        with open(self.filename, 'w') as outfile:
            json.dump(self.data, outfile)
        

    def add_highscore(self, time, name):

        newEntry = [time, name]

        index = 0

        for i in range(len(self.highscores)):
            entry = self.highscores[i]
            entryTime = entry[0]

            if time < entryTime:
                index = i
                break


        self.highscores.insert(index, newEntry)

        #print(self.highscores)



