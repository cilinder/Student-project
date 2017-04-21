#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Box2D import (b2PolygonShape, b2World)

world = b2World()  # default gravity is (0,-10) and doSleep is True
groundBody = world.CreateStaticBody(position=(0, -10),
                                    shapes=b2PolygonShape(box=(50, 10)),
                                    )

# Create a dynamic body at (0, 4)
body = world.CreateDynamicBody(position=(0, 4))

# And add a box fixture onto it (with a nonzero density, so it will move)
box = body.CreatePolygonFixture(box=(1, 1), density=1, friction=0.3)

# Prepare for simulation. Typically we use a time step of 1/60 of a second
# (60Hz) and 6 velocity/2 position iterations. This provides a high quality
# simulation in most game scenarios.
timeStep = 1.0 / 60
vel_iters, pos_iters = 6, 2


