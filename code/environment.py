"""
Implements environment of the elevator model.
"""

import constants as const
import control
import numpy as np
import numpy.random as rnd

from time import time, sleep

class Environment:
    """
    Combines all separate parts of an environment.
    """
    def __init__(self):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass


class Building:
    def __init__(self):
        self.num_elevators = 1
        self.num_floors = 4

        # etc.


class Floor:
    def __init__(self):
        self.level = 1
        self.people_waiting = []


class Passenger:
    def __init__(self):
        self.start_floor = 1
        self.target_floor = 1
        # derived from start and target floor
        self.direction = 'UP'
        self.waiting_time = 0


class Elevator:
    def __init__(self):
        self.controller = control.ReinforcementAgent()
        self.pos = 1
        self.direction = 1
        self.action = 1
        self.load = 1
        self.num_passengers = 1
        self.action_in_progress = False


class Event:
    """
    
    """
    pass


class TrafficProfile:
    def __init__(self):
        pass