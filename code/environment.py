"""
Implements environment of the elevator model.
"""

# user modules
import constants as const
import control

# other modules
import numpy as np
import numpy.random as rnd
from queue import Queue

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

    def update(self):
        """
        Update environment state.
        """



class Building:
    def __init__(self, num_floors, num_elevators, arrival_rates):
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.floors = [Floor(level, arrival_rates[level]) for level in range(self.num_floors)]

        # etc.

    def update(self):
        pass

    def get_floors(self):
        return self.floors


class Floor:
    def __init__(self, level, arrival_rate):
        self.level = level
        self.arrival_rate = arrival_rate
        # TODO: Max size or not? -> modelling
        self.queue = Queue(maxsize=0)

    def get_level(self):
        return self.level

    def get_arrival_rate(self):
        return self.arrival_rate

    def set_arrival_rate(self, rate):
        self.arrival_rate = rate

    def add_passenger(self, passenger):
        self.queue.put(passenger)
        passenger.set_floor(self)

    def get_num_passengers(self):
        # qsize may not be reliable
        return len(self.queue.qsize())


class Passenger:
    def __init__(self, target):
        self.floor = 1
        self.target = 1
        # derived from start and target floor
        self.direction = 'UP'
        self.waiting_time = 0

    def get_floor(self):
        return self.floor

    def get_target(self):
        return self.target

    def set_floor(self, floor):
        self.floor = floor

    def get_waiting_time(self):
        return self.waiting_time


class Elevator:
    def __init__(self):
        self.controller = control.ReinforcementAgent()
        self.pos = 1
        self.direction = 1
        self.action = 1
        self.load = 1
        self.num_passengers = 1
        self.action_in_progress = False


class TrafficProfile:
    def __init__(self):
        pass

# q = Queue()
# q.put(1)
# print(q.qsize())