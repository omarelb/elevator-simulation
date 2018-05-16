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
        self.num_waiting = 0

    def get_level(self):
        return self.level

    def get_arrival_rate(self):
        return self.arrival_rate

    def set_arrival_rate(self, rate):
        self.arrival_rate = rate

    def add_passenger(self, passenger):
        self.queue.put(passenger)
        passenger.set_floor(self)
        self.num_waiting += 1


    def remove_passenger(self, passenger):
        self.queue.get()
        self.num_waiting -= 1


    def get_num_waiting(self):
        # qsize may not be reliable
        return self.num_waiting

    
    def __str__(self):
        return '[Level: {}, num_waiting: {}]'.format(self.level, self.num_waiting)

    def __repr__(self):
        return '[Level: {}, num_waiting: {}]'.format(self.level, self.queue)

class Passenger:
    num_passengers_total = 0

    def __init__(self, floor=0, target=0):
        self.floor = floor
        self.target = target
        # derived from start and target floor
        # self.direction = 'UP'
        self.waiting_time = 0
        self.id = Passenger.num_passengers_total
        Passenger.num_passengers_total += 1

    def get_floor(self):
        return self.floor

    def get_target(self):
        return self.target

    def set_floor(self, floor):
        self.floor = floor

    def get_waiting_time(self):
        return self.waiting_time

    def update(self):
        pass


class Elevator:
    IDLE = 'IDLE'
    STOPPED = 'STOPPED'
    MOVING = 'MOVING'

    def __init__(self):
        self.controller = control.ReinforcementAgent()
        self.level = 1
        self.direction = 1
        self.action = 1
        self.load = 1
        self.num_passengers = 1
        self.action_in_progress = False
        self.state = Elevator.IDLE

    def update(self):
        pass


    def get_legal_actions(self, state):
        # TODO: modelling
        if self.state == Elevator.IDLE:
            pass
        elif self.state == Elevator.MOVING:
            pass


class State:
    def __init__(self):
        pass


class TrafficProfile:
    def __init__(self):
        pass

# q = Queue()
# q.put(1)
# print(q.qsize())