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
    def __init__(self, num_floors, num_elevators, arrival_rates):
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.floors = [Floor(level, arrival_rates[level]) for level in range(self.num_floors)]
        self.elevators = []
        self.state = State.get_initial_state()
        # etc.

    def update(self):
        """
        Update environment state.
        """
        pass

    def get_floors(self):
        return self.floors

    def __str__(self):
        pass

    def __repr__(self):
        pass


class Floor:
    def __init__(self, level, arrival_rate):
        self.level = level
        self.arrival_rate = arrival_rate
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
    WAITING = 'WAITING'
    BOARDED = 'BOARDED'

    num_passengers_total = 0

    def __init__(self, floor=0, target=0):
        self.floor = floor
        self.target = target
        self.status = Passenger.WAITING
        # derived from start and target floor
        # self.direction = 'UP'
        self.waiting_time = 0
        self.id = Passenger.num_passengers_total
        Passenger.num_passengers_total += 1

    def get_floor(self):
        return self.floor

    def get_target(self):
        return self.target

    def get_status(self):
        return self.status

    def set_floor(self, floor):
        self.floor = floor

    def set_target(self, target):
        self.target = target

    def get_waiting_time(self):
        return self.waiting_time

    def update(self):
        pass


class Elevator:
    UP = 1
    DOWN = -1
    STOPPED = 0

    IDLE = 'IDLE'
    ACCELERATING = 'ACCELERATING'
    DECELERATING = 'DECELERATING'
    FULL_SPEED = 'FULL_SPEED'

    def __init__(self):
        self.controller = control.ReinforcementAgent()
        self.floor = 0
        self.direction = Elevator.STOPPED
        self.current_action = 1
        self.capacity = 20 
        self.num_passengers = 0
        self.action_in_progress = False
        self.status = Elevator.IDLE
        self.constrained = False
        self.state = State()

        # acceleration, velocity and position
        self.acc = acc
        self.vel = vel
        self.pos = pos

        # time, acc, vel, pos, action taken
        self.history = []
        self.decision_time = 0
        self.decision_made = False
        self.acc_update = self.dacc

    def dacc(self, dt):
        return np.cos(C * self.time) * dt * (self.time < ACCEL_TIME)

    def dacc2(self):
        # c = [3.36406177, -6.15411438, 0.84932998, 1.94148245]
        c = [3.51757258, -6.4762952, 0.9575183, 1.94148245]
        x = self.time - self.decision_time

        return (2 * c[0] * x + c[1]) * self.dt

    def acc_(self):
        return np.sin(C * self.time) * (self.time < ACCEL_TIME)

    def dvel(self):
        return self.acc * self.dt

    def vel_(self):
        return - 1 / C**2 * (math.cos(C * self.time) - 1)

    def dpos(self):
        return self.vel * self.dt

    def update(self):
        if self.pos >= 1.83:
            self.acc_update = self.dacc2
            if not self.decision_made:
                # print(self.time)
                # print(self.acc)
                # print(self.vel)
                # print(self.pos)
                self.decision_time = self.time
                self.decision_made = True
        else:
            self.acc_update = self.dacc
        self.acc += self.direction * self.acc_update()
        self.vel += self.dvel()
        self.pos += self.dpos()

    def update(self):
        pass


    def get_legal_actions(self, state):
        # TODO: modelling
        if self.state == Elevator.IDLE:
            pass
        elif self.state == Elevator.MOVING:
            pass

    def get_controller(self):
        return self.controller

    def get_floor(self):
        return floor

    def get_direction(self):
        return self.direction

    def get_action(self):
        return self.current_action

    def get_capacity(self):
        return self.capacity 

    def get_num_passengers(self):
        return self.num_passengers

    def is_action_in_progress(self):
        return self.action_in_progress

    def get_status(self):
        return self.status

    def is_constrained(self):
        return self.constrained

    def get_acceleration(self):
        return self.acc

    def get_velocity(self):
        return self.vel

    def get_position(self):
        return self.pos

    def get_history(self):
        return self.history

    def get_decision_time(self):
        return self.decision_time

    def is_decision_made(self):
        return self.decision_made

    def get_acc_update(self):
        return self.acc_update

    def set_controller(self, controller):
        self.controller = controller

    def set_floor(self, floor):
        self.floor = floor

    def set_direction(self, direction):
        self.direction = direction

    def set_action(self, action):
        self.current_action = action

    def set_num_passengers(self, num_passengers):
        self.num_passengers = num_passengers

    def set_status(self, status):
        self.status = status

    def set_acceleration(self, acceleration):
        self.acc = acceleration

    def set_velocity(self, velocity):
        self.vel = velocity

    def set_position(self, position):
        self.pos = position

    def set_history(self, history):
        self.history = history

    def set_decision_time(self, decision_time):
        self.decision_time = decision_time

    def set_acc_update(self, acc_update):
        self.acc_update = acc_update


class State:
    def __init__(self):
        # the number of remaining up hall calls from floors higher than the current position
        self.hall_up_higher = 0
        self.hall_down_higher = 0
        self.hall_up_lower = 0
        self.hall_down_lower = 0
        # the number of remaining car calls in the current moving direction
        self.current_car_calls = 0
        self.floor = 0
        self.direction = Elevator.STOPPED

    def get_state(self):
        return (self.hall_up_higher, self.hall_down_higher, self.hall_up_lower, self.hall_down_lower, self.current_car_calls, self.floor, self.direction)

    def set_state(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_hall_up_higher(self):
        return self.hall_up_higher

    def get_hall_down_higher(self):
        return self.hall_down_higher

    def get_hall_up_lower(self):
        return self.hall_up_lower

    def get_hall_down_lower(self):
        return self.hall_down_lower

    def get_current_car_calls(self):
        return self.current_car_calls

    def get_floor(self):
        return self.floor

    def get_direction(self):
        return self.floor

    def set_initial_state(self):
        pass

    def set_hall_up_higher(self, num_up_higher):
        self.hall_up_higher = num_up_higher

    def set_hall_down_higher(self, num_down_higher):
        self.hall_down_higher = num_down_higher

    def set_hall_up_lower(self, num_up_lower):
        self.hall_up_lower = num_up_lower

    def set_hall_down_lower(self, num_down_lower):
        self.hall_down_lower = num_down_lower

    def set_current_car_calls(self, current_car_calls):
        self.current_car_calls = current_car_calls

    def set_floor(self, floor):
        self.floor = floor

    def set_direction(self, direction):
        self.floor = direction

    def set_initial_state(self):
        pass


class Action:
    """This class contains static methods relating to actions."""

    def get_legal_actions(self, state):
        pass



class TrafficProfile:
    def __init__(self):
        pass

# q = Queue()
# q.put(1)
# print(q.qsize())
