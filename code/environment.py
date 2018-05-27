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
        # initialize floors
        self.floors = [Floor(level, arrival_rates[level]) for level in range(self.num_floors)]
        self.elevators = []
        self.state = State.get_initial_state()
        # etc.

    def update(self):
        """
        Update environment state.
        """
        pass

    def getCurrentState(self):
        """
        Returns the current state of enviornment
        """
        pass

    def getPossibleActions(self, state):
        """
          Returns possible actions the agent
          can take in the given state. Can
          return the empty list if we are in
          a terminal state.
        """
        pass

    def doAction(self, action):
        """
          Performs the given action in the current
          environment state and updates the enviornment.

          Returns a (reward, nextState) pair
        """
        pass

    def reset(self):
        """
          Resets the current state to the start state
        """
        pass

    def isTerminal(self):
        """
          Has the enviornment entered a terminal
          state? This means there are no successors
        """
        state = self.getCurrentState()
        actions = self.getPossibleActions(state)
        return len(actions) == 0

    def get_floors(self):
        return self.floors

    def __str__(self):
        pass

    def __repr__(self):
        pass


class Floor:
    def __init__(self, level, arrival_rate):
        self.level = level
        self.pos = const.FLOOR_HEIGHT * self.level
        self.arrival_rate = arrival_rate
        self.queue = Queue(maxsize=0)
        self.num_waiting = 0

    def get_level(self):
        return self.level

    def get_pos(self):
        return self.pos

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


    def __lt__(self, other):
        return self.level < other.level 

    def __gt__(self, other):
        return self.level > other.level 
    
    def __eq__(self, other):
        return self.level == other.level 


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
    # elevator direction
    UP = 1
    DOWN = -1
    STOPPED = 0

    # elevator status
    IDLE = 2
    ACCELERATING = 3
    DECELERATING = 4
    FULL_SPEED = 5
 
    # elevator actions
    STOP = 6
    CONTINUE = 7
    NO_ACTION = 8

    def __init__(self, controller=control.ReinforcementAgent(), floor=0, direction=Elevator.STOPPED,
                 current_action=Elevator.NO_ACTION, capacity=20, num_passengers=0, action_in_progress=False, 
                 status=Elevator.IDLE, constrained=False, acc=0, vel=0, pos=0, history=None,
                 decision_time=None, decision_made=False, acc_update=None):
        self.controller = controller
        self.floor = floor
        self.direction = direction
        self.current_action = current_action
        self.capacity = capacity
        self.num_passengers = num_passengers
        self.action_in_progress = action_in_progress
        self.status = status
        # True if current elevator action was constrained
        self.constrained = constrained
        # acceleration, velocity and position
        self.acc = acc
        self.vel = vel
        self.pos = pos

        # time, acc, vel, pos, action taken
        self.history = history
        self.decision_time = decision_time
        self.decision_made = decision_made
        self.acc_update = acc_update

    def dacc(self, time, dt):
        """
        Return the change in acceleration for a given timestep.

        Change in acceleration is approximated: da(t) approx a'(t)dt
        """
        # TODO: condition return depending on status (and position?)
        return np.cos(const.ACCEL_CONST * time) * dt 

    def dacc2(self, dt):
        # c = [3.36406177, -6.15411438, 0.84932998, 1.94148245]
        x = self.time - self.decision_time

        return (2 * const.ACCEL_DECEL[0] * x + const.ACCEL_DECEL[1]) * dt


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


    def get_legal_actions(self, state):
        # TODO: modelling
        if self.state == Elevator.IDLE:
            pass
        elif self.state == Elevator.MOVING:
            pass

    def get_controller(self):
        return self.controller

    def get_floor(self):
        return self.floor

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

    def get_motion(self):
        return (self.acc, self.vel, self.pos)

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


    @staticmethod
    def get_initial_state():
        return (0, 0, 0, 0, 0, 0, Elevator.STOPPED)
        

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
