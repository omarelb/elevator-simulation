"""
Implements environment of the elevator model.
"""
# user modules
import constants as const

# other modules
import logging
import numpy as np
import numpy.random as rnd
from abc import ABC, abstractmethod
from os.path import join

# from time import time, sleep
from qlearningAgents import ElevatorQAgent
from heuristicAgents import RandomAgent
import events

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler(join(const.LOG_DIR, 'environment.log'), mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class Environment:
    """
    Combines all separate parts of an environment.

    Attributes
    ----------
    traffic_profile :
        object that describes traffic profile of the environment
    num_floors : int
        number of floors in the building, including ground floor
    num_elevators : int
        number of elevators in the building
    floors : list
        floor i can be accessed by floors[i], where ground floor is floor[0].
    elevators : list
        elevator i can be accessed by elevators[i]
    """
    def __init__(self, num_floors=5, num_elevators=1, traffic_profile='down_peak'):
        if traffic_profile == 'down_peak':
            self.traffic_profile = DownPeak(num_floors)
        else:
            self.traffic_profile = DownPeak(num_floors)

        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.floors = [Floor(level) for level in range(self.num_floors)]
        self.elevators = [ElevatorState(self) for _ in range(self.num_elevators)]

    def update(self, simulator):
        """
        Update environment state.
        """
        for elevator_state in self.elevators:
            elevator_state.update(simulator)

    def observe(self, simulator):
        """
        Check state of the environment and see if new events should be made.
        """
        for elevator_state in self.elevators:
            elevator_state.observe(simulator)

    def get_current_state(self):
        """
        Returns the current state of enviornment
        """
        pass

    def get_learning_state(self, elevator_state):
        """
        Return the state used by a learning agent.

        The learning state is mapped to an action by the learning agent.
        It consists of, in the following order:
            - the number of up hall calls above the elevator
            - the number of down hall calls above the elevator
            - the number of up hall calls below the elevator
            - the number of down hall calls below the elevator
            - the number of car calls in the current direction of the elevator
            - current position (floor) of the elevator
            - current direction of the elevator

        Parameters
        ----------
        elevator_state :
            state of an elevator

        Returns
        -------
        tuple
            learning state of the elevator as defined above
        """
        es = elevator_state
        return (self.num_hall_calls(es.floor, down=True, above=True), self.num_hall_calls(es.floor, down=False, above=True),
                self.num_hall_calls(es.floor, down=True, above=False), self.num_hall_calls(es.floor, down=False, above=False),
                elevator_state.num_car_calls(), elevator_state.floor, elevator_state.direction)

    def get_buttons(self, down=False, up=False, down_up=False):
        """
        Return button state of all floors.

        Parameters
        ----------
        down : bool
            If true, return only down button state
        up : bool
            If true, return only up button state
        down_up : bool
            If true return both down and up buttons as a tuple down_buttons, up_buttons 

        Returns
        -------
        tuple
            every element i contains the button state of floor i
        """
        all_buttons = tuple(floor.get_buttons() for floor in self.floors)
        if down_up:
            return tuple(buttons[0] for buttons in all_buttons), tuple(buttons[1] for buttons in all_buttons)
        if down:
            return tuple(buttons[0] for buttons in all_buttons)
        if up:
            return tuple(buttons[1] for buttons in all_buttons)

        return all_buttons

    def no_buttons_pressed(self):
        """
        Return True if no buttons are pressed i.e. no waiting passengers
        """
        buttons_down, buttons_up = self.get_buttons(down_up=True)
        return sum(buttons_down) + sum(buttons_up) == 0

    def num_hall_calls(self, level, down=False, above=False, down_up=False):
        """
        Return number of hall calls relative to a floor level.

        Parameters
        ----------
        level : int
            floor to look around
        down : bool
            If true, look at down hall calls, up hall calls otherwise
        above : bool
            If true, look at hall calls above elevator, below otherwise

        Returns
        -------
        int
            number of up/down hall calls above/below elevator
        """
        buttons = self.get_buttons(down, above, down_up)

        if above:
            return sum(buttons[level + 1:])

        return sum(buttons[:level])

    def is_hall_call(self, level, down=False, above=False, down_up=False):
        """
        Return True if there is at least one hall call in the specified direction.

        Parameters
        ----------
        level : int
            floor to look around
        down : bool
            If true, look at down hall calls, up hall calls otherwise
        above : bool
            If true, look at hall calls above elevator, below otherwise
        down_up : bool
            If true, look at both up and down hall calls

        Returns
        -------
        int
            number of up/down hall calls above/below elevator
        """
        return self.num_hall_calls(level, down_up=True) > 0

    def get_possible_actions(self, elevator_state):
        """
        Return possible actions the agent can take given the state of the environment.

        Actions that can be taken depend on position and status of the elevator, as well
        as state of passengers. If agent has already made a decision, returns the empty tuple.

        If returns empty tuple, no action should be taken. If tuple contains one action, that action
        should be taken. If tuple contains two actions, controller needs to make a decision which action
        to take.

        Returns
        -------
        tuple
            actions which be taken by the agent
        """
        status = elevator_state.status
        num_hall_call_above_up = self.num_hall_calls(elevator_state.floor, down=False, above=True)
        num_hall_call_above_down = self.num_hall_calls(elevator_state.floor, down=False, above=True)
        num_hall_calls_above = num_hall_call_above_up + num_hall_call_above_down

        # cannot take new action when action is still in progress or no passengers in system
        if elevator_state.is_action_in_progress() or self.no_buttons_pressed():
            return ()

        if status == ElevatorState.IDLE:
                # heuristic: prefer to move up
                if num_hall_calls_above > 0:
                    return (ElevatorState.MOVE_UP,)
                return (ElevatorState.MOVE_DOWN,)

        if status == ElevatorState.BOARDING:
                if elevator_state.direction == ElevatorState.UP:
                    # TODO: Check order of calling this and updating passengers on this floor
                    # if still hall calls in current direction, can not turn
                    if self.is_hall_call(elevator_state.floor, above=True, down_up=True):
                        return (ElevatorState.MOVE_UP,)
                elif elevator_state.direction == ElevatorState.DOWN:
                    if self.is_hall_call(elevator_state.floor, above=False, down_up=True):
                        return (ElevatorState.MOVE_DOWN,)

        last_floor_pos = self.floors[elevator_state.floor].pos
        elevator_dist = abs(elevator_state.motion.pos - last_floor_pos)
        if elevator_state.status == ElevatorState.ACCELERATING:
            # decision point, half of floor height
            if elevator_dist >= const.FLOOR_HEIGHT / 2 - const.GENERAL_EPS:
                stop_target = elevator_state.next_floor(self.floors)
                elevator_state.stop_target = stop_target.level
                # cannot pass floor if a passenger wants to get off there
                if stop_target.level in elevator_state.car_calls():
                    return (ElevatorState.STOP,)
                if elevator_dist.direction == ElevatorState.UP:
                    if stop_target.num_down() > 0 and num_hall_call_above_up == 0:
                        return (ElevatorState.STOP,)
                if not elevator_state.is_passenger_next_floor(self.floors):
                    # no passenger wants to get on or off next floor -> force continue
                    return (ElevatorState.CONTINUE,)
                else:
                    return (ElevatorState.STOP, ElevatorState.CONTINUE)

        if elevator_state.status == ElevatorState.FULL_SPEED:
            if elevator_dist >= const.ACCEL_DECISION_DIST - const.GENERAL_EPS:
                stop_target = elevator_state.next_floor(self.floors, amount=2).level
                elevator_state.stop_target = stop_target
                if (stop_target in elevator_state.car_calls() or
                    elevator_state.is_passenger_next_floor(self.floors, amount=2)):
                    # passenger wants to go to next floor or passenger at next floor -> force stop action
                    return (ElevatorState.STOP,)
                else:
                    return (ElevatorState.STOP, ElevatorState.CONTINUE)
        
        return ()

    def process_actions(self, simulator):
        for elevator in self.elevators:
            possible_actions = self.get_possible_actions(elevator)

            # a constrained decision is made
            if len(possible_actions) == 1:
                simulator.insert(events.ElevatorActionEvent(simulator.now(), elevator, possible_actions[0]))
            elif len(possible_actions) == 2:
                simulator.insert(events.ElevatorControlEvent(simulator.now(), elevator))

    def complete_actions(self, simulator):
        for elevator in self.elevators:
            elevator.complete_action(simulator)

    def do_action(self, action, elevator_id=0):
        """
        Performs the given action in the current environment state and updates the environment.
        """
        self.elevators[elevator_id].do_action(action)

    def reset(self):
        """
          Resets the current state to the blank state
        """
        init_state = self.get_initial_state()
        for elevator in self.elevators:
            elevator.reset(init_state)

        for floor in self.floors:
            floor.reset()

    def get_initial_state(self):
        """
        Return a dictionary containing initial state parameters.

        Returns
        -------
        dict
            containing initial state parameters
        """
        return {'start_floor': 0, 'start_direction': ElevatorState.STOPPED, 'capacity': 20,
                'acc': 0, 'vel': 0, 'pos': 0}

    def is_terminal(self):
        """
          Has the enviornment entered a terminal
          state? This means there are no successors
        """
        state = self.get_current_state()
        actions = self.get_possible_actions(state)
        return len(actions) == 0

    def end_episode(self):
        """
        Handle everything that needs to be handled to end the episode.
        """
        pass

    def __str__(self):
        res = 'elevator positions - '
        for i in range(self.num_elevators):
            res += '{}: {}, '.format(i, self.elevators[i].floor)
        return res[:-2]

    def __repr__(self):
        return 'Environment(num_floors={}, num_elevators={}, traffic_profile={})'.format(
            self.num_floors, self.num_elevators, self.traffic_profile)


class ElevatorState:
    """
    Represents state of an elevator.

    Attributes
    ----------
    id : int
        elevator id
    environment :
        state of the environment
    controller :
        agent controlling the elevator
    floor : int
        floor the elevator is at. if elevator is moving between floor, value is the last floor
        the elevator passed/stopped. NOT A FLOOR OBJECT
    direction : int
        current direction of the elevator: up, down, or stopped
    current_action :
        current action the elevator is taking
    capacity : int
        maximum number of people that can get into the elevator at the same time.
    passengers : dict
        dictionary of lists mapping floor to passengers traveling to that floor
    status : int
        indicate elevator status: accelerating, decelerating, full speed or idle
    constrained : bool
        true if current action taken was constrained i.e. the elevator had no choice.
    motion :
        object handling elevator motion
    history : list
        containing ?
    decision_time : float
        time at which last decision was made
    decision_made : bool
        true if a decision was made (?)
    stop_target : int
        floor number of floor to stop at if action is stop
    """
    # elevator direction
    UP = 1
    DOWN = -1
    STOPPED = 0  # and status

    # elevator status
    IDLE = 2
    ACCELERATING = 3
    FULL_SPEED_DECELERATING = 4  # decelerating after reaching full speed
    ACCEL_DECELERATING = 5  # decelerating after just accelerating
    FULL_SPEED = 6  # moving at full speed
    BOARDING = 7  # stopped at floor and passengers are getting in/out

    # elevator actions
    STOP = 8
    CONTINUE = 9
    NO_ACTION = 10
    MOVE_UP = 11
    MOVE_DOWN = 12

    num_elevators = 0

    def __init__(self, environment, controller=RandomAgent(), floor=0, direction=None,
                 current_action=None, capacity=20, status=None, constrained=False,
                 acc=0, vel=0, pos=0, history=None, decision_time=None, decision_made=False):
        self.id = ElevatorState.num_elevators
        ElevatorState.num_elevators += 1
        self.environment = environment
        self.controller = controller
        self.floor = floor
        self.direction = direction if direction else ElevatorState.STOPPED
        self.current_action = current_action if current_action else ElevatorState.NO_ACTION
        self.capacity = capacity
        self.passengers = {i: [] for i in range(self.environment.num_floors)}
        self.status = status if status else ElevatorState.IDLE
        self.constrained = constrained
        self.motion = ElevatorMotion(self, acc, vel, pos)
        self.history = history if history else []
        self.decision_time = decision_time
        self.decision_made = decision_made
        self.stop_target = -1

    def is_constrained(self):
        """Return True if current action choice was constrained."""
        return self.constrained

    def is_decision_made(self):
        """
        Return True if elevator has made a decision at the moment
        """
        return self.decision_made

    def capacity_left(self):
        """
        Return number of passengers that can still fit in the elevator right now.
        """
        return self.capacity - self.num_passengers()

    def is_full(self):
        """
        Return True if number of passengers has reached elevator capacity.
        """
        return self.capacity_left == 0

    def add_passengers(self, passengers):
        """
        Add given group of passengers to elevator

        Parameters
        ----------
        passengers : list

        passengers to be added
        """
        for passenger in passengers:
            self.add_passenger(passenger)

    def add_passenger(self, passenger):
        """
        Add given passenger to elevator

        Parameters
        ----------
        passenger : Passenger
        """
        passenger.enter_elevator(self)

    def car_calls(self):
        """
        Return remaining car calls in current direction, sorted in increasing floor order.
        """
        calls = []

        for target_floor, passengers in self.passengers.items():
            # someone going to that floor
            if passengers:
                if ((self.direction == ElevatorState.UP and target_floor > self.floor) or
                    (self.direction == ElevatorState.DOWN and target_floor < self.floor)):
                    calls.append(target_floor)

        return calls

    def num_car_calls(self):
        """
        Return number of remaining car calls in current direction.
        """
        return len(self.car_calls())

    def is_passenger_next_floor(self, floors, amount=1):
        """
        Return True if a passenger is waiting on the next floor.

        Parameters
        ----------
        amount : int
            number of floors to look ahead
        """
        next_floor = self.next_floor(floors, amount)
        # if self.direction == ElevatorState.UP:
        return next_floor.num_up() + next_floor.num_down() > 0
        # else:

    def next_floor(self, floors, amount=1):
        """
        Return next floor the elevator is moving to.
        
        Parameters
        ----------
        amount : int
            number of floors to look ahead

        Returns
        -------
        floor object
            next (amount) floor
        """
        return floors[self.floor + self.direction * amount]

    def num_passengers(self):
        """
        Return number of passengers in the elevator.
        """
        res = 0
        for _, passengers in self.passengers.items():
            res += len(passengers)

        return res

    def is_empty(self):
        """
        Return True if there are no passengers in the elevator.
        """
        return self.num_passengers() == 0

    def passengers_as_list(self):
        """
        Return list of passengers instead of dict for iteration purposes.
        """
        res = []
        for _, passengers in self.passengers.items():
            res += passengers

        return res

    def update(self, simulator):
        """
        Update elevator state. Called by the environment every simulator loop.
        """
        self.motion.update(simulator)
        if self.motion.vel >= const.MAX_SPEED - const.GENERAL_EPS:
            self.status = ElevatorState.FULL_SPEED
            self.motion.vel = const.MAX_SPEED

        # update elevator's floor when it crosses the floor
        if abs(self.motion.pos - simulator.environment.floors[self.floor].pos) >= const.FLOOR_HEIGHT - const.GENERAL_EPS:
            self.floor = self.next_floor(simulator.environment.floors).level
            self.motion.pos = simulator.environment.floors[self.floor].pos

    def observe(self, simulator):
        # position and action checks
        # if self
        pass

    def do_action(self, simulator, action):
        """
        Update status of elevator according to given action.

        Parameters
        ----------
        action : int
            action elevator is taking
        """
        if action == ElevatorState.MOVE_UP:
            self.direction = ElevatorState.UP
            self.status = ElevatorState.ACCELERATING
            logger.info('elevator %d status set to accelerating (up)', self.id)
        elif action == ElevatorState.MOVE_DOWN:
            self.direction = ElevatorState.DOWN
            self.status = ElevatorState.ACCELERATING
            logger.info('elevator %d status set to accelerating (down)', self.id)
        elif action == ElevatorState.STOP:
            logger.info('elevator %d plans to stop at floor %d', self.id, self.stop_target)
            if self.status == ElevatorState.FULL_SPEED:
                self.status = ElevatorState.FULL_SPEED_DECELERATING
                logger.info('elevator %d status set to decelerating from full speed', self.id)
            else:
                self.status = ElevatorState.ACCEL_DECELERATING
                logger.info('elevator %d status set to decelerating from accelerating', self.id)

    def complete_action(self, simulator):
        """
        Complete an action by updating elevator status.
        """
        if self.current_action == ElevatorState.STOP and self.floor == self.stop_target:
        # elevator arrives at floor
            self.status = ElevatorState.BOARDING
            self.current_action == ElevatorState.NO_ACTION
            simulator.environment.floors[self.floor].board_passengers(simulator, self)
            # TODO: adjust dacc so that it is 0 whenever boarding
        elif self.current_action == ElevatorState.MOVE_UP or self.current_action == ElevatorState.MOVE_DOWN:
            self.current_action == ElevatorState.NO_ACTION
        
        # TODO: ADD TIME TO BOARD PASSENGERS
        # TODO: DIFFERENT WAY TO INDICATE ACTION IN PROGRESS

    def is_action_in_progress(self):
        return self.current_action != ElevatorState.NO_ACTION

    def reset(self, initial_state):
        """
        Reset the elevator to a given state.

        Parameters
        ----------
        initial_state : dict
            contains parameters defining the initial state
        """
        # self.controller = initial_state['controller']
        self.floor = initial_state['start_floor']
        self.direction = initial_state['start_direction']
        self.current_action = ElevatorState.NO_ACTION
        self.capacity = initial_state['capacity']
        # dictionary of lists mapping floor to passengers traveling to that floor
        self.passengers = {i: [] for i in range(self.environment.num_floors)}
        self.status = ElevatorState.IDLE
        # True if current elevator action was constrained
        self.constrained = False
        # acceleration, velocity and position
        self.motion.acc = initial_state['acc']
        self.motion.vel = initial_state['vel']
        self.motion.pos = initial_state['pos']
        logger.debug('environment reset')

    def __repr__(self):
        return 'ElevatorState(environment, controller={}, floor={}, direction={},\
current_action={}, capacity={}, action_in_progress={},\
status={}, constrained={}, acc={}, vel={}, pos={}, decision_time={},\
decision_made={})'.format(self.controller, self.floor, self.direction,
                 self.current_action, self.capacity, self.is_action_in_progress(), self.status,
                 self.constrained, self.motion.acc, self.motion.vel, self.motion.pos, self.decision_time,
                 self.decision_made)


class ElevatorMotion:
    """
    Contains information regarding motion of the elevator as well as how to update it.

    Attributes
    ----------
    elevator_state :
        state of the elevator
    acc : float
        elevator acceleration in m/s^2
    vel : float
        elevator velocity in m/s
    pos : float
        elevator position in m
    acc_update : func
        function that dictates how elevator acceleration is updated. depends on state, action, etc.
    reference_time: float
        time in s which acceleration function evaluates as 0
    """
    def __init__(self, elevator_state, acc=0, vel=0, pos=0, reference_time=0):
        self.elevator_state = elevator_state
        self.acc = acc
        self.vel = vel
        self.pos = pos
        self.reference_time = reference_time

    def dacc(self, simulator):
        """
        Return the change in acceleration for a given timestep.

        The change in acceleration depends on state of the elevator,
        action taken by the elevator.

        Change in acceleration is approximated: da(t) approx a'(t)dt

        Parameters
        ----------
        simulator :
            simulator object which has information about time and timestep size
        """
        t = simulator.now() - self.reference_time

        if self.elevator_state.status == ElevatorState.ACCELERATING:
            res = np.cos(const.ACCEL_CONST * t)
        elif self.elevator_state.status == ElevatorState.ACCEL_DECELERATING:
            res = (2 * const.ACCEL_DECEL[0] * t + const.ACCEL_DECEL[1])
        elif self.elevator_state.status == ElevatorState.FULL_SPEED_DECELERATING:
            res = - np.cos(const.ACCEL_CONST * t)
        else:
            return 0

        return res * simulator.time_step

    def dvel(self, simulator):
        """
        Return change of velocity in a single time step.
        """
        return self.acc * simulator.time_step

    # def vel_(self):
    #     return - 1 / C**2 * (math.cos(C * self.time) - 1)

    def dpos(self, simulator):
        """
        Return change of position in a single time step.
        """
        return self.vel * simulator.time_step

    def update(self, simulator):
        """
        Update elevator motion state.
        
        Acceleration is set to zero if its status is idle or full speed, and if
        the direction is STOPPED.
        """
        if (self.elevator_state.status == ElevatorState.IDLE or
            self.elevator_state.direction == ElevatorState.STOPPED or
                self.elevator_state.status == ElevatorState.FULL_SPEED):
            self.acc = 0
        else:
            self.acc += self.elevator_state.direction * self.dacc(simulator)

        self.vel += self.dvel(simulator)
        self.pos += self.dpos(simulator)

    def __repr__(self):
        return 'ElevatorMotion(elevator_state, acc={}, vel={}, pos={}, reference_time={})'.format(
            self.acc, self.vel, self.pos, self.reference_time)


class Floor:
    """
    Represents a floor in a building.

    Attributes
    ----------
    level : int
        floor number
    pos : float
        vertical position in meters
    arrival_rate : float
        mean = 1 / lambda of poisson arrival process
    passengers_up : list
        contains passengers on floor going up
    passengers_down : list
        contains passengers on floor going down
    up : bool
        up hall button on this floor True if on
    down : bool
        down hall button on this floor True if on
    """
    def __init__(self, level):
        self.level = level
        self.pos = const.FLOOR_HEIGHT * self.level
        self.passengers_up = []
        self.passengers_down = []
        self.up = False
        self.down = False

    def add_passenger(self, passenger):
        """
        Add passenger to floor.

        Parameters
        ----------
        passenger : Passenger
        """
        if passenger.going_up():
            self.passengers_up.append(passenger)
        else:
            self.passengers_down.append(passenger)

        self.update_button(target=passenger.target)

    def all_passengers(self):
        """
        Return passengers going down and up

        Returns
        -------
        all passengers : list
            combined list of down and up passengers, in that order.
        """
        return self.passengers_down + self.passengers_up

    def num_waiting(self):
        """
        Return number of passengers waiting on the floor
        """
        return len(self.all_passengers())

    def num_up(self):
        """
        Return number of passengers going up on this floor
        """
        return len(self.passengers_up)

    def num_down(self):
        """
        Return number of passengers going down on this floor
        """
        return len(self.passengers_down)

    def update_button(self, passenger=True, target=None, elevator_direction=None):
        """
        Update button state given an arriving passenger or arriving elevator

        To be called by arriving elevator or passenger.

        Parameters
        ----------
        passenger : bool
            true if button is updated by passenger arriving
            false if button is updated by elvator arriving

        target : int
            if button is updated by passenger arriving, indicates target floor

        elevator_direction : int
            indicates direction if arriving elevator
        """
        # if no passengers: button - false -> true, if passengers already: true -> true
        if passenger:
            if target < self.level and not self.down:
                self.down = True
                logger.info('down button %d turns on', self.level)
            elif target > self.level and not self.up:
                self.up = True
                logger.info('up button %d turns on', self.level)
        else:
            if elevator_direction == ElevatorState.UP:
                self.up = False
                logger.info('up button %d turns off', self.level)
            else:
                self.down = False
                logger.info('down button %d turns off', self.level)

    def get_buttons(self):
        """
        Return floor buttons state.

        Returns
        -------
        tuple
            state of the down and up buttons in that order
        """
        return (self.down, self.up)

    def waiting_time(self, simulator):
        """
        Return sum of passenger waiting times on this floor.
        """
        result = 0
        for passenger in self.all_passengers():
            result += passenger.waiting_time(simulator)
        return result

    def board_passengers(self, simulator, elevator_state):
        """
        Transfer passengers from floor to elevator and vice versa

        If capacity of elevator is not enough to accomodate passengers, transfer only first arrived
        passengers until elevator capacity is full.

        Called when elevator stops at floor.
        """
        now = simulator.now()
        passengers_off = elevator_state.passengers[elevator_state.floor]
        for passenger in passengers_off:
            # TODO: MAKE SURE THIS IS DONE AFTER ELEVATOR FLOOR HAS BEEN UPDATED CORRECTLY
            simulator.insert(events.PassengerTransferEvent(now, passenger, elevator_state, to_elevator=False))
        capacity_left = elevator_state.capacity_left() + len(passengers_off)
        if elevator_state.direction == ElevatorState.UP:
            if capacity_left < self.num_up():
                passengers_boarding = self.passengers_up[:capacity_left]
                # del self.passengers_up[:capacity_left]
            else:
                passengers_boarding = self.passengers_up[:]
                # del self.passengers_up[:]
        else:
            if capacity_left < self.num_down():
                passengers_boarding = self.passengers_down[:capacity_left]
                # del self.passengers_down[:capacity_left]
            else:
                passengers_boarding = self.passengers_down[:]
                # del self.passengers_down[:]

        now = simulator.now()
        for i, passenger in enumerate(passengers_boarding):
            # TODO: time from truncated erlang instead of 1 second
            simulator.insert(events.PassengerTransferEvent(now + i + 1, passenger, elevator_state, to_elevator=True))

        simulator.insert(events.DoneBoardingEvent(now + i + 1 + const.GENERAL_EPS, elevator_state))
        # elevator_state.add_passengers(passengers_boarding)

    def reset(self):
        self.passengers_up = []
        self.passengers_down = []
        self.up = False
        self.down = False

    def update(self):
        # self.update_button()
        pass

    def __lt__(self, other):
        return self.level < other.level

    def __gt__(self, other):
        return self.level > other.level
    
    def __eq__(self, other):
        return self.level == other.level

    def __str__(self):
        return '[Level: {}, num waiting: {}]'.format(self.level, self.num_waiting())

    def __repr__(self):
        return 'Floor(level={})'.format(self.level)


class Passenger:
    """
    Represents a passenger.

    Parameters
    ----------
    target : int
        indicates floor to which the passenger wants to travel
    environment :
        environment
    floor :
        floor on which the passenger is waiting
    elevator_state :
        elevator the passenger enters
    status : int
        passenger status: either waiting or boarded
    arrival_time : float
        time when the passenger arrived on the floor
    boarding_time : float
        time when the passenger boarded an elevator
    id : int
        unique passenger identifier
    """
    WAITING = 0
    BOARDED = 1

    num_passengers_total = 0

    def __init__(self, floor, target=None):
        """
        Initialize passenger and immediately handle updating the floor state
        """
        if target:
            self.target = target
        self.status = Passenger.WAITING
        self.boarded_time = 0
        self.id = Passenger.num_passengers_total
        Passenger.num_passengers_total += 1
        self.floor = floor

    def arrive_at_floor(self, simulator):
        logger.info('passenger %d arrives at floor %d', self.id, self.floor.level)
        self.arrival_time = simulator.now()
        self.target = self.choose_target(simulator.environment)
        self.floor.add_passenger(self)

    def system_time(self, simulator):
        """Return time passenger has been in system."""
        return simulator.now() - self.arrival_time

    def waiting_time(self, simulator):
        """Return time passenger has/had waited for an elevator."""
        return simulator.now() - self.arrival_time - self.boarded_time * (self.status == Passenger.BOARDED)

    def boarding_time(self, simulator):
        """Return time passenger has been in elevator"""
        return (simulator.now() - self.boarded_time) * (self.status == Passenger.BOARDED)

    def update_time(self):
        """Update waiting or boarding time."""
        pass

    def going_up(self):
        """Return True if passenger is going up."""
        return self.target > self.floor.level

    def going_down(self):
        """Return True if passenger is going up."""
        return self.target < self.floor.level
    
    def choose_target(self, environment):
        """
        Return target floor according to current traffic
        """
        target = environment.traffic_profile.choose_target(self.floor)
        logger.info('passenger %d chooses floor %d', self.id, target)
        return target

    def enter_elevator(self, elevator_state):
        """
        Change state of passenger when entering elevator.

        Called by elevator or floor.

        Changes passenger status and adds passenger to elevator's queue.

        Parameters
        ----------
        elevator_state :
            called by the elevator defined in that elevator state
        """
        self.status = Passenger.BOARDED

        elevator_state.passengers[self.target].append(self)
        logger.info('passenger %d enters elevator %d', self.id, elevator_state.id)

    def exit_elevator(self, elevator_state):
        """
        Remove passenger from elevator and system when exiting elevator.

        Parameters
        ----------
        elevator_state :
            called by the elevator defined in that elevator state
        """
        elevator_state.passengers[self.target].remove(self)
        logger.info('passenger %d exits elevator %d', self.id, elevator_state.id)

    def update(self):
        pass

    def __repr__(self):
        return 'Passenger(floor={}, target={})'.format(self.floor, self.target)


class Action:
    """This class contains static methods relating to actions."""

    def get_legal_actions(self, state):
        pass


class TrafficProfile(ABC):
    def __init__(self, num_floors, interfloor=0):
        """
        Base class for traffic profiles such as DownPeak, UpPeak, etc.

        Attributes
        ----------
        interfloor : float
            \in [0, 1], percentage of interfloor travel in terms of total arrival rate
        """
        self.num_floors = num_floors
        self.interfloor = interfloor

    @abstractmethod
    def choose_target(self, passenger):
        pass


class DownPeak(TrafficProfile):
    def __init__(self, num_floors, interfloor=0):
        """
        Attributes
        ----------
        target_floor : int
            floor to which most passengers are headed
        arrival_rates: tuple
            mean number of passengers during a typical afternoon business hour
        """
        super().__init__(num_floors, interfloor)
        self.target_floor = 0

    def choose_target(self, floor):
        # with prob `interfloor' choose floor != target else choose target
        if rnd.rand() < self.interfloor:
            possible_floors = [pos_floor for pos_floor in range(self.num_floors) if pos_floor not in (0, floor)]
            target = rnd.choice(possible_floors)
        else:
            target = self.target_floor

        return target

    def arrival_rate(self, time):
        """
        Return mean number of people arriving in this timeframe.

        Parameters
        ----------
        time : float
            time in milliseconds after starting simulation
        """
        minutes_in = time / const.SECONDS_PER_MINUTE
        index = int(minutes_in / const.MINUTES_PER_TIME_INTERVAL)
        return const.DOWNPEAK_RATES[index]

    def __repr__(self):
        return 'DownPeak(num_floors={}, interfloor={})'.format(self.num_floors, self.interfloor)