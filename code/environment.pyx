"""
Implements environment of the elevator model.
"""
# user modules
import constants as const

# other modules
import logging
import csv
import random
import math
from abc import ABC, abstractmethod
from os.path import join
from io import StringIO

# from time import time, sleep
from qlearningAgents import ElevatorQAgent
from heuristicAgents import RandomAgent, BestFirstAgent
import events

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler(join(const.LOG_DIR, 'environment.log'), mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


cdef class Environment:
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
        floor i can be accessed by floors[i], where ground floor is floor[0]
    elevators : list
        elevator i can be accessed by elevators[i]
    last_accumulator_event_time :
        time when a passenger arrival, passenger transfer or elevator control event occurred.
        necessary for updating accumulated costs for reinforcement agents
    passenger_times : list
        contains waiting time for all passengers who have exited the system as tuples
    write_files : bool
        indicates whether existing files should be overwritten or updated. set to false when
        testing flag --testing is true
    """
    cdef public int num_floors, num_elevators
    cdef public float last_accumulator_event_time
    cdef public object floors, elevators, passenger_times, traffic_profile
    cdef public bint write_files

    def __init__(self, int num_floors=5, int num_elevators=1, object traffic_profile='DownPeak',
                 float interfloor=0.1, **args):
        if traffic_profile == 'DownPeak':
            self.traffic_profile = DownPeak(num_floors, interfloor)
        else:
            self.traffic_profile = DownPeak(num_floors, interfloor)

        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.floors = [Floor(level) for level in range(self.num_floors)]
        self.elevators = [ElevatorState(environment=self, index=i, **args) for i in range(self.num_elevators)]
        self.last_accumulator_event_time = 0
        self.passenger_times = []
        self.write_files = args['write_files']

    def start_episode(self, object simulator):
        """
        Resets elevator and floor states and generates first passenger arrival events.

        Called by simulator to start an episode.
        """
        logger.debug('initializing environment')
        self.reset(simulator)

    def reset(self, object simulator):
        self.last_accumulator_event_time = 0
        self.passenger_times = []
        for floor in self.floors[1:]:
            floor.reset()
            events.PassengerArrivalEvent(simulator.now(), floor).generate(simulator)

        for elevator in self.elevators:
            elevator.reset()
            elevator.controller.start_episode()

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

    cpdef get_learning_state(self, elevator_state):
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
        return (self.num_hall_calls(es.floor, down=True, above=True, down_up=False),
                self.num_hall_calls(es.floor, down=False, above=True, down_up=False),
                self.num_hall_calls(es.floor, down=True, above=False, down_up=False),
                self.num_hall_calls(es.floor, down=False, above=False, down_up=False),
                es.num_car_calls(), es.floor, es.direction)

    def get_buttons(self, bint down=False, bint down_up=False):
        """
        Return button state of all floors.

        Parameters
        ----------
        down : bool
            If true, return only down button state, only up button state otherwise
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
        return tuple(buttons[1] for buttons in all_buttons)

    def no_buttons_pressed(self):
        """
        Return True if no buttons are pressed i.e. no waiting passengers
        """
        buttons_down, buttons_up = self.get_buttons(down_up=True)
        return sum(buttons_down) + sum(buttons_up) == 0

    def num_hall_calls(self, level, down=False, above=False, down_up=True):
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
        down_up : bool
            If true, look at both up and down hall calls

        Returns
        -------
        int
            number of up/down hall calls above/below elevator
        """
        buttons = self.get_buttons(down, down_up)
        if down_up:
            if above:
                return sum(buttons[0][level + 1:]) + sum(buttons[1][level + 1:])
            return sum(buttons[0][:level]) + sum(buttons[1][:level])

        if above:
            return sum(buttons[level + 1:])

        return sum(buttons[:level])

    def is_hall_call(self, level, down=False, above=False, down_up=True):
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
        return self.num_hall_calls(level, down, above, down_up) > 0

    def get_passengers_system(self):
        """
        Return all passengers in system.

        Returns
        -------
        list
            passenger objects representing passengers in elevators and floors.
        """
        return self.get_passengers_boarded() + self.get_passengers_waiting()

    def get_passengers_waiting(self):
        """
        Return passengers waiting on every floor.

        Returns
        -------
        list
            passenger objects representing passengers waiting
        """
        result = []
        for floor in self.floors:
            result += floor.all_passengers()
        return result

    def get_passengers_boarded(self):
        """
        Return passengers boarded in elevators.

        Returns
        -------
        list
            passenger objects representing passengers in elevators
        """
        result = []
        for elevator in self.elevators:
            result += elevator.passengers_as_list()
        return result

    cdef get_possible_actions(self, object elevator_state):
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
        cdef int status, num_pass_up, num_pass_down, amount, stop_target
        status = elevator_state.status

        # cannot take new action when action is still in progress or no passengers in system
        if (elevator_state.is_action_in_progress() or status == ElevatorState.BOARDING or
            (self.no_buttons_pressed() and elevator_state.is_empty())):
            return ()

        if status == ElevatorState.IDLE:
                # heuristic: prefer to move up
                if self.is_hall_call(elevator_state.floor, above=True, down_up=True):
                    return (ElevatorState.MOVE_UP,)
                return (ElevatorState.MOVE_DOWN,)

        if status == ElevatorState.DONE_BOARDING:
            # not have both passengers going up and down
            num_pass_up = elevator_state.num_passengers_up()
            num_pass_down = elevator_state.num_passengers_down()
            assert num_pass_up * num_pass_down == 0, 'elevator contains passengers going up AND down'
            # has to service car calls in current direction
            if num_pass_down > 0:
                return (ElevatorState.MOVE_DOWN,)
            elif num_pass_up > 0:
                return (ElevatorState.MOVE_UP,)
            else:
                if self.is_hall_call(elevator_state.floor, above=True, down_up=True):
                    return (ElevatorState.MOVE_UP,)
                if self.is_hall_call(elevator_state.floor, above=False, down_up=True):
                    return (ElevatorState.MOVE_DOWN,)
            return ()

        if elevator_state.is_decision_point(self):
            if elevator_state.status == ElevatorState.ACCELERATING:
                amount = 1
            elif elevator_state.status == ElevatorState.FULL_SPEED:
                amount = 2
            stop_target = elevator_state.next_floor(self.floors, amount=amount).level
            elevator_state.stop_target = stop_target
            if stop_target == 0 or stop_target == self.num_floors - 1:
                # cannot go past ground or top floor
                return (ElevatorState.STOP,)
            
            # cannot continue if passenger wants to get off at current stop target
            if stop_target in elevator_state.car_calls():
                return (ElevatorState.STOP,)
            # no passenger wants to get on or off next floor -> force continue
            # ADJUSTED: SEE WHAT HAPPENS WHEN REMOVING THIS CONSTRAINT
            if (not elevator_state.is_passenger_next_floor(self.floors, amount=amount) or
                    elevator_state.is_full()):
                # if elevator_state.is_full():
                return (ElevatorState.CONTINUE,)

            return (ElevatorState.STOP, ElevatorState.CONTINUE)
            
        return ()

    def process_actions(self, simulator):
        for elevator in self.elevators:
            possible_actions = self.get_possible_actions(elevator)

            # a constrained decision is made
            if len(possible_actions) == 1:
                simulator.insert(events.ElevatorActionEvent(simulator.now(), elevator, possible_actions[0]))
                logger.debug('time:%.3f:possible actions: %s', simulator.now(), const.MAP_CONST_STR[possible_actions[0]])
            elif len(possible_actions) == 2:
                simulator.insert(events.ElevatorControlEvent(simulator.now(), elevator))
                logger.debug('time:%.3f:possible actions: (%s, %s)', simulator.now(), const.MAP_CONST_STR[possible_actions[0]],
                             const.MAP_CONST_STR[possible_actions[1]])

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

    def update_accumulated_cost(self, simulator, event_time):
        """
        Update accumulated cost of elevators over time period
        """
        for elevator in self.elevators:
            try:
                elevator.controller.update_accumulated_cost(simulator, event_time)
            except AttributeError:
                # controller is not a reinforcement agent
                pass

    def stop_episode(self, simulator):
        """
        Handle everything that needs to be handled to end the episode.

        Parameters
        ----------
        simulator
        """
        for elevator in self.elevators:
            elevator.controller.episodes_so_far += 1
            try:
                elevator.controller.final(self.write_files, simulator)
                append_name = 'train' if elevator.controller.is_training else 'test'
                num_training = elevator.controller.num_training
            except AttributeError:
                # not a reinforcement agent
                append_name = 'heuristic'
                num_training = 0
        passenger_datafile = join(simulator.data_dir, '{}_{}_{}.csv'.format(simulator.stats_file, append_name, num_training))
        if self.write_files:
            with open(passenger_datafile, 'a') as f:
                # if not os.path.isfile(passenger_datafile):
                #     f.write('episode,waiting_time,boarding_time,system_time,threshold\r\n')
                avg_waiting = sum([x[0] for x in self.passenger_times]) / len(self.passenger_times)
                avg_boarding = sum([x[1] for x in self.passenger_times]) / len(self.passenger_times)
                avg_system = sum([x[2] for x in self.passenger_times]) / len(self.passenger_times)
                avg_threshold = sum([x[3] for x in self.passenger_times]) / len(self.passenger_times)
                csv_writer = csv.writer(f)
                csv_writer.writerow((self.elevators[0].controller.episodes_so_far, avg_waiting, avg_boarding, avg_system, avg_threshold))

    def __str__(self):
        res = 'elevator positions - '
        for i in range(self.num_elevators):
            res += '{}: {}, '.format(i, self.elevators[i].floor)
        return res[:-2]

    def __repr__(self):
        return 'Environment(num_floors={}, num_elevators={}, traffic_profile={})'.format(
            self.num_floors, self.num_elevators, self.traffic_profile)


class ElevatorState(object):
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
    motion :
        object handling elevator motion
    history : list
        containing ?
    accelerating_decision_made : bool
        true if a stop/continue decision at acceleration was made
    full_speed_decision_made : bool
        true if a stop/continue decision at full was made
    stop_target : int
        floor number of floor to stop at if action is stop
    """
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
    DONE_BOARDING = 13  # last passenger has boarded

    # elevator actions
    STOP = 8
    CONTINUE = 9
    NO_ACTION = 10
    MOVE_UP = 11
    MOVE_DOWN = 12

    def __init__(self, environment, controller='BestFirstAgent', floor=0, direction=None, index=0,
                 current_action=None, capacity=20, status=None, acc=0, vel=0, pos=0, history=None, **args):
        self.id = index
        self.environment = environment
        if controller == 'BestFirstAgent':
            self.controller = BestFirstAgent(index=self.id, **args)
        elif controller == 'ElevatorQAgent':
            self.controller = ElevatorQAgent(index=self.id, **args)
        elif controller == 'RandomAgent':
            self.controller = RandomAgent(index=self.id, **args)
        self._floor = floor
        self.direction = direction if direction else ElevatorState.STOPPED
        self._current_action = current_action if current_action else ElevatorState.NO_ACTION
        self.capacity = capacity
        self.passengers = {i: [] for i in range(self.environment.num_floors)}
        self._status = status if status else ElevatorState.IDLE
        self.motion = ElevatorMotion(self, acc, vel, pos)
        self.history = history if history else []
        # TODO: UPDATE DECISION TIME WHEN DECISION IS MADE
        self.accelerating_decision_made = False
        self.full_speed_decision_made = False
        self.stop_target = -1

    @property
    def floor(self):
        return self._floor

    @floor.setter
    def floor(self, value):
        self._floor = value
        logger.info('elevator %d reaches floor %d', self.id, value)

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value
        logger.info('elevator %d status changes to %s', self.id, const.MAP_CONST_STR[value])

    @property
    def current_action(self):
        return self._current_action

    @current_action.setter
    def current_action(self, value):
        self._current_action = value
        logger.info('elevator %d current action changes to %s', self.id, const.MAP_CONST_STR[value])

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

    def add_passenger(self, passenger, now):
        """
        Add given passenger to elevator

        Parameters
        ----------
        passenger : Passenger
        now : float
            current time in simulation
        """
        passenger.enter_elevator(self, now)

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
        return next_floor.has_passengers()

    def next_floor(self, floors, amount=1):
        """
        Return next floor the elevator is moving to.
        
        Parameters
        ----------
        floors : list
            list of floor objects in increasing order
        amount : int
            number of floors to look ahead

        Returns
        -------
        floor object
            next (amount) floor
        """
        assert 0 <= self.floor + self.direction * amount <= len(floors), 'next_floor checked on non-existing floor'
        return floors[self.floor + self.direction * amount]

    def is_decision_point(self, environment):
        """
        Return True if elevator reaches decision point.
        """
        last_floor_pos = environment.floors[self.floor].pos
        elevator_dist = abs(self.motion.pos - last_floor_pos)
        return ((self.status == ElevatorState.ACCELERATING and not self.accelerating_decision_made and
                 elevator_dist >= const.ACCEL_DECISION_DIST - const.GENERAL_EPS) or
                (self.status == ElevatorState.FULL_SPEED and not self.full_speed_decision_made and
                 elevator_dist >= const.FULL_SPEED_DECISION_DIST - const.GENERAL_EPS))

    def num_passengers(self):
        """
        Return number of passengers in the elevator.
        """
        res = 0
        for _, passengers in self.passengers.items():
            res += len(passengers)

        return res

    def num_passengers_up(self):
        """
        Return number of passengers going up in the elevator.
        """
        res = 0
        for _, passengers in self.passengers.items():
            if passengers and passengers[0].going_up(): 
                res += len(passengers)

        return res

    def num_passengers_down(self):
        """
        Return number of passengers going up in the elevator.
        """
        res = 0
        for _, passengers in self.passengers.items():
            if passengers and passengers[0].going_down(): 
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
        if (abs(self.motion.vel) >= const.MAX_SPEED - const.GENERAL_EPS and
            not (self.status == ElevatorState.FULL_SPEED or
                 self.status == ElevatorState.FULL_SPEED_DECELERATING)):
            self.status = ElevatorState.FULL_SPEED
            self.motion.vel = self.direction * const.MAX_SPEED

        # update elevator's floor when it crosses the floor
        if abs(self.motion.pos - simulator.environment.floors[self.floor].pos) >= const.FLOOR_HEIGHT - const.GENERAL_EPS:
            self.floor = self.next_floor(simulator.environment.floors).level
            self.motion.pos = simulator.environment.floors[self.floor].pos
            # if no stop action was taken, reset decision made variables
            if not (self.status == ElevatorState.ACCEL_DECELERATING or self.status == ElevatorState.FULL_SPEED_DECELERATING): 
                self.accelerating_decision_made = False
                self.full_speed_decision_made = False

    def do_action(self, simulator, action):
        """
        Update status of elevator according to given action.

        Parameters
        ----------
        action : int
            action elevator is taking
        """
        self.motion.reference_time = simulator.now()
        self.current_action = action
        if action == ElevatorState.MOVE_UP:
            self.direction = ElevatorState.UP
            self.status = ElevatorState.ACCELERATING
        elif action == ElevatorState.MOVE_DOWN:
            self.direction = ElevatorState.DOWN
            self.status = ElevatorState.ACCELERATING
        elif action == ElevatorState.STOP:
            logger.info('elevator %d plans to stop at floor %d', self.id, self.stop_target)
            if self.status == ElevatorState.FULL_SPEED:
                self.status = ElevatorState.FULL_SPEED_DECELERATING
                self.full_speed_decision_made = True
            elif self.status == ElevatorState.ACCELERATING:
                self.status = ElevatorState.ACCEL_DECELERATING
                self.accelerating_decision_made = True
        elif self.current_action == ElevatorState.CONTINUE:
            if self.status == ElevatorState.FULL_SPEED:
                self.full_speed_decision_made = True
            elif self.status == ElevatorState.ACCELERATING:
                self.accelerating_decision_made = True

    def complete_action(self, simulator):
        """
        Complete an action by updating elevator status.
        """
        # elevator arrives at floor
        floor = simulator.environment.floors[self.floor]
        if self.current_action == ElevatorState.STOP and self.floor == self.stop_target:
            self.arrive_at_floor(simulator, floor)
        elif self.current_action == ElevatorState.MOVE_UP or self.current_action == ElevatorState.MOVE_DOWN:
            self.current_action = ElevatorState.NO_ACTION
        elif self.current_action == ElevatorState.CONTINUE:
            self.current_action = ElevatorState.NO_ACTION
        
        # TODO: ADD TIME TO BOARD PASSENGERS

    def arrive_at_floor(self, simulator, floor):
        """
        Update environment when elevator stops at floor `floor'.

        Parameters
        ----------
        floor :
            floor object at which elevator stops
        """
        self.status = ElevatorState.BOARDING
        if floor.has_passengers() or self.is_passenger_getting_off():
            floor.board_passengers(simulator, self)
        else:
            simulator.insert(events.DoneBoardingEvent(simulator.now(), self))
        self.current_action = ElevatorState.NO_ACTION
        self.full_speed_decision_made = False
        self.accelerating_decision_made = False

    def is_passenger_getting_off(self):
        """Return True if elevator has arrived at car called floor."""
        return bool(self.passengers[self.floor])

    def is_action_in_progress(self):
        return self.current_action != ElevatorState.NO_ACTION

    def reset(self):
        """
        Reset the elevator state.
        """
        # self.controller = initial_state['controller']
        self.floor = 0
        self.direction = ElevatorState.STOPPED
        self.current_action = ElevatorState.NO_ACTION
        # dictionary of lists mapping floor to passengers traveling to that floor
        self.passengers = {i: [] for i in range(self.environment.num_floors)}
        self.status = ElevatorState.IDLE
        # acceleration, velocity and position
        self.motion.acc = 0
        self.motion.vel = 0
        self.motion.pos = 0
        self.stop_target = -1
        logger.debug('environment reset')

    def __repr__(self):
        return 'ElevatorState(environment, controller={}, floor={}, direction={}, \
current_action={}, capacity={}, action_in_progress={}, status={}, \
acc={}, vel={}, pos={}, accelerating_decision_made={}, full_speed_decision_made={})'.format(
                self.controller, self.floor, const.MAP_CONST_STR[self.direction],
                const.MAP_CONST_STR[self.current_action], self.capacity, self.is_action_in_progress(),
                const.MAP_CONST_STR[self.status], self.motion.acc, self.motion.vel,
                self.motion.pos, self.accelerating_decision_made, self.full_speed_decision_made)


cdef class ElevatorMotion:
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
    reference_time: float
        time in s which acceleration function evaluates as 0
    """
    cdef public float acc, vel, pos, reference_time
    cdef public object elevator_state

    def __init__(self, object elevator_state, float acc=0, float vel=0, float pos=0, float reference_time=0):
        self.elevator_state = elevator_state
        self.acc = acc
        self.vel = vel
        self.pos = pos
        self.reference_time = reference_time

    cdef dacc(self, object simulator):
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
        cdef float t, res
        t = simulator.now() - self.reference_time

        if self.elevator_state.status == ElevatorState.ACCELERATING:
            res = math.cos(const.ACCEL_CONST * t)
        elif self.elevator_state.status == ElevatorState.ACCEL_DECELERATING:
            res = (2 * const.ACCEL_DECEL[0] * t + const.ACCEL_DECEL[1])
        elif self.elevator_state.status == ElevatorState.FULL_SPEED_DECELERATING:
            res = - math.cos(const.ACCEL_CONST * t)
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
            self.elevator_state.status == ElevatorState.BOARDING):
            self.acc = 0
            self.vel = 0
        elif self.elevator_state.status == ElevatorState.FULL_SPEED:
            self.acc = 0
            self.vel = self.elevator_state.direction * const.MAX_SPEED
        else:
            self.acc += self.elevator_state.direction * self.dacc(simulator)

        self.vel += self.dvel(simulator)
        self.pos += self.dpos(simulator)

        if round(simulator.now(), 2) % 1 == 0:
            logger.debug('time:%.3f:elevator %d motion - acc:%.3f vel:%.3f pos:%.3f', simulator.now(), self.elevator_state.id,
                        self.acc, self.vel, self.pos)

    def __repr__(self):
        return 'ElevatorMotion(elevator_state, acc={}, vel={}, pos={}, reference_time={})'.format(
            self.acc, self.vel, self.pos, self.reference_time)


cdef class Floor(object):
    """
    Represents a floor in a building.

    Attributes
    ----------
    level : int
        floor number
    pos : float
        vertical position in meters
    passengers_up : list
        contains passengers on floor going up
    passengers_down : list
        contains passengers on floor going down
    up : bool
        up hall button on this floor True if on
    down : bool
        down hall button on this floor True if on
    """
    cdef public int level
    cdef public float pos
    cdef public list passengers_up, passengers_down
    cdef public bint _up
    cdef public bint _down
    def __init__(self, int level):
        self.level = level
        self.pos = const.FLOOR_HEIGHT * self.level
        self.passengers_up = []
        self.passengers_down = []
        self._up = False
        self._down = False

    @property
    def up(self):
        return self._up

    @up.setter
    def up(self, value):
        self._up = value
        msg = 'on' if value else 'off'
        logger.info('up button on floor %d turns %s', self.level, msg)

    @property
    def down(self):
        return self._down
    
    @down.setter
    def down(self, value):
        self._down = value
        msg = 'on' if value else 'off' 
        logger.info('down button on floor %d turns %s', self.level, msg)

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

        self.update_button(passenger.target)

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

    def has_passengers(self):
        return self.num_waiting() > 0

    cdef update_button(self, int target):
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
        """
        # if no passengers: button - false -> true, if passengers already: true -> true
        if target < self.level and not self.down:
            self.down = True
        elif target > self.level and not self.up:
            self.up = True

    def get_buttons(self):
        """
        Return floor buttons state.

        Returns
        -------
        tuple
            state of the down and up buttons in that order
        """
        return (self.down, self.up)

    def waiting_time(self, object simulator):
        """
        Return sum of passenger waiting times on this floor.
        """
        cdef float result
        result = 0
        for passenger in self.all_passengers():
            result += passenger.waiting_time(simulator)
        return result

    def board_passengers(self, object simulator, object elevator_state):
        """
        Transfer passengers from floor to elevator and vice versa

        If capacity of elevator is not enough to accomodate passengers, transfer only first arrived
        passengers until elevator capacity is full.

        Called when elevator stops at floor.
        """
        cdef float now, boarding_time
        cdef int capacity_left, num_up, num_down
        now = simulator.now()
        boarding_time = 1
        # TODO: WHICH PASSENGER BOARDING DIRECTION DEPENDS ON ELEVATOR PASSENGERS AS WELL
        passengers_off = elevator_state.passengers[elevator_state.floor]
        for passenger in passengers_off:
            simulator.insert(events.PassengerTransferEvent(now + boarding_time, passenger, elevator_state, to_elevator=False))
            boarding_time += 1  # TODO: Make boarding time random variable
        capacity_left = elevator_state.capacity_left() + len(passengers_off)
        num_up = self.num_up()
        num_down = self.num_down()
        passengers_boarding = None
        # TODO: WHEN ELEVATOR REACHES TOP OR BOTTOM FLOOR, CHANGE DIRECTION TO ?STOPPED?
        if elevator_state.direction == ElevatorState.UP:
            if num_up > 0:
                if capacity_left < num_up:
                    passengers_boarding = self.passengers_up[:capacity_left]
                else:
                    passengers_boarding = self.passengers_up[:]
                    simulator.environment.floors[elevator_state.floor].up = False
            # elif num_down > 0 and not simulator.environment.is_hall_call(elevator_state.floor, above=True, down_up=True):
            elif num_down > 0 and elevator_state.num_passengers_up() == 0:
                if capacity_left < num_down:
                    passengers_boarding = self.passengers_down[:capacity_left]
                else:
                    passengers_boarding = self.passengers_down[:]
                    simulator.environment.floors[elevator_state.floor].down = False
        elif elevator_state.direction == ElevatorState.DOWN:
            if num_down > 0:
                if capacity_left < num_down:
                    passengers_boarding = self.passengers_down[:capacity_left]
                else:
                    passengers_boarding = self.passengers_down[:]
                    simulator.environment.floors[elevator_state.floor].down = False
            elif num_down > 0 and elevator_state.num_passengers_down() == 0:
                if capacity_left < num_up:
                    passengers_boarding = self.passengers_up[:capacity_left]
                else:
                    passengers_boarding = self.passengers_up[:]
                    simulator.environment.floors[elevator_state.floor].up = False

        if passengers_boarding:
            for passenger in passengers_boarding:
                # TODO: time from truncated erlang instead of 1 second
                simulator.insert(events.PassengerTransferEvent(now + boarding_time, passenger, elevator_state, to_elevator=True))
                boarding_time += 1
            simulator.insert(events.DoneBoardingEvent(now + boarding_time - 1 + const.GENERAL_EPS, elevator_state))
        else:
            simulator.insert(events.DoneBoardingEvent(now + boarding_time - 1, elevator_state))

    def reset(self):
        self.passengers_up = []
        self.passengers_down = []
        self.up = False
        self.down = False

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
    # cdef public int status, target, _id
    # cdef public float arrival_time, boarded_time
    # cdef public object floor
    
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
        self.id = Passenger.num_passengers_total
        Passenger.num_passengers_total += 1
        self.floor = floor
        self.arrival_time = 0
        self.boarded_time = 0

    def arrive_at_floor(self, simulator):
        """
        Passenger chooses a target floor and is added to its arriving floor's queue.
        """
        logger.info('time:%.3f:passenger %d arrives at floor %d', simulator.now(), self.id, self.floor.level)
        self.arrival_time = simulator.now()
        self.target = self.choose_target(simulator.environment)
        self.floor.add_passenger(self)

    def system_time(self, float t):
        """
        Return time passenger has been in system at time t.
        
        Parameters
        ----------
        t : float
            time in seconds
        """
        return (t - self.arrival_time) * (t > self.arrival_time)

    def waiting_time(self, float t):
        """
        Return time passenger has/had waited for an elevator at time t.
        
        Waiting time is zero if t is smaller than the arrival time.

        Parameters
        ----------
        t : float
            time in seconds
        """
        return self.system_time(t) - self.boarding_time(t)

    def boarding_time(self, float t):
        """
        Return time passenger has been in elevator at time t.
        
        Parameters
        ----------
        t : float
            time in seconds
        """
        return (t - self.boarded_time) * (self.status == Passenger.BOARDED)

    def going_up(self):
        """Return True if passenger is going up."""
        return self.target > self.floor.level

    def going_down(self):
        """Return True if passenger is going up."""
        return self.target < self.floor.level
    
    def choose_target(self, object environment):
        """
        Return target floor according to current traffic
        """
        target = environment.traffic_profile.choose_target(self.floor)
        logger.info('passenger %d chooses floor %d', self.id, target)
        return target

    def enter_elevator(self, object elevator_state, float now):
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
        self.boarded_time = now

        elevator_state.passengers[self.target].append(self)
        logger.info('passenger %d enters elevator %d', self.id, elevator_state.id)

    def exit_elevator(self, object elevator_state, float now, object environment):
        """
        Remove passenger from elevator and system when exiting elevator.

        Parameters
        ----------
        elevator_state :
            called by the elevator defined in that elevator state
        """
        environment.passenger_times.append((self.waiting_time(now), self.boarding_time(now),
                                            self.system_time(now), self.waiting_time(now) > 60))
        # write waiting time and boarding time to string stream
        # data = (elevator_state.controller.episodes_so_far, self.waiting_time(now),
        #         self.boarding_time(now))
        elevator_state.passengers[self.target].remove(self)
        logger.info('passenger %d exits elevator %d', self.id, elevator_state.id)

    def update(self):
        pass

    def __repr__(self):
        return 'Passenger(floor={}, target={})'.format(self.floor, self.target)


class TrafficProfile(ABC):
    """
    Base class for traffic profiles such as DownPeak, UpPeak, etc.

    Attributes
    ----------
    interfloor : float
        \in [0, 1], percentage of interfloor travel in terms of total arrival rate
    """
    def __init__(self, num_floors, interfloor=0):
        self.num_floors = num_floors
        self.interfloor = interfloor

    @abstractmethod
    def choose_target(self, passenger):
        pass


class DownPeak(TrafficProfile):
    """
    Represents a downpeak traffic profile.

    Attributes
    ----------
    target_floor : int
        floor to which most passengers are headed
    arrival_rates: tuple
        mean number of passengers during a typical afternoon business hour
    """
    def __init__(self, num_floors, interfloor=0.5):
        super().__init__(num_floors, interfloor)
        self.target_floor = 0

    def choose_target(self, floor):
        # with prob `interfloor' choose floor != target else choose target
        cdef list possible_floors
        if random.random() < self.interfloor:
            possible_floors = [pos_floor for pos_floor in range(self.num_floors) if pos_floor not in (0, floor.level)]
            target = random.choice(possible_floors)
        else:
            target = self.target_floor

        return target

    def arrival_rate(self, float time):
        """
        Return mean number of people arriving in this timeframe.

        Parameters
        ----------
        time : float
            time in seconds after starting simulation
        """
        cdef float minutes_in
        cdef int index
        minutes_in = time / const.SECONDS_PER_MINUTE
        index = int(minutes_in / const.MINUTES_PER_TIME_INTERVAL)
        return const.DOWNPEAK_RATES[index]

    def __repr__(self):
        return 'DownPeak(num_floors={}, interfloor={})'.format(self.num_floors, self.interfloor)
