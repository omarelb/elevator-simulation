"""
Implements environment of the elevator model.
"""
# user modules
import constants as const

# other modules
import numpy as np
import numpy.random as rnd
from abc import ABC, abstractmethod
# from queue import Queue

# from time import time, sleep
from qlearningAgents import ElevatorQAgent


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

    def update(self):
        """
        Update environment state.
        """
        pass

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
        return (self.num_hall_calls(es, down=True, above=True), self.num_hall_calls(es, down=False, above=True),
                self.num_hall_calls(es, down=True, above=False), self.num_hall_calls(es, down=False, above=False),
                elevator_state.num_car_calls(), elevator_state.floor, elevator_state.direction)

    def get_buttons(self, down=False, up=False):
        """
        Return button state of all floors.

        Parameters
        ----------
        down : bool
            If true, return only down button state
        up : bool
            If true, return only up button state

        Returns
        -------
        tuple
            every element i contains the button state of floor i
        """
        all_buttons = tuple(floor.get_buttons() for floor in self.floors)

        if down:
            return tuple(buttons[0] for buttons in all_buttons)
        if up:
            return tuple(buttons[1] for buttons in all_buttons)

        return all_buttons

    def num_hall_calls(self, elevator_state, down=False, above=False):
        """
        Return number of hall calls relative to elevator position.

        Parameters
        ----------
        elevator_state :
            state of elevator
        down : bool
            If true, look at down hall calls, up hall calls otherwise
        above : bool
            If true, look at hall calls above elevator, below otherwise

        Returns
        -------
        int
            number of up/down hall calls above/below elevator
        """
        current_floor = elevator_state.floor

        buttons = self.get_buttons(down)

        if above:
            return sum(buttons[current_floor + 1:])

        return sum(buttons[:current_floor])

    def get_possible_actions(self, state):
        """
          Returns possible actions the agent
          can take in the given state. Can
          return the empty list if we are in
          a terminal state.
        """
        pass

    def do_action(self, action):
        """
          Performs the given action in the current
          environment state and updates the enviornment.

          Returns a (reward, next_state) pair
        """
        pass

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
        the elevator passed/stopped.
    direction : int
        current direction of the elevator: up, down, or stopped
    current_action :
        current action the elevator is taking
    capacity : int
        maximum number of people that can get into the elevator at the same time.
    passengers : dict
        dictionary of lists mapping floor to passengers traveling to that floor
    action_in_progress : bool
        true if elevator is currently performing an action
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
    """
    # elevator direction
    UP = 1
    DOWN = -1
    STOPPED = 0

    # elevator status
    IDLE = 2
    ACCELERATING = 3
    FULL_SPEED_DECELERATING = 4  # decelerating after reaching full speed
    ACCEL_DECELERATING = 5  # decelerating after just accelerating
    FULL_SPEED = 6  # moving at full speed

    # elevator actions
    STOP = 7
    CONTINUE = 8
    NO_ACTION = 9

    num_elevators = 0

    def __init__(self, environment, controller=ElevatorQAgent(), floor=0, direction=None,
                 current_action=None, capacity=20, action_in_progress=False,
                 status=None, constrained=False, acc=0, vel=0, pos=0, history=None,
                 decision_time=None, decision_made=False):
        self.id = ElevatorState.num_elevators
        ElevatorState.num_elevators += 1
        self.environment = environment
        self.controller = controller
        self.floor = floor
        self.direction = direction if direction else ElevatorState.STOPPED 
        self.current_action = current_action if current_action else ElevatorState.NO_ACTION 
        self.capacity = capacity
        self.passengers = {i: [] for i in range(self.environment.num_floors)}
        self.action_in_progress = action_in_progress
        self.status = status
        self.status = status if status else ElevatorState.IDLE
        self.constrained = constrained
        self.motion = ElevatorMotion(self, acc, vel, pos)
        self.history = history if history else []
        self.decision_time = decision_time
        self.decision_made = decision_made

    def is_action_in_progress(self):
        """Return True if an action is currently in progress."""
        return self.action_in_progress

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
        for target_floor, passengers in self.passengers:
            # someone going to that floor
            if len(passengers) > 0:
                calls.append(target_floor)

        return calls

    def num_car_calls(self):
        """
        Return number of remaining car calls in current direction.
        """
        return len(self.car_calls())

    def num_passengers(self):
        res = 0
        for _, passengers in self.passengers:
            res += len(passengers)

        return res

    def passengers_as_list(self):
        """
        Return list of passengers instead of dict for iteration purposes.
        """
        res = []
        for _, passengers in self.passengers:
            res += passengers

        return res

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
        self.action_in_progress = False
        self.status = ElevatorState.IDLE
        # True if current elevator action was constrained
        self.constrained = False
        # acceleration, velocity and position
        self.motion.acc = initial_state['acc']
        self.motion.vel = initial_state['vel']
        self.motion.pos = initial_state['pos']

    def __repr__(self):
        return 'ElevatorState(environment, controller={}, floor={}, direction={},\
                 current_action={}, capacity={}, action_in_progress={},\
                 status={}, constrained={}, acc={}, vel={}, pos={}, decision_time={},\
                 decision_made={})'.format(self.controller, self.floor, self.direction,
                 self.current_action, self.capacity, self.action_in_progress, self.status,
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
        """Update elevator motion state."""
        if self.elevator_state.status == ElevatorState.IDLE or self.elevator_state.status == ElevatorState.FULL_SPEED:
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
        if passenger.moving_up():
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
            if target < self.level:
                self.down = True
            else:
                self.up = True
        else:
            if elevator_direction == ElevatorState.UP:
                self.up = False
            else:
                self.down = False

    def get_buttons(self):
        """
        Return floor buttons state.

        Returns
        -------
        tuple
            state of the down and up buttons in that order
        """
        return (self.down, self.up)

    def waiting_time(self):
        """
        Return sum of passenger waiting times on this floor.
        """
        result = 0
        for passenger in self.all_passengers():
            result += passenger.waiting_time

    def board_passengers(self, elevator_state):
        """
        Transfer passengers from floor to elevator.

        If capacity of elevator is not enough to accomodate passengers, transfer only first arrived
        passengers until elevator capacity is full.

        Called when elevator stops at floor.
        """
        capacity_left = elevator_state.capacity_left()
        if elevator_state.direction == ElevatorState.UP:
            if capacity_left < self.num_up():
                passengers_boarding = self.passengers_up[:capacity_left]
                del self.passengers_up[:capacity_left]
            else:
                passengers_boarding = self.passengers_up[:]
                del self.passengers_up[:]
        else:
            if capacity_left < self.num_down():
                passengers_boarding = self.passengers_down[:capacity_left]
                del self.passengers_down[:capacity_left]
            else:
                passengers_boarding = self.passengers_down[:]
                del self.passengers_down[:]

        elevator_state.add_passengers(passengers_boarding)

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

    def arrive_at_floor(self, simulator, environment):
        self.arrival_time = simulator.now()
        self.target = self.choose_target(environment)
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
        return self.target > self.floor

    def going_down(self):
        """Return True if passenger is going up."""
        return self.target < self.floor
    
    def choose_target(self, environment):
        """
        Return target floor according to current traffic
        """
        return environment.traffic_profile.choose_target(self.floor)

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