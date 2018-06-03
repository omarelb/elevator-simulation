"""
Implements environment of the elevator model.
"""

# user modules
import constants as const
import control

# other modules
import numpy as np
import numpy.random as rnd
from abc import ABC, abstractmethod
from queue import Queue

from time import time, sleep


class Environment:
    """
    Combines all separate parts of an environment.

    Attributes
    ----------
    arrival_rates
    traffic_profile
    num_floors
    num_elevators
    floors
    elevators
    state
    """
    def __init__(self, num_floors=5, num_elevators=1, traffic_profile='down_peak'):
        if traffic_profile == 'down_peak':
            self.traffic_profile = DownPeak(num_floors)
        else:
            self.traffic_profile = DownPeak(num_floors)
        
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        # initialize floors
        self.floors = [Floor(level) for level in range(self.num_floors)]
        self.elevators = []
        # etc.

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
        else:
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
        for elevator in self.elevators:
            elevator.reset()

        for floor in self.floors:
            floor.reset()

        
    def is_terminal(self):
        """
          Has the enviornment entered a terminal
          state? This means there are no successors
        """
        state = self.get_current_state()
        actions = self.get_possible_actions(state)
        return len(actions) == 0


    def __str__(self):
        pass

    def __repr__(self):
        pass


class ElevatorState:
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

    num_elevators = 0

    def __init__(self, environment, controller=control.ReinforcementAgent(), floor=0, direction=ElevatorState.STOPPED,
                 current_action=ElevatorState.NO_ACTION, capacity=20, passengers=None, action_in_progress=False, 
                 status=ElevatorState.IDLE, constrained=False, acc=0, vel=0, pos=0, history=None,
                 decision_time=None, decision_made=False, acc_update=None):
        """
        Represents state of an elevator.

        Attributes
        ----------
        id : int

        environment :
        controller :

        floor : int

        direction : int

        current_action :

        capacity : int

        passengers : dict
            dictionary of lists mapping floor to passengers traveling to that floor
        action_in_progress : bool

        status : int
        constrained : bool
        True if current elevator action was constrained
        eleration, velocity and position
        acc : float
        vel : float
        pos : float
        history :
        decision_time : float
        decision_made : bool
        acc_update :

        """
        self.id = ElevatorState.num_elevators
        ElevatorState.num_elevators += 1
        self.environment = environment
        self.controller = controller
        self.floor = floor
        self.direction = direction
        self.current_action = current_action
        self.capacity = capacity
        # dictionary of lists mapping floor to passengers traveling to that floor
        self.passengers = {i : [] for i in range(self.environment.num_floors)}
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
        self.passengers = {i : [] for i in range(self.environment.num_floors)}
        self.action_in_progress = False
        self.status = ElevatorState.IDLE
        # True if current elevator action was constrained
        self.constrained = False
        # acceleration, velocity and position
        self.acc = initial_state['acc']
        self.vel = initial_state['vel']
        self.pos = initial_state['pos']
        

    # def get_legal_actions(self, state):
    #     # TODO: modelling
    #     if self.state == Elevator.IDLE:
    #         pass
    #     elif self.state == Elevator.MOVING:
    #         pass

    # def get_motion(self):
    #     return (self.acc, self.vel, self.pos)


class Floor:
    def __init__(self, environment, level):
        """
        Represents a floor in a building

        Attributes
        ----------
        environment : Environment
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
        self.environment = environment
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
        passenger.floor = self


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
        return '[Level: {}, num_waiting: {}]'.format(self.level, self.num_waiting)

    def __repr__(self):
        return '[Level: {}, num_waiting: {}]'.format(self.level, self.num_waiting())


class Passenger:
    WAITING = 0
    BOARDED = 1

    num_passengers_total = 0

    def __init__(self, environment, floor, target=0):
        self.environment = environment
        # floor object
        self.floor = floor
        self.target = self.choose_target()
        # passenger presses button
        self.floor.update_button(self.target)
        self.status = Passenger.WAITING
        # time passenger waits until elevator arrives
        self.waiting_time = 0
        # time passenger is in elevator
        self.boarding_time = 0

        self.id = Passenger.num_passengers_total
        Passenger.num_passengers_total += 1


    def system_time(self):
        """Return time passenger has been in system."""
        return self.waiting_time + self.boarding_time

    
    def update_time(self):
        """Update waiting or boarding time."""
        pass


    def going_up(self):
        """Return True if passenger is going up."""
        return self.target > self.floor


    def going_down(self):
        """Return True if passenger is going up."""
        return self.target < self.floor
    
    
    def choose_target(self):
        """
        Return target floor according to current traffic
        """
        return self.environment.traffic_profile.choose_target(self.floor)


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
        self.num_floors =  num_floors
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
        super.__init__(num_floors, interfloor)
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
        """
        # index = 
        # return const.DOWNPEAK_RATES[index]
        pass
        # TODO:
