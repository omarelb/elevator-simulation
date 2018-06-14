import random
from abc import ABC, abstractmethod
import numpy.random as rnd

import environment as env
import constants as const


class Event(ABC):
    """
    Base class for events happening furing simulation.
    """
    def __init__(self, time):
        self.time = time

    @abstractmethod
    def execute(self, simulator, environment):
        pass

    def __lt__(self, other):
        return self.time < other.time

    def __gt__(self, other):
        return self.time > other.time


class PassengerArrivalEvent(Event):
    """
    Passenger arrives at floor
    """
    def __init__(self, time, floor):
        super().__init__(time)
        self.floor = floor

    def execute(self, simulator):
        """
        Handle passenger arrival event.

        Passenger is added to floor and is asked to press a hall button.
        A new passenger arrival on the same floor is then generated. 
        """
        new_passenger = env.Passenger(self.floor)
        new_passenger.arrive_at_floor(simulator)
        self.generate(simulator)

        simulator.environment.update_accumulated_cost(simulator, self.time)
        simulator.environment.last_accumulator_event_time = self.time

    def generate(self, simulator):
        """
        Generate a passenger arrival event according to a poisson(rate) process.

        Rate depends on time in-simulation.
        """
        arrival_rate = simulator.environment.traffic_profile.arrival_rate(simulator.now()) / const.MINUTES_PER_TIME_INTERVAL
        inter_arrival_time = random.expovariate(arrival_rate) * const.SECONDS_PER_MINUTE
        simulator.insert(PassengerArrivalEvent(simulator.now() + inter_arrival_time, self.floor))

    def __repr__(self):
        return 'PassengerArrivalEvent(time={:.3f}, floor={})'.format(self.time, self.floor)


class PassengerTransferEvent(Event):
    """
    Event generated when a passenger is transferred from a floor to the elevator or vice versa.

    Attributes
    ----------
    time : float
        current simulator time
    passenger :
        passenger transferred
    elevator_state :
        elevator ro be transferred to/from
    to_elevator : bool
        True if passenger is transferred from floor to elevator, False otherwise
    """
    def __init__(self, time, passenger, elevator_state, to_elevator):
        super().__init__(time)
        self.passenger = passenger
        self.elevator_state = elevator_state
        self.to_elevator = to_elevator

    def execute(self, simulator):
        # remove passenger from floor
        if self.to_elevator:
            if self.passenger.going_up():
                self.passenger.floor.passengers_up.pop(0)
            else:
                self.passenger.floor.passengers_down.pop(0)
        
            self.passenger.enter_elevator(self.elevator_state, simulator.now())
        # self.passenger.floor.passengers
        else:
            self.passenger.exit_elevator(self.elevator_state, simulator.now(), simulator.environment)
        
        simulator.environment.update_accumulated_cost(self.time, simulator)
        simulator.environment.last_accumulator_event_time = self.time

    def __repr__(self):
        return 'PassengerTransferEvent(time={:.3f}, passenger={}, to_elevator={})'.format(
            self.time, self.passenger.id, self.to_elevator)


class DoneBoardingEvent(Event):
    def __init__(self, time, elevator_state):
        super().__init__(time)
        self.elevator_state = elevator_state

    def execute(self, simulator):
        if self.elevator_state.is_empty() and simulator.environment.no_buttons_pressed():
            self.elevator_state.status = env.ElevatorState.IDLE
            self.elevator_state.direction = env.ElevatorState.STOPPED
        else:
            self.elevator_state.status = env.ElevatorState.DONE_BOARDING

    def __repr__(self):
        return 'DoneBoardingEvent(time={:.3f})'.format(self.time)


class ElevatorActionEvent(Event):
    """
    Event generated whenever elevator is required to take an action.

    Deals with changing state of the elevator.

    If the elevator action is not constrained i.e. the elevator needs to make a decision,
    an ElevatorControlEvent is scheduled
    """
    def __init__(self, time, elevator_state, action):
        super().__init__(time)
        self.elevator_state = elevator_state
        self.action = action

    def execute(self, simulator):
        # simulator.environment.do_action()
        # TODO: Put code below into environment.do_action
        self.elevator_state.do_action(simulator, self.action)

    def __repr__(self):
        return 'ElevatorActionEvent(time={}, elevator_state={}, action={})'.format(self.time, self.elevator_state, const.MAP_CONST_STR[self.action])


class ElevatorControlEvent(Event):
    """
    Gets inserted when elevator controller is asked to make a decision.

    Once a decision is returned by the controller, generates an ElevatorActionEvent
    """
    def __init__(self, time, elevator_state):
        super().__init__(time)
        self.elevator_state = elevator_state

    def execute(self, simulator):
        # if self.elevator_state.controller is reinforcement agent
        simulator.environment.update_accumulated_cost(simulator, self.time)
        new_state = simulator.environment.get_learning_state(self.elevator_state)
        if self.elevator_state.controller.last_action:
            try:
                self.elevator_state.controller.observe_transition(self.time, new_state,
                                        self.elevator_state.controller.cost_accumulator)
                simulator.environment.last_accumulator_event_time = self.time
            except AttributeError:
                # not a reinforcement agent
                pass
        action = self.elevator_state.controller.get_action(simulator)
        self.elevator_state.controller.decision_time = self.time
        self.elevator_state.controller.cost_accumulator = 0

        simulator.insert(ElevatorActionEvent(simulator.now(), self.elevator_state, action))

    def __repr__(self):
        return 'ElevatorControlEvent(time={:.3f})'.format(self.time)
