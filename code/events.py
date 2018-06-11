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

    def generate(self, simulator):
        """
        Generate a passenger arrival event according to a poisson(rate) process.

        Rate depends on time in-simulation.
        """
        arrival_rate = simulator.environment.traffic_profile.arrival_rate(simulator.now())
        next_arrival_time = rnd.exponential(scale=1 / arrival_rate) * const.SECONDS_PER_MINUTE
        simulator.insert(PassengerArrivalEvent(simulator.now() + next_arrival_time, self.floor))

    # def __str__(self):
    #     return 'Arrival time: {}, Floor: {}'.format(self.time, self.floor)

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
        
            self.passenger.enter_elevator(self.elevator_state)
        # self.passenger.floor.passengers
        else:
            self.passenger.exit_elevator(self.elevator_state)

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
        else:
            self.elevator_state.status = env.ElevatorState.DONE_BOARDING

    def __repr__(self):
        return 'DoneBoardingEvent(time={:.3f})'.format(self.time)


class HallCallEvent(Event):
    def __init__(self):
        super().__init__()
        self.down = False
        self.up = False

    def execute(self, simulator):
        pass


class HallCallHandled(Event):
    pass


class ElevatorArrivalEvent(Event):
    def __init__(self):
        pass

    def execute(self, simulator):
        pass


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
        # self.elevator_state.controller.observation_function(simulator)
        # self.elevator_state.controller.observe_transition()
        action = self.elevator_state.controller.get_action(simulator)

        simulator.insert(ElevatorActionEvent(simulator.now(), self.elevator_state, action))

    def __repr__(self):
        return 'ElevatorControlEvent(time={:.3f})'.format(self.time)