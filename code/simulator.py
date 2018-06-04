import numpy as np
import numpy.random as rnd

import heapq
from abc import ABC, abstractmethod
from sortedcontainers import SortedList

import constants as const

from environment import Environment, Floor, ElevatorState, Passenger


class Simulator():
    """
    Organizes simulation of the system.

    Parameters
    ----------
    time_step : float
        time progress every loop (in seconds)
    time : float
        total time that has passed after starting the simulation (in seconds)
    events : 
        list that keeps track of events in the simulation. implemented as priorityqueue, priority being time.
    seed :
        seed for the random generator
    environment :
        object that keeps track of environment state
    max_time : float
        amount of time one episode should run for
    """
    def __init__(self):
        self.time_step = const.TIME_STEP
        self.time = 0
        # could also be implemented as priority queue
        self.events = []
        self.seed = 42
        rnd.seed(self.seed)

        # TODO: parameterize this
        self.environment = Environment()
        
        self.max_time = 100 # seconds 60 * 60 seconds=1 hour

    def initialize_simulation(self):
        """
        Start off the simulation by generating the first arrivals.
        """
        for floor in self.environment.floors:
            PassengerArrivalEvent(self.now(), floor).generate(self, self.environment)

    def insert(self, event):
        """
        Insert event into queue.

        Parameters
        ----------
        event :
            any event object
        """
        heapq.heappush(self.events, event)

    def now(self):
        """Return current running time."""
        return self.time

    def step(self):
        """
        Update simulator time.
        """
        self.time = self.time + self.time_step

    def observe(self):
        pass

    def update(self):
        # event happened
        try:
            if self.time >= self.events[0].time:
                # handle event
                event = heapq.heappop(self.events)
                event.execute(self)
        # no events in queue yet
        except IndexError:
            pass

    def run(self):
        self.initialize_simulation()

        # running = True
        # while running:
        #     if round(self.time, 5) % 1 == 0:
        #         print('{} seconds passed'.format(round(self.time, 3)))
        #         print(self.events)
        #     # main loop
        #     self.step()
        #     self.update()

        #     if self.time > self.max_time:
        #         running = False


# abstract base class
class Event(ABC):
    """
    Base class for events happening furing simulation.
    """
    def __init__(self, time):
        self.time = time

    @abstractmethod
    def execute(self, simulator):
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

    def execute(self, simulator, environment):
        """
        Handle passenger arrival event.

        Passenger is added to floor and is asked to press a hall button.
        A new passenger arrival on the same floor is then generated. 
        """
        new_passenger = Passenger(simulator, self.floor)
        new_passenger.arrive_at_floor(simulator, environment)

        self.generate(simulator, environment)

    def generate(self, simulator, environment):
        """
        Generate a passenger arrival event according to a poisson(rate) process.

        Rate depends on time in-simulation.
        """
        arrival_rate = environment.traffic_profile.arrival_rate(simulator.now())
        next_arrival_time = rnd.exponential(scale=1 / arrival_rate) * const.SECONDS_PER_MINUTE
        simulator.insert(PassengerArrivalEvent(simulator.now() + next_arrival_time, self.floor))

    def __str__(self):
        return 'Arrival time: {}, Floor: {}'.format(self.time, self.floor)

    def __repr__(self):
        return 'PassengerArrivalEvent(time={:.3f}, floor={})'.format(self.time, self.floor)


class PassengerTransferEvent(Event):
    pass


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


class ElevatorControlEvent(Event):
    """
    Get scheduled every set amount of time (or when state has changed?) -> ask controller for action
    """
    pass


class EventHandler:
    pass


if __name__ == '__main__':
    sim = Simulator()
    sim.run()
