import numpy as np
import numpy.random as rnd

from abc import ABC, abstractmethod
from sortedcontainers import SortedList

import constants as const

from environment import Environment, Floor, ElevatorState, Passenger


class Simulator():

    def __init__(self):
        self.time_step = const.TIME_STEP
        self.time = 0
        # could also be implemented as priority queue
        self.events = SortedList()
        self.seed = 42
        rnd.seed(self.seed)

        # TODO: parameterize this
        self.environment = Environment()
        
        self.max_time = 100 # seconds


    def initialize_simulation(self):
        for floor in self.building.get_floors():
            PassengerSchedulerEvent(self.get_time(), floor).execute(self)

    def insert(self, event):
        self.events.add(event)

    def now(self):
        return self.time

    def do_all_events(self):
        for _ in range(len(self.events)):
            # remove earliest event and execute it
            first_event = self.events.pop(0)
            first_event.execute(self)


    def get_events(self):
        return self.events

    
    def get_time(self):
        return self.time


    def step(self):
        self.time = self.time + self.time_step
        

    def observe(self):
        pass
    

    def update(self):
        # event happened
        try:
            if self.get_time() >= self.get_events()[0].get_time():
                # handle event
                event = self.get_events().pop(0)
                event.execute(self)

                print(self.get_events())
        # no events in queue yet
        except IndexError:
            pass


    def run(self):
        self.initialize_simulation()

        running = True
        while running:
            if round(self.get_time(), 5) % 1 == 0:
                print('{} seconds passed'.format(round(self.get_time(), 3)))
                print(self.get_events())
            # main loop
            self.step()
            self.update()

            # running=False
            # if self.get_time() > 0:
            if self.get_time() > self.max_time:
            # if self.get_time() > self.max_time:
                running = False


# abstract base class
class Event(ABC):
    def __init__(self, time):
        self.time = time

    @abstractmethod
    def execute(self, simulator):
        pass

    def __lt__(self, other):
        return self.time < other.time

    def __gt__(self, other):
        return self.time > other.time

    def get_time(self):
        return self.time


class PassengerArrivalEvent(Event):
    """
    Passenger arrives at floor
    """
    def __init__(self, time, floor):
        super().__init__(time)
        self.floor = floor

    def execute(self, simulator):
        # print(self.floor)
        # adds passenger to floor and sets passenger.floor to self.floor
        # print('\nexecuting arrival')
        # print(simulator.get_time())
        # print(self.get_floor())
        self.floor.add_passenger(Passenger())
        # print(self.get_floor())
        new_time = self.get_time() + 2 * Simulator.TIME_STEP
        # print(new_time)
        # after arrival of a passenger, new passenger arrival gets scheduled for same floor
        simulator.get_events().add(PassengerSchedulerEvent(new_time, self.get_floor()))
        # print(simulator.get_events())

        # exit()


    def get_floor(self):
        return self.floor

    
    def __str__(self):
        return 'Arrival time: {}, Floor: {}'.format(self.time, self.floor)

    def __repr__(self):
        return 'Arrival time: {}, Floor: {}'.format(self.time, self.floor)

class PassengerSchedulerEvent(Event):
    def __init__(self, time, floor):
        super().__init__(time)
        # self.prev_passenger = passenger
        self.floor = floor

    
    def execute(self, simulator):
        rate = self.floor.get_arrival_rate()
        # arrival according to poisson(rate) process, where rate differs per floor
        # rate is expressed in minutes and value is then converted to number of seconds
        next_time = rnd.exponential(scale=1 / rate) * const.SECONDS_PER_MINUTE
        # TODO: randomize target floor
        simulator.get_events().add(PassengerArrivalEvent(simulator.get_time() + next_time, self.floor))

    def __str__(self):
        return 'Schedule time: {}, Floor: {}'.format(self.time, self.floor)

    def __repr__(self):
        return 'Handletime: {}, Floor: {}'.format(self.time, self.floor)


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


sim = Simulator()
sim.run()
# pas = PassengerArrivalEvent(0)
# print(pas.get_time())
# print(x.events)