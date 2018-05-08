import numpy as np
import numpy.random as rnd

from abc import ABC, abstractmethod
from sortedcontainers import SortedList

from environment import Passenger


class Simulator():
    def __init__(self):
        self.time = 0
        self.events = SortedList()
        self.seed = 42
        rnd.seed(self.seed)

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
    def __init__(self, time, passenger):
        super().__init__(time)
        self.passenger = passenger

    def execute(self, simulator):
        simulator.get_events().add(self)
        # after arrival of a passenger, new passenger arrival gets scheduled for same floor
        simulator.get_events().add(PassengerSchedulerEvent(self.get_time(), self.get_passenger()))

    def get_passenger(self):
        return self.passenger        


class PassengerSchedulerEvent(Event):
    def __init__(self, time, prev_passenger):
        super().__init__(time)
        # self.prev_passenger = passenger
        self.new_passenger = Passenger(passenger.get_floor())
    
    def execute(self):
        rate = self.new_passenger.get_floor().get_rate()
        # arrival according to poisson(rate) process, where rate differs per floor
        next_time = rnd.exponential(scale=1 / rate)
        simulator.get_events().add(PassengerArrivalEvent(next_time, self.new_passenger))


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


class EventHandler:
    pass

sim = Simulator()
pas = PassengerArrivalEvent(0)
print(pas.get_time())
# print(x.events)