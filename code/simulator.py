from os.path import join

import heapq
import logging
import numpy as np
import numpy.random as rnd

from environment import Environment, Floor, ElevatorState, Passenger
import constants as const
import events

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler(join(const.LOG_DIR, 'simulator.log'), mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


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
        
        self.max_time = 100  # seconds 60 * 60 seconds=1 hour

    def initialize_simulation(self):
        """
        Start off the simulation by generating the first arrivals. No arrivals on ground floor.
        """
        for floor in self.environment.floors[1:]:
            events.PassengerArrivalEvent(self.now(), floor).generate(self)

    def insert(self, event):
        """
        Insert event into queue.

        Parameters
        ----------
        event :
            any event object
        """
        heapq.heappush(self.events, event)
        logger.info('%s inserted.', event)

    def now(self):
        """Return current running time."""
        return self.time

    def step(self):
        """
        Update simulator time.
        """
        self.time = self.time + self.time_step
        logger.debug('step:new time:%.3f', self.time)

    def run(self):
        """
        Main loop
        """
        self.initialize_simulation()

        running = True
        while running:
            self.update()

            if self.time > self.max_time:
                running = False

        self.end_episode()

    def update(self, steps=1):
        """
        Responsible for updating the state of the simulation.

        First, the time is moved forward. Then, the environment is updated according to
        dynamics. Then, checks are done to see if new events should be scheduled. After events
        are scheduled and the environment state is updated appropriately, events are handled.

        Parameters
        ----------
        steps : int
            number of timesteps to simulate
        """
        for _ in range(steps):
            self.step()
            self.environment.update(self)
            self.environment.process_actions(self)
            # self.environment.observe(self)
            self.process_events()
            self.environment.complete_actions(self)

    def process_events(self):
        """
        Process events that need to be processed.
        """
        try:
            while self.time >= self.events[0].time:
                # handle event
                event = heapq.heappop(self.events)
                event.execute(self)
                logger.info('%s handled.', event)
        # no events in queue
        except IndexError:
            pass
        
    def end_episode(self):
        """
        Handle everything that needs to be handled when episode ends.        
        """
        # TODO: shutdown episode and possibly restart new one depending on parameters
        self.environment.end_episode()


if __name__ == '__main__':
    sim = Simulator()
    # sim.run()
    sim.initialize_simulation()
    env = sim.environment
    elev = env.elevators[0]
    # sim.update(steps=2850)
    sim.update(steps=3049)
    sim.update(steps=100)
    # sim.update(steps=10)
    # sim.insert(events.PassengerArrivalEvent(time=sim.now(), floor=sim.environment.floors[3]))
    # sim.update()
    # sim.insert(events.PassengerArrivalEvent(time=sim.now(), floor=sim.environment.floors[3]))
    # sim.update()