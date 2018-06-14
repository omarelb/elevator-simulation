import heapq
import logging
import configparser  # parse configuration files
import numpy.random as rnd
import os
import random

from os.path import join

import constants as const
import events
import learningAgents
from environment import Environment, Floor, ElevatorState, Passenger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler(join(const.LOG_DIR, 'simulator.log'), mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class Simulator:
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
    def __init__(self, seed=42, max_time=60 * 60, **args):
        self.time_step = const.TIME_STEP
        self.time = 0
        self.events = []
        self.seed = seed
        self.max_time = max_time
        random.seed(a=seed)
        # self.environment = Environment(kwargs['num_floors'], kwargs['num_elevators'], kwargs['traffic_profile'],
        #                                kwargs['interfloor'], kwargs['controller'])
        self.environment = Environment(**args)

    def start_episode(self):
        """
        Start off the simulation by generating the first arrivals. No arrivals on ground floor.
        """
        self.reset()
        self.environment.start_episode(self)

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
        self.start_episode()
        try:
            self.update(steps=int(self.max_time * const.STEPS_PER_SECOND))
        except KeyboardInterrupt:
            self.stop_episode()
            quit()
        self.stop_episode()

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
            while self.time >= self.events[0].time - const.GENERAL_EPS:
                # handle event
                event = heapq.heappop(self.events)
                event.execute(self)
                logger.info('%s handled.', event)
        # no events in queue
        except IndexError:
            pass

    def stop_episode(self):
        """
        Handle everything that needs to be handled when episode ends.
        """
        # TODO: shutdown episode and possibly restart new one depending on parameters
        self.environment.stop_episode()

    def reset(self):
        """
        Reset simulator time and events.
        """
        self.time = 0
        self.events = []


def parse_config(config_filename):
    """
    Load configuration file into variables.
    """
    config = configparser.ConfigParser()
    config.read(config_filename)
    args = {}
    args['max_time'] = int(config['simulation']['max_time'])
    args['seed'] = int(config['simulation']['seed'])
    args['num_elevators'] = int(config['environment']['num_elevators'])
    args['num_floors'] = int(config['environment']['num_floors'])
    args['controller'] = str(config['elevator']['controller'])
    args['traffic_profile'] = str(config['traffic_profile']['type'])
    args['interfloor'] = float(config['traffic_profile']['interfloor'])
    args['use_q_file'] = config['learning'].getboolean('use_q_file')
    args['data_dir'] = config['learning']['data_dir']
    if not os.path.isdir(args['data_dir']):
        os.mkdir(args['data_dir'])
    args['q_file'] = join(args['data_dir'], config['learning']['q_file'])
    args['annealing_factor'] = float(config['learning']['annealing_factor'])
    args['is_training'] = config['learning'].getboolean('is_training')
    args['num_testing_episodes'] = int(config['learning']['num_testing_episodes'])

    return args


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='config.ini', help='simulation configuration filename')
    parser.add_argument("-v", "--verbose", action='store_true', help='include this if you want a more verbose output')
    parser.add_argument("-n", "--num_episodes", type=int, help='number of episodes to run. overrides calculated number of episodes from annealing factor.')
    parsed_args = parser.parse_args()
    config_file = parsed_args.config

    args = parse_config(config_file)
    if parsed_args.num_episodes:
        args['num_episodes'] = parsed_args.num_episodes
    args['verbose'] = parsed_args.verbose
    sim = Simulator(**args)
    if args['is_training'] and isinstance(sim.environment.elevators[0].controller, learningAgents.ReinforcementAgent):
        num_episodes = (sim.environment.elevators[0].controller.num_training -
                        sim.environment.elevators[0].controller.episodes_so_far)
    else:
        num_episodes = args['num_testing_episodes']
    
    for i in range(num_episodes):
        print('starting episode {}'.format(i))
        sim.run()

    print('simulation done')
