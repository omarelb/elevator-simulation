"""
Define constants to be used in all files.
"""

# building constants
NUM_FLOORS = 5
ARRIVAL_RATES = [1] * 5
FLOOR_HEIGHT = 3.66

# elevator constants
STOP_TIME = 1
LOAD_TIME = 1
FLOOR_TIME = 1

# traffic constants
# DOWNPEAK_RATES = (1, 2, 4, 4, 18, 12, 8, 7, 18, 5, 3, 2)
DOWNPEAK_RATES = (0.25, 0.5, 1, 1, 4.5, 3, 2, 1.75, 4.5, 1.25, 0.75, 0.5)

# motion dynamics
ACCEL_TIME = 3.595  # seconds to accelerate to full speed from zero.
MAX_SPEED = 2.54  # m/s
# motion for accelerating and decelerating from full speed: a(t) = sin(C*t) <- C 
ACCEL_CONST = 0.8871057
# decision point distance for elevator at full speed
# to next floor
# FULL_SPEED_DECISION_DIST = 4.6363218403
# from last floor
FULL_SPEED_DECISION_DIST = 2.6836781597
# decision point distance for accelerating elevator
ACCEL_DECISION_DIST = 1.83  # FLOOR_HEIGHT / 2

# parameters for decelerating while accelerating at half floor height
# parabola: a(t) = c_1*x^2 + c_2 * x + c_3
#           --> da(t) \approx  2*c_1*x + c_2
ACCEL_DECEL = (3.51757258, -6.4762952, 0.9575183, 1.94148245)
# c = [3.36406177, -6.15411438, 0.84932998, 1.94148245] <-- alternative


# other
TIME_STEP = 0.01  # in seconds
STEPS_PER_SECOND = int(round(1 / 0.01))
SECONDS_PER_MINUTE = 60
MILLISECONDS_PER_SECOND = 1000
MINUTES_PER_TIME_INTERVAL = 5
NUM_EPS_UPDATE = 5

# numerical help with comparisons
GENERAL_EPS = 0.0001

LOG_DIR = 'logs'
MAP_CONST_STR = {-1: 'DOWN', 0: 'STOPPED', 1: 'UP', 2: 'IDLE', 3: 'ACCELERATING',
                 4: 'FULL_SPEED_DECELERATING', 5: 'ACCEL_DECELERATING', 6: 'FULL_SPEED',
                 7: 'BOARDING', 8: 'STOP', 9: 'CONTINUE', 10: 'NO_ACTION', 11: 'MOVE_UP',
                 12: 'MOVE_DOWN', 13: 'DONE_BOARDING'}