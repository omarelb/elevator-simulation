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


# motion dynamics
ACCEL_TIME = 3.595 # seconds to accelerate to full speed from zero.
MAX_SPEED = 2.54 # m/s
# motion for accelerating and decelerating from full speed: a(t) = sin(C*t) <- C 
ACCEL_CONST = 0.8871057 
# decision point distance for elevator at full speed
FULL_SPEED_DECISION_DIST = 4.6363218403
# decision point distance for accelerating elevator
ACCEL_DECISION_DIST = 1.83 # FLOOR_HEIGHT / 2

# parameters for decelerating while accelerating at half floor height
# parabola: a(t) = c_1*x^2 + c_2 * x + c_3
ACCEL_DECEL = (3.51757258, -6.4762952, 0.9575183, 1.94148245)


# other
SECONDS_PER_MINUTE = 60

# numerical help with comparisons
GENERAL_EPS = 0.0001