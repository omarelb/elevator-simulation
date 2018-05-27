import numpy as np
import math
from scipy.optimize import newton 
import matplotlib.pyplot as plt

ACCEL_TIME = 3.595 # seconds
C = 0.8871057

FLOOR_HEIGHT = 3.66

step = 0.01 # seconds

class Particle():
    def __init__(self, time=0, step=0.01, acc=0, vel=0, pos=0):
        self.acc = acc
        self.vel = vel
        self.pos = pos
        self.time = time
        self.dt = step
        self.history = []
        self.direction = 1
        self.decision_time = 0
        self.decision_made = False
        self.acc_update = self.dacc

    def dacc(self):
        return math.cos(C * self.time) * self.dt * (self.time < ACCEL_TIME)

    def dacc2(self):
        # stop time = 2 seconds
        # c = [2.09310932, -4.61088363,  0.84932998,  1.94148245]
        # stop time = 3 seconds
        # c = [0.71455054, -2.42676161, 0.84932998, 1.94148245]
        # stop time = 1.679 seconds
        # c = [3.36406177, -6.15411438, 0.84932998, 1.94148245]
        c = [3.51757258, -6.4762952, 0.9575183, 1.94148245]
        x = self.time - self.decision_time

        return (2 * c[0] * x + c[1]) * self.dt

    def acc_(self):
        return math.sin(C * self.time) * (self.time < ACCEL_TIME)

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
                print(self.time)
                print(self.acc)
                print(self.vel)
                print(self.pos)
                self.decision_time = self.time
                self.decision_made = True
        else:
            self.acc_update = self.dacc
        self.acc += self.direction * self.acc_update()
        self.vel += self.dvel()
        self.pos += self.dpos()
    

    def run(self, end_time):
        while self.time < end_time:
            self.history.append((self.time, self.acc, self.vel, self.pos))
            self.update()
            self.time += self.dt

def acceleration(times):
    return np.sin(C * times)

def velocity(times):
    return - 1 / C**2 * (np.cos(C * times) - 1)

def position(times):
    return - 1 / C**2 * (1 / C * np.sin(C * times) - times)


end_time = 4.5 # seconds
p = Particle(step=step)
p.run(end_time)

times = [x[0] for x in p.history]
accs = [x[1] for x in p.history]
vels = [x[2] for x in p.history]
positions = [x[3] for x in p.history]

# plotting
# fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

# ax.plot(times, accs, linewidth=4, linestyle='--', color='black')
# ax.plot(times, vels, linewidth=4, linestyle=':', color='red')
# ax.plot(times, positions, linewidth=4, linestyle='-')
# ax.axhline(y=FLOOR_HEIGHT / 2)
# ax.axhline(y=FLOOR_HEIGHT * 1)
# ax.set_xlim((0, 2.3975 + 1.679))
# ax.set_xlabel('Time (s)', fontsize=18)
# ax.set_yticks((-2, -1, 0, 1, FLOOR_HEIGHT / 2, 3, FLOOR_HEIGHT))
# ax.axhline(y=FLOOR_HEIGHT * 1.5)
# ax.axhline(y=FLOOR_HEIGHT * 2)
# ax.axhline(y=2.54, color='black')
# ax.axvline(x=3.595)
# ax.axvline(x=2.3975)
# ax.axvline(x=2.3975 + 1.679)
# ax.axhline(y=0, color='black', lw=3)
# plt.setp(ax.get_yticklabels(), fontsize=14)
# plt.setp(ax.get_xticklabels(), fontsize=14)

# plt.legend(['acceleration', 'velocity', 'position'], prop={'size': 12})

# plt.show()