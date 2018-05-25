import numpy as np
import matplotlib.pyplot as plt


def parab_acc(x, c):
    x_res = np.array([x ** 2, x, 1, 0])
    return c.dot(x_res)

def parab_vel(x, c):
    x_res = np.array([x ** 3 / 3, x ** 2 / 2, x, 1])
    return c.dot(x_res)

def parab_pos(x, c):
    x_res = np.array([x**4 / 12, x**3 / 6, x ** 2 / 2, x])
    return c.dot(x_res)

t_star = 2.3975474

# a_0 = 0.849329981441
a_0 = 0.957518301704
v_0 = 1.94148244737

t_2 = 1.679 # seconds
x_t = 1.83

# A = np.array([[t_2**2, t_2, 0],
#               [t_2**3 / 3, t_2**2 / 2, 0],
#               [t_2**4 / 12, t_2**3 / 6, 1]])
# b = np.array([-a_0, -(a_0 * t_2 + v_0), x_t -(a_0 * t_2**2 / 2 + v_0 * t_2)])

A = np.array(
             [
              [t_2**2, t_2, 1, 0], # acc a(0) = ?
            #   [t_2**3 / 3, t_2**2 / 2, t_2, 1], # vel : v(0) = ?
              [0, 0, 1, 0], #a(0) = a_0
              [0, 0, 0, 1], #v(0) = v_0
              [t_2**4 / 12, t_2**3 / 6, t_2**2 / 2, t_2]]) # pos : c_5 = 0

b = np.array([0, a_0, v_0, x_t])

c = np.linalg.solve(A, b)

# c = np.concatenate((c, [a_0, v_0]))

print(c)

p = parab_pos(t_2, c)
v = parab_vel(t_2, c)
a = parab_acc(t_2, c)
p_0 = parab_pos(0, c)
v_0 = parab_vel(0, c)
a_0 = parab_acc(0, c)
print(p_0, p)
print(v_0, v)
print(a_0, a)

x = np.linspace(0, t_2, 100)

# plt.plot(x, parab_acc(x, c))
# plt.plot(x, parab_vel(x, c))
# plt.plot(x, parab_pos(x, c))
# plt.axhline(y=0)
# plt.axvline(x=t_2)
# plt.legend(['acc', 'vel', 'pos'])
# plt.show()

t_end = 2
t = 0
step = 0.01

a = a_0
v = v_0
p = x_t
history = []
while t < t_end:
    history.append((t, a, v, p))
    a += (2 * c[0] * t + c[1]) * step
    v += a * step
    p += v * step
    t += step

ts = [x[0] for x in history]
ass = [x[1] for x in history]
vs = [x[2] for x in history]
ps = [x[3] for x in history]

plt.figure()
plt.plot(ts, ass)
plt.plot(ts, vs)
plt.plot(ts, ps)
plt.axhline(y=0)
plt.axhline(y=x_t * 2)
plt.axvline(x=t_2)
plt.legend(['acc', 'vel', 'pos'])
plt.show()