import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#model of covid19 in north carolina

#model: SEIR--susceptible, exposed, infected, removed
#s(t) = -b(t)s(t)i(t)
#e(t) = b(t)s(t)i(t) - theta e(t)
#b(t) = transmission rate
#r = 1 - s - e - i
#theta(t) = infection rate
#c = i + r (all those who have had the infection)
#x1 = F(x, t), x1 := (s, e, i)
#theta = 1/5 (incubation oof 5.days)
#y = 1/18 (illness duration of 18 days)
#b(t) := R(t)y

pop_size = 3.3e8 
y = 1/18
theta = 1/ 5

def F(x, t, R0 = 1.6):
	s, e, i = x
	b = R0(t) if callable (R0) else R0 * y
	ne = b * s* i

	ds = - ne
	de = ne - theta * e
	di = theta * e - y * i

	return ds, de, di

i_0 = 1e-7
e_0 = 4* i_0
s_0 = 1 - i_0 - e_0

x_0 = s_0, e_0, i_0

def solve_path(R0, t_vec, x_init=x_0):
	G = lambda x, t: F(x, t, R0)
	s_path, e_path, i_path = odeint(G, x_init, t_vec).transpose()
	c_path = 1 - s_path - e_path

	return i_path, c_path

t_length = 700
grid_size = 1000
t_vec = np.linspace(0, t_length, grid_size)

R0_vals = np.linspace(1.6, 3.0, 5)
labels = [f'$R0 = {r:.2f}$' for r in R0_vals]
i_paths, c_paths = [], []

for r in R0_vals:
	i_path, c_path = solve_path(r, t_vec)
	i_paths.append(i_path)
	c_paths.append(c_path)

def plot_paths(paths, labels, times=t_vec):
	fig, ax = plt.subplots()
	for path, label in zip(paths, labels):
		ax.plot(times, path, label=label)
	ax.legend(loc='upper left')
	plt.show()

plot_paths(i_paths, labels)

