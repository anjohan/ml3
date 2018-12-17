import sys
from tqdm import trange
import tensorflow as tf
import numpy as np
from math import pi

variant = int(sys.argv[1])
architecture = [100, 100, 100]

dx = 0.05
dt = 0.005
T = 0.2

Nx = int(round(1 / dx))
Nt = int(round(T / dt))

x = np.arange(Nx + 1) * dx
t = np.arange(Nt + 1) * dt

x, t = np.meshgrid(x, t, indexing="ij")

x = x.ravel()
t = t.ravel()
u_exact = np.exp(-pi ** 2 * t) * np.sin(pi * x)

data = np.column_stack((x, t))
x_tf = tf.convert_to_tensor(x)
t_tf = tf.convert_to_tensor(t)
x_tf = tf.reshape(x_tf, shape=(-1, 1))
t_tf = tf.reshape(t_tf, shape=(-1, 1))
data = tf.convert_to_tensor(data)

print(data.shape)

previous = data

for num_nodes in architecture:
    previous = tf.layers.dense(previous, num_nodes, activation=tf.nn.tanh)

n = tf.layers.dense(previous, 1)
if variant == 1:
    u = tf.sin(pi * x_tf) + t_tf * x_tf * (1 - x_tf) * n
else:
    u = tf.sin(pi * x_tf) * (1 + t_tf * n)

u_exact = tf.exp(-pi ** 2 * t_tf) * tf.sin(pi * x_tf)

dudx, dudt = tf.gradients(u, [x_tf, t_tf])
dudx2 = tf.gradients(dudx, [x_tf])[0]

zeros = tf.zeros_like(u)

cost = tf.losses.mean_squared_error(zeros, dudx2 - dudt)

error = tf.math.reduce_mean((u - u_exact) ** 2)

minimiser = tf.train.AdamOptimizer()
minimisation = minimiser.minimize(cost)

x = np.arange(Nx + 1) * dx
Nx += 1
Nt += 1

init = tf.global_variables_initializer()
cost_file = open("data/nn_cost_" + str(variant) + ".dat", "w")
max_epochs = 10000
with tf.Session() as s:
    init.run()
    cost_file.write("0 %g %g\n" % (s.run(cost), s.run(error)))
    for i in trange(1, int(max_epochs / 10) + 1):
        for j in range(10):
            s.run(minimisation)
        cost_file.write("%d %g %g\n" % (10 * i, s.run(cost), s.run(error)))
        if i == 10 or i == max_epochs / 10:
            u_res = s.run(u).reshape((Nx, Nt))
            output = np.column_stack(
                (x, u_res[:, 0], u_res[:, int(Nt / 2)], u_res[:, -1])
            )
            np.savetxt("data/nn_u_%d_%d.dat" % (10 * i, variant), output)
cost_file.close()
