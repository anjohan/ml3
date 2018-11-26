import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tqdm import trange
import tensorflow as tf
import numpy as np
from math import pi
import sys

input = sys.argv[1]

activation, optimiser, architecture = input.split("_")

activation = eval("tf.nn." + activation)

if optimiser == "Adam":
    optimiser = tf.train.AdamOptimizer()
else:
    learning_rate, momentum = eval(optimiser)
    optimiser = tf.train.MomentumOptimizer(learning_rate, momentum)

architecture = eval(architecture)

if not isinstance(architecture, tuple):
    architecture = [architecture]

# print(activation, optimiser, architecture)
# sys.exit()

# ===== Input data ===== #
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

data = np.column_stack((x, t))
x_tf = tf.convert_to_tensor(x)
t_tf = tf.convert_to_tensor(t)
x_tf = tf.reshape(x_tf, shape=(-1, 1))
t_tf = tf.reshape(t_tf, shape=(-1, 1))
u_exact = tf.exp(-pi ** 2 * t_tf) * tf.sin(pi * x_tf)
data = tf.convert_to_tensor(data)
# ======================= #

# Feed forward network:
previous = data
for num_nodes in architecture:
    previous = tf.layers.dense(previous, num_nodes, activation=activation)

n = tf.layers.dense(previous, 1)
u = tf.sin(pi * x_tf) + t_tf * x_tf * (1 - x_tf) * n

dudx, dudt = tf.gradients(u, [x_tf, t_tf])
dudx2 = tf.gradients(dudx, [x_tf])[0]

cost = tf.math.reduce_sum((dudx2 - dudt) ** 2)
minimisation = optimiser.minimize(cost)

error = tf.math.reduce_sum((u - u_exact) ** 2)

init = tf.global_variables_initializer()
with tf.Session() as s:
    s.run(init)
    for _ in trange(10000, desc="Epochs:"):
        s.run(minimisation)

    with open("data/nn_cost_" + input + ".dat", "w") as outfile:
        outfile.write(
            " ".join(input.split("_")) + " %g %g\n" % (s.run(cost), s.run(error))
        )
    u = s.run(u)
    Nx += 1
    Nt += 1
    u = u.reshape((Nx, Nt))
    output = np.column_stack((np.arange(Nx) * dx, u[:, 0], u[:, int(Nt / 2)], u[:, -1]))
    np.savetxt("data/nn_u_%s.dat" % input, output)
