import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tqdm import trange
import tensorflow as tf
import numpy as np
from math import pi


# ===== Values for parameter sweep ===== #
activation_functions = [
    (tf.nn.sigmoid, "sigmoid"),
    (tf.nn.relu, "relu"),
    (tf.nn.tanh, "tanh"),
]
learning_rates = [10 ** i for i in range(-4, 1)]
momentums = [0] + [10 ** i for i in range(-4, 1)]

optimisers = [
    tf.train.MomentumOptimizer(learning_rate, momentum)
    for learning_rate in learning_rates
    for momentum in momentums
] + [tf.train.AdamOptimizer()]

optimiser_names = [
    r"$\alpha=\num{%g},\gamma=\num{%g}$" % (learning_rate, momentum)
    for learning_rate in learning_rates
    for momentum in momentums
] + ["Adam"]
architectures = [[10 for i in range(N)] for N in range(1, 8)] + [
    [1000],
    [100],
    [100, 100],
    [100, 10, 10],
    [10, 100, 10],
    [10, 10, 100],
]

num_activations = len(activation_functions)
num_optimisers = len(optimisers)
num_architectures = len(architectures)

# ====================================== #

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


outfile = open("data/nn_raw_table.dat", "w")
outfile.write("# Activation Optimiser Architecture Cost Error\n")

# ===== Parameter sweep ===== #
for i in trange(num_activations, desc="Activation functions:"):
    activation, activation_name = activation_functions[i]
    for j in trange(num_optimisers, desc="Optimisers:"):
        optimiser = optimisers[j]
        for k in trange(num_architectures, desc="Architectures:"):
            architecture = architectures[k]

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
            # """
            with tf.Session() as s:
                s.run(init)
                # """
                for _ in trange(10000, desc="Epochs:"):
                    s.run(minimisation)

                outfile.write(
                    "%s %s %s %g %g\n"
                    % (
                        activation_name,
                        optimiser_names[j],
                        str(architecture).replace(" ", ""),
                        s.run(cost),
                        s.run(error),
                    )
                )
                # """
            # """
outfile.close()
# =========================== #
