from tqdm import trange
import tensorflow as tf
import numpy as np
from math import pi

architecture = (100, 100, 100)

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

previous1 = data
previous2 = data

for num_nodes in architecture:
    previous1 = tf.layers.dense(previous1, num_nodes, activation=tf.nn.tanh)
    previous2 = tf.layers.dense(previous2, num_nodes, activation=tf.nn.tanh)

n1 = tf.layers.dense(previous1, 1)
n2 = tf.layers.dense(previous2, 1)
u1 = tf.sin(pi * x_tf) + t_tf * x_tf * (1 - x_tf) * n1
u2 = tf.sin(pi * x_tf) * (1 + t_tf * n2)
u_exact = tf.exp(-pi ** t * t_tf) * tf.sin(pi * x_tf)

du1dx, du1dt = tf.gradients(u1, [x_tf, t_tf])
du1dx2 = tf.gradients(du1dx, [x_tf])[0]
du2dx, du2dt = tf.gradients(u2, [x_tf, t_tf])
du2dx2 = tf.gradients(du2dx, [x_tf])[0]

cost1 = tf.math.reduce_mean((du1dx2 - du1dt) ** 2)
cost2 = tf.math.reduce_mean((du2dx2 - du2dt) ** 2)

error1 = tf.math.reduce_mean((u1 - u_exact) ** 2)
error2 = tf.math.reduce_mean((u2 - u_exact) ** 2)

minimiser1 = tf.train.AdamOptimizer()
minimiser2 = tf.train.AdamOptimizer()
minimisation1 = minimiser1.minimize(cost1)
minimisation2 = minimiser2.minimize(cost2)

x = np.arange(Nx + 1) * dx
Nx += 1
Nt += 1

init = tf.global_variables_initializer()
cost_file = open("data/nn_cost.dat", "w")
max_epochs = 10000
with tf.Session() as s:
    init.run()
    cost_file.write(
        "0 %g %g %g %g\n" % (s.run(cost1), s.run(error1), s.run(cost2), s.run(error2))
    )
    for i in trange(1, int(max_epochs / 10) + 1):
        for j in range(10):
            s.run(minimisation1)
            s.run(minimisation2)
        cost_file.write(
            "%d %g %g %g %g\n"
            % (10 * i, s.run(cost1), s.run(error1), s.run(cost2), s.run(error2))
        )
        if i == 10 or i == max_epochs / 10:
            u1_res = s.run(u1).reshape((Nx, Nt))
            u2_res = s.run(u2).reshape((Nx, Nt))
            output = np.column_stack(
                (
                    x,
                    u1_res[:, 0],
                    u1_res[:, int(Nt / 2)],
                    u1_res[:, -1],
                    u2_res[:, 0],
                    u2_res[:, int(Nt / 2)],
                    u2_res[:, -1],
                )
            )
            np.savetxt("data/nn_u_%d.dat" % (10 * i), output)
cost_file.close()
