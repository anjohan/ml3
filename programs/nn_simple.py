import tensorflow as tf
import numpy as np
from math import pi

architecture = (10, 10)

dx = 0.01
dt = 0.5*dx**2
T = 0.2

Nx = int(round(1/dx))
Nt = int(round(T/dt))

x = np.arange(Nx+1)*dx
t = np.arange(Nt+1)*dt

x, t = np.meshgrid(x, t, indexing="ij")
print(Nx, Nt, x.shape)

x = x.ravel()
t = t.ravel()

data = np.column_stack((x, t))
x_tf = tf.convert_to_tensor(x)
t_tf = tf.convert_to_tensor(t)
x_tf = tf.reshape(x_tf, shape=(-1, 1))
t_tf = tf.reshape(t_tf, shape=(-1, 1))
data = tf.convert_to_tensor(data)

print(data.shape)

previous = data

for num_nodes in architecture:
    previous = tf.layers.dense(previous, num_nodes, activation=tf.nn.relu)

n = tf.layers.dense(previous, 1)
u = tf.sin(pi*x_tf) + t_tf*x_tf*(1-x_tf)*n

dudx, dudt = tf.gradients(u, [x_tf, t_tf])
dudx2 = tf.gradients(dudx, [x_tf])[0]

cost = tf.math.reduce_sum((dudx2 - dudt)**2)
minimiser = tf.train.AdadeltaOptimizer(1.0)
minimisation = minimiser.minimize(cost)
print(dudx2)

print(dudx.shape, dudt.shape, dudx2.shape)

init = tf.global_variables_initializer()
with tf.Session() as s:
    init.run()
    for i in range(100):
        s.run(minimisation)
        print(s.run(cost))
    u = s.run(u)

x = np.arange(Nx+1)*dx
Nx += 1
Nt += 1
u = u.reshape((Nx, Nt))
output = np.column_stack((x, u[:,0], u[:,int(Nt/2)], u[:,-1]))
np.savetxt("data/nn_test.dat", output)
