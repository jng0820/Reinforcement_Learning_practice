import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt


def one_hot(x):
    return np.identity(16)[x:x + 1]

env = gym.make("FrozenLake-v0")
input_size = env.observation_space.n
output_size = env.action_space.n

learning_rates = 0.1
dis = 0.99

X = tf.placeholder(shape=[1,input_size],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([input_size,output_size], 0, 0.01))
Y = tf.placeholder(shape=[1,output_size],dtype=tf.float32)

Qpred = tf.matmul(X,W)

loss = tf.reduce_sum(tf.square(Y-Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rates).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

num_episodes = 2000
rList = []

for i in range(num_episodes):
    s = env.reset()
    e = 1. /((i/50)+10)
    rAll = 0
    done = False
    local_loss = []
    while not done:
        Qs = sess.run(Qpred, feed_dict={X: one_hot(s)})
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        s1, reward, done, info = env.step(a)

        if done:
            Qs[0, a] = reward
        else:
            Qs1 = sess.run(Qpred, feed_dict={X: one_hot(s1)})
            Qs[0, a] = reward + dis * np.max(Qs1)

        sess.run(train,feed_dict={X:one_hot(s),Y:Qs})

        rAll += reward
        s = s1

    rList.append(rAll)
    local_loss.append(loss)

print("Success rate : "+ str(sum(rList)/num_episodes))

plt.bar(range(len(rList)),rList,color='blue')
plt.show()