import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env.reset()
random_episodes = 0
reward_sum = 0
learning_rates =0.1
dis = .99

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

X = tf.placeholder(tf.float32,shape=[None,input_size])
W1 = tf.get_variable(name="W1",shape=[input_size,output_size],initializer=tf.contrib.layers.xavier_initializer())
Y = tf.placeholder(tf.float32,shape=[None,output_size])

Qpred = tf.matmul(X,W1)

loss = tf.reduce_sum(tf.square(Y-Qpred))
train = tf.train.AdamOptimizer(learning_rate=learning_rates).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

num_episodes = 5000
rList = []

for i in range(num_episodes):
    s = env.reset()
    e = 1. /((i/10)+1)
    step_count = 0
    rAll = 0
    done = False

    while not done:
        step_count += 1
        x = np.reshape(s,[1,input_size])

        Qs = sess.run(Qpred, feed_dict={X: x})

        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        s1, reward, done, info = env.step(a)

        if done:
            Qs[0, a] = -100
        else:
            x1 = np.reshape(s1, [1, input_size])
            Qs1 = sess.run(Qpred,feed_dict={X:x1})
            Qs[0,a] = reward + dis*np.max(Qs1)

        sess.run(train,feed_dict={X:x,Y:Qs})

        rAll += reward
        s = s1

    rList.append(rAll)
    print("EPisode: {} steps: {}".format(i,step_count))
    if len(rList) > 10 and np.mean(rList[-10:]) > 500:
        break

observation = env.reset()
reward_sum = 0
while True:
    env.render()

    x = np.reshape(observation,[1,input_size])
    Qs = sess.run(Qpred,feed_dict={X:x})
    a = np.argmax(Qs)

    observation, reward, done, _ = env.step(a)
    reward_sum += reward
    if done:
        print("Total Score : {}".format(reward_sum))
        break

