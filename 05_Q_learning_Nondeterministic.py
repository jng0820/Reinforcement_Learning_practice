import gym
#from gym.envs.registration import register
import numpy as np
import random as pr
import matplotlib.pyplot as plt

# register(
#     id='FrozenLake-v0',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name': '4x4', 'is_slippery': True}
# )

env = gym.make("FrozenLake-v0")
#env.render()
Q = np.zeros([env.observation_space.n,env.action_space.n])

num_episodes = 2000
gamma = 0.99
learning_rates = 0.85
r_List = []

for i in range(num_episodes):
    e = 1./((i//100)+1)
    state = env.reset()
    rAll = 0
    done = False
    while not done:
        # if np.random.rand(1) < e:
        #     action = env.action_space.sample()
        # else:
        #     action = np.argmax(Q[state,:]+np.random.rand(1,env.action_space.n)/(i+1))

        action = np.argmax(Q[state, :] + np.random.rand(1, env.action_space.n) / (i + 1))
        new_state , reward,done,_ = env.step(action)

        #Q[state, action] = (reward + gamma * np.max(Q[new_state, :]))
        Q[state,action] = (1-learning_rates)*Q[state,action] +learning_rates*(reward + gamma * np.max(Q[new_state,:]))
        rAll += reward
        state = new_state

    r_List.append(rAll)

print("Success rate : "+ str(sum(r_List)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)

plt.bar(range(len(r_List)),r_List,color='blue')
plt.show()