import gym
from gym.envs.registration import register
import numpy as np
import random as pr
import matplotlib.pyplot as plt

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

env = gym.make("FrozenLake-v3")
env.render()
Q = np.zeros([env.observation_space.n,env.action_space.n])
num_episodes = 2000

rList = []

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        #arg들의 값이 같을경우 랜덤으로
        action = rargmax(Q[state,:])

        #action에 따른 정보 얻어옴
        new_state, reward, done, _ = env.step(action)

        #Q-Table Update
        Q[state,action] = reward + np.max(Q[new_state,:])

        #상태 변경
        state = new_state
        rAll += reward
        env.render()

    rList.append(rAll)

print("Successful rate : " + str(sum(rList)/num_episodes) )
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)

plt.bar(range(len(rList)),rList,color="blue")
plt.show()