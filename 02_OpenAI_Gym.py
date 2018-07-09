import gym
from msvcrt import getch
from gym.envs.registration import register

#inkey = _Getch()

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


arrow_keys={72:UP,
80:DOWN,
77:RIGHT,
75:LEFT}

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)




env = gym.make("FrozenLake-v3")
observation = env.reset()
env.render()

while True:
    key = getch()
    if ord(key) == 224:
        key = ord(getch())

    if key not in arrow_keys.keys():
        print("Game aborted!")
        break
    action = arrow_keys[key]
    state,reward,done,info = env.step(action)
    env.render()
    print("State : ",state, "Action : ", action, "Reward : ", reward, "Info : ", info)

    if done:
        print("Finished with reward", reward)
        break


