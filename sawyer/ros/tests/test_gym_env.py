import rospy
import numpy as np
from sawyer.ros.envs.sawyer import ToyEnv

rospy.init_node('test_gym_env')
toy_env = ToyEnv(simulated=False, control_mode='task_space')

actions = []
x = np.array([0.1, 0, 0, 1])
y = np.array([0, 0.1, 0, 1])
z = np.array([0, 0, -0.07, 1])
z_ = np.array([0, 0, +1.0, -1])
g1 = np.array([0, 0, 0, -1])
g2 = np.array([0, 0, 0, 1])
actions.append(g2)
actions.append(x)
actions.append(y)
#actions.append(z)
#actions.append(g1)
#actions.append(z_)

i = 0
while i < 3:
    obs, r, done, info = toy_env.step(actions[i])
    print("==============", i, "==============")
    print(obs)
    #print(r)
    print(done)
    #print(info)
    i += 1

#toy_env.reset()