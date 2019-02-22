"""TransitionEnv task for the Sawyer robot."""
import rospy
import numpy as np
import h5py
from sawyer.ros.envs.sawyer import TransitionPickEnv, TransitionPlaceEnv, TransitionPickAndPlaceEnv

rospy.init_node('test_transition_env')
transition_pick_env = TransitionPickEnv(simulated=False, control_mode='task_space')
transition_place_env = TransitionPlaceEnv(simulated=False, control_mode='task_space')
transition_pick_and_place_env = TransitionPickAndPlaceEnv(simulated=False, control_mode='effort')

def get_action():
    a4 = np.random.uniform(-2, 2)
    a5 = np.random.uniform(-2, 2)
    a6 = np.random.uniform(-2, 2)
    action = np.array([0, 0, 0, 0, a4, a5, a6, 0])
    return action

i = 0
n = 'N'
f = h5py.File("/root/code/garage/data/testfile.hdf5", "w")

observations = []
while n != '':
    # run pick primitive
    print("======================")
    print("Run Pick Primitive")
    obs = transition_pick_env.get_obs()
    transition_pick_env.act(obs)

    # run transition policy
    print("======================")
    print("Run Transition Policy")
    while i<2:
        action = get_action()
        obs, r, done, info = transition_pick_and_place_env.step(action)
        observations.append(obs)
        print(action)
        print(obs) 
        print(r)
        print(done)
        print(info)
        i += 1
    
    # run place primitive
    print("======================")
    print("Run Place Primitive")
    obs = transition_place_env.get_obs()
    transition_place_env.act(obs)

    # reset
    print("======================")
    print("Reset")
    transition_place_env.reset()
    i = 0

    n = input("-------> Put enter to finish : ")

f.create_dataset('observations', data=observations)
