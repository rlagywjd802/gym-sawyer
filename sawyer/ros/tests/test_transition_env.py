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
j = 0
n = 'N'
f = h5py.File("/root/code/transition-robot/gym-sawyer/data/testfile.hdf5", "w")

observations = []

while n != '':
    # run pick primitive
    if n not in ['2', '3', '4']:
        print("\n================================================")
        print("Run Pick Primitive")
        inital_obs = transition_pick_env.get_obs()
        transition_pick_env.act(inital_obs)

    # run transition policy
    if n not in ['1', '3', '4']:
        print("\n================================================")
        print("Run Transition Policy")
        while i<0:
            action = get_action()
            obs, r, done, info = transition_pick_and_place_env.step(action)
            observations.append(obs)
            print(action)
            print(obs)
            print(r)
            print(done)
            print(info)
            i += 1
    
    done = False    
    # run place primitive
    if n not in ['1', '2', '4']:
        print("\n================================================")
        print("Run Place Primitive")
        while not done:
            action = np.array([0.0, 0.0, -0.015, -1.0])
            obs, r, done, info = transition_place_env.step(action)
            print(action)
            print(obs) 
            print(r)
            print(done)
            print("has_peg: "+str(info['grasped_peg']))
            print("is_success: "+str(info['success']))
            print("is_done: "+str(info['d']))
            j += 1

    # reset
    if n not in ['1', '2', '3']:
        print("\n================================================")
        print("Reset")
        print(inital_obs)
        transition_pick_and_place_env.reset()
        
    i = 0
    j = 0

    n = input("-------> Put enter to finish : ")

f.create_dataset('observations', data=observations)
