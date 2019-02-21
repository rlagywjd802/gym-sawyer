"""TransitionEnv task for the Sawyer robot."""
import rospy
import numpy as np
from sawyer.ros.envs.sawyer import TransitionPickEnv, TransitionPlaceEnv

rospy.init_node('test_transition_env')
transition_pick_env = TransitionPickEnv(simulated=False, control_mode='task_space')
transition_place_env = TransitionPlaceEnv(simulated=False, control_mode='task_space')

# run pick primitive
obs = transition_pick_env.get_obs()
transition_pick_env.act(obs)

# run place primitive
obs = transition_place_env.get_obs()
transition_place_env.act(obs)
