"""TransitionEnv task for the Sawyer robot."""
import rospy
import numpy as np
from sawyer.ros.envs.sawyer import TransitionPickEnv, TransitionPlaceEnv

rospy.init_node('test_transition_env')
transition_env = TransitionPickEnv(simulated=False, control_mode='task_space')

obs = transition_env.get_obs()
transition_env.act(obs)
