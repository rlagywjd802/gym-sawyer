import numpy as np

from sawyer.ros.envs.ros_env import RosEnv
from sawyer.ros.util.common import rate_limited
from sawyer.garage.envs.base import Step
try:
    from sawyer.garage.config import STEP_FREQ
except ImportError:
    raise NotImplementedError(
        "Please set STEP_FREQ in sawyer.garage.config_personal.py!"
        "example 1: "
        "   STEP_FREQ = 5")


class SawyerEnv(RosEnv):
    def __init__(self,
                 simulated,
                 control_cost_coeff=0.,
                 action_scale=1.):
        RosEnv.__init__(self, simulated=simulated)
        self._control_cost_coeff = control_cost_coeff
        self._action_scale = action_scale

    def _initial_setup(self):
        self._robot.reset()
        self._world.initialize()

    def shutdown(self):
        self._world.terminate()

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation :
            the initial observation of the space.
            (Initial reward is assumed to be 0.)
        """
        self._robot.reset()
        self._world.reset()
        self.goal = self.sample_goal()
        initial_observation = self.get_observation().observation
        return initial_observation

    @rate_limited(STEP_FREQ)
    def step(self, action):
        """
        Perform a step in gazebo. When end of episode
        is reached, reset() should be called to reset
        the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation :
            agent's observation of the current environment
        reward: float
            amount of reward due to the previous action
        done :
            a boolean, indicating whether the episode has ended
        info :
            a dictionary containing other diagnostic information
            from the previous action
        """
        action = np.concatenate((action, np.zeros(1)))
        assert action.shape == self.action_space.shape
        a = action.copy()  # Note: you MUST copy the action if you modify it
        a *= self._action_scale
        a = np.clip(a, self.action_space.low, self.action_space.high)
        self._robot.send_command(a)

        obs = self.get_observation()

        reward = self.reward(obs.achieved_goal, self.goal)
        reward -= self._control_cost_coeff * np.linalg.norm(a)
        done = self.done(obs.achieved_goal, self.goal)
        next_observation = obs.observation
        return Step(observation=next_observation, reward=reward, done=done)

    @property
    def action_space(self):
        return self._robot.action_space
