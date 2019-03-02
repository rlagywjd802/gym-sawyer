"""ToyEnv task for the Sawyer robot."""

import sys
import gym
import moveit_commander
import numpy as np

from sawyer.mujoco.tasks import (TransitionPickTask,
                                 TransitionPlaceTask,
                                 TransitionPickAndPlaceTask)
from sawyer.ros.envs.sawyer.sawyer_env import SawyerEnv
from sawyer.ros.robots.sawyer import Sawyer
from sawyer.ros.worlds.toy_world import ToyWorld
from sawyer.garage.core import Serializable
from sawyer.ros.util.common import rate_limited
try:
    from sawyer.config import STEP_FREQ
except ImportError:
    raise NotImplementedError(
        "Please set STEP_FREQ in sawyer.config_personal.py!"
        "example 1: ")

class TransitionEnv(SawyerEnv, Serializable):
    def __init__(self,
                 simulated=False,
                 collision_penalty=0.,
                 terminate_on_collision=True,
                 control_mode='task_space'):
        Serializable.quick_init(self, locals())

        self.simulated = simulated
        self._step = 0
        self._collision_penalty = collision_penalty
        self._terminate_on_collision = terminate_on_collision
        self._config = {}
        self.control_mode = control_mode

        # Initialize moveit to get safety check
        moveit_commander.roscpp_initialize(sys.argv)
        self._moveit_robot = moveit_commander.RobotCommander()
        self._moveit_scene = moveit_commander.PlanningSceneInterface()
        self._moveit_group_name = 'right_arm'
        self._moveit_group = moveit_commander.MoveGroupCommander(
            self._moveit_group_name)

        self._robot = Sawyer(moveit_group=self._moveit_group_name,
                             control_mode=control_mode)
        self._world = ToyWorld(self._moveit_scene,
                               self._moveit_robot.get_planning_frame(),
                               simulated)

        SawyerEnv.__init__(self, simulated=simulated)

        self._active_task = TransitionPickAndPlaceTask()
        self.reward_type = []
        self.ob_shape = {"joint": [4], "box": [7], "goal": [3]}
        #self.ob_shape = {"joint": [4], "box": [7]}
        self.ob_type = self.ob_shape.keys()

    @property
    def observation_space(self):
        # WE WILL NOT USE IT
        spaces = []
        spaces.append(self._world.observation_space)
        spaces.append(self._robot.observation_space)

        high = np.concatenate([sp.high for sp in spaces]).ravel()
        low = np.concatenate([sp.low for sp in spaces]).ravel()
        return gym.spaces.Box(high, low, dtype=np.float32)

    @rate_limited(STEP_FREQ)
    def step(self, action, control_mode=None):
        #assert action.shape == self.action_space.shape, 'action.shape={} self.action_space.shape={}'.format(action.shape, self.action_space.shape)
        if control_mode == None:
            control_mode = self.control_mode
        # assert self._run

        # Do the action
        self._robot.send_command(action, control_mode)
        self._step += 1
        obs = self.get_obs()

        # Robot obs
        robot_obs = self._robot.get_observation()

        # World obs
        world_obs = self._world.get_observation()

        # Grasp state obs
        grasped_peg_obs = self.has_peg()

        in_collision = self._robot.in_collision_state()

        info = {
            'l': self._step,
            'action': action,
            'in_collision': in_collision,
            'robot_obs': robot_obs,
            'world_obs': world_obs,
            'gripper_position': self._robot.gripper_pose['position'],
            'gripper_state': self._robot.gripper_state,
            'grasped_peg': grasped_peg_obs,
        }

        r = 0
        done = False
        successful = False
        print(self._active_task)

        if self._active_task.is_success(obs, info):            
            r += self._active_task.completion_bonus
            done = True
            successful = True

        if in_collision:
            r -= self._collision_penalty
            if self._terminate_on_collision:
                done = True
                successful = False

        info['r'] = r
        info['d'] = done
        info['is_success'] = successful

        return obs, r, done, info

    def reset(self):
        self._step = 0
        self._goal = self.sample_goal()
        if self.simulated:
            self._robot.reset()
            self._world.reset()

        return self.get_obs()

    def sample_goal(self):
        """
        Samples a new goal and returns it.
        """
        return np.array([0.65 + np.random.uniform(-0.05, 0.05),
                         0 + np.random.uniform(-0.05, 0.05),
                         0])

    def get_obs(self):
        # Robot obs
        robot_obs = self._robot.get_observation()

        # World obs
        world_obs = self._world.get_observation()

        # Construct obs specified by observation_space
        obs = []
        obs.append(robot_obs['gripper_position'])
        obs.append(robot_obs['gripper_state'])
        obs.append(world_obs['peg_position'])
        obs.append(world_obs['peg_orientation'])
        obs.append(self._goal)
        obs = np.concatenate(obs).ravel()

        return obs

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            return {
                'joint': ob[:, :4],
                'box': ob[:, 4:11],
                'goal': ob[:, 11:14]
            }
        return {
            'joint': ob[:4],
            'box': ob[4:11],
            'goal': ob[11:14]
        }

    def done(self, achieved_goal, goal):
        """
        :return if done: bool
        """
        # WE WILL NOT USE IT
        raise NotImplementedError

    def is_terminate(self, obs, init=False, env=None):
        return self._active_task.is_terminate(obs, init)

    def get_next_primitive(self, obs, prev_primitive):
        return self._active_task.get_next_primitive(obs, prev_primitive)

    def reward(self, achieved_goal, goal):
        """
        Compute the reward for current step.
        """
        # WE WILL NOT USE IT
        raise NotImplementedError

    def has_peg(self):
        gripper_state = self._robot.gripper_state
        gripper_thereshold = -0.5
        print("has_peg : "+str(gripper_state))
        if gripper_state > gripper_thereshold:
            return False

        peg_pos = self._world.get_peg_pose().position
        gripper_pos = self._robot.gripper_pose['position']
        max_xy_diff = 0.02
        max_z_diff = 0.2
        return ( abs(peg_pos.x - gripper_pos.x) < max_xy_diff and
            abs(peg_pos.y - gripper_pos.y) < max_xy_diff and
            abs(peg_pos.z - gripper_pos.z) < max_z_diff )

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, value):
        self._goal = value


class TransitionPickEnv(TransitionEnv, Serializable):
    def __init__(self,
                 simulated=False,
                 collision_penalty=0.,
                 terminate_on_collision=True,
                 control_mode='task_space'):
        TransitionEnv.__init__(self, simulated=simulated,
                               collision_penalty=collision_penalty,
                               terminate_on_collision=terminate_on_collision,
                               control_mode=control_mode)

        self._active_task = TransitionPickTask()
        self.reward_type = []
        self.ob_shape = {"joint": [4], "box": [7]}
        self.ob_type = self.ob_shape.keys()

    def get_obs(self):
        # Robot obs
        robot_obs = self._robot.get_observation()

        # World obs
        world_obs = self._world.get_observation()

        # Construct obs specified by observation_space
        obs = []
        obs.append(robot_obs['gripper_position'])
        obs.append(robot_obs['gripper_state'])
        obs.append(world_obs['peg_position'])
        obs.append(world_obs['peg_orientation'])
        obs = np.concatenate(obs).ravel()

        return obs

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            return {
                'joint': ob[:, :4],
                'box': ob[:, 4:11],
            }
        return {
            'joint': ob[:4],
            'box': ob[4:11],
        }

    def act(self, ob):
        #if len(ob.shape) > 1:
        #    pass
        #else:
            peg_position = ob[4:11]
            print("peg_position: "+str(peg_position))

            # gripper open
            self._robot._gripper_open()
            print("gripper open")
            
            # move to the peg position
            peg_position[2] += 0.13
            self._robot._move_to_target_position(peg_position)
            print("move to the peg position")

            # go down
            peg_position[2] -= 0.11
            self._robot._move_to_target_position(peg_position)
            print("go down")

            # gripper close
            self._robot._gripper_close()
            print("gripper close")

            # go up
            peg_position[2] += 0.13
            self._robot._move_to_target_position(peg_position)
            print("go up")

            # move to the hole position
            hole_position = [0.701, -0.010, 0.170, 1.0, 0.0, 0.0, 0.0]
            hole_position[2] += 0.1
            self._robot._move_to_target_position(hole_position)
            print("move to the hole position")

            # go down
            hole_position[2] -= 0.03
            self._robot._move_to_target_position(hole_position)
            print("go down")

class TransitionPlaceEnv(TransitionEnv, Serializable):
    def __init__(self,
                 simulated=False,
                 collision_penalty=0.,
                 terminate_on_collision=True,
                 control_mode='task_space'):
        TransitionEnv.__init__(self, simulated=simulated,
                               collision_penalty=collision_penalty,
                               terminate_on_collision=terminate_on_collision,
                               control_mode=control_mode)

        self._active_task = TransitionPlaceTask()
        self.reward_type = []
        self.ob_shape = {"joint": [4], "box": [7], "goal": [3]}
        self.ob_type = self.ob_shape.keys()

        self._goal = self.sample_goal()

    def get_obs(self):
        # Robot obs
        robot_obs = self._robot.get_observation()

        # World obs
        world_obs = self._world.get_observation()

        # Construct obs specified by observation_space
        obs = []
        obs.append(robot_obs['gripper_position'])
        obs.append(robot_obs['gripper_state'])
        obs.append(world_obs['peg_position'])
        obs.append(world_obs['peg_orientation'])
        obs.append(self._goal)
        obs = np.concatenate(obs).ravel()

        return obs

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            return {
                'joint': ob[:, :4],
                'box': ob[:, 4:11],
                'goal': ob[:, 11:14]
            }
        return {
            'joint': ob[:4],
            'box': ob[4:11],
            'goal': ob[11:14]
        }

    def act(self, ob):
        print('ob from act() in place', ob)
        #if len(ob.shape) > 1:
        #    pass
        #else:
        return np.array([0.0, 0.0, -0.015, -1.0])

    def reset(self):
        # move to the hole position
        hole_position = [0.701, -0.010, 0.170, 1.0, 0.0, 0.0, 0.0]
        hole_position[2] += 0.1
        self._robot._move_to_target_position(hole_position)
        print("move to the hole position")

        # go down
        #hole_position[2] -= 0.03
        #self._robot._move_to_target_position(hole_position)
        #print("go down")

        return self.get_obs()

class TransitionPickAndPlaceEnv(TransitionEnv, Serializable):
    def __init__(self,
                 simulated=False,
                 collision_penalty=0.,
                 terminate_on_collision=True,
                 control_mode='task_space'):
        TransitionEnv.__init__(self, simulated=simulated,
                               collision_penalty=collision_penalty,
                               terminate_on_collision=terminate_on_collision,
                               control_mode=control_mode)

        self._active_task = TransitionPickAndPlaceTask()
        self.reward_type = []
        self.ob_shape = {"joint": [4], "box": [7], "goal": [3]}
        self.ob_type = self.ob_shape.keys()

        self._goal = self.sample_goal()

    def get_obs(self):
        # Robot obs
        robot_obs = self._robot.get_observation()

        # World obs
        world_obs = self._world.get_observation()

        # Construct obs specified by observation_space
        obs = []
        obs.append(robot_obs['gripper_position'])
        obs.append(robot_obs['gripper_state'])
        obs.append(world_obs['peg_position'])
        obs.append(world_obs['peg_orientation'])
        obs.append(self._goal)
        obs = np.concatenate(obs).ravel()

        return obs

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            return {
                'joint': ob[:, :4],
                'box': ob[:, 4:11],
                'goal': ob[:, 11:14]
            }
        return {
            'joint': ob[:4],
            'box': ob[4:11],
            'goal': ob[11:14]
        }
        

