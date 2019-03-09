"""ToyEnv task for the Sawyer robot."""

import sys
import gym
import moveit_commander
import numpy as np
import copy

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

#GOAL_POS = [0.702, -0.008, 0.180, 1.0, 0.0, 0.0, 0.0]

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
        world_obs = self._world.get_observation()
        print(world_obs)
        self._goal = np.concatenate((world_obs['box_lid_position'],
                                    np.array([1.0, 0.0, 0.0, 0.0])))
        self._goal[0] += 0.61036 - 0.65102
        self._goal[1] += -0.04772 + 0.05355
        self._goal[2] += 0.18557 - 0.1019
        print('New goal position: {}'.format(self._goal))

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

        if control_mode == 'effort':
            action[-1] = 0
        # Do the action
        print('step function with action ({}) and control mode ({})'.format(action, control_mode))
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

        if self._active_task.is_success(obs, info):
            r += self._active_task.completion_bonus
            done = True
            successful = True

        if in_collision:
            r -= self._collision_penalty
            if self._terminate_on_collision:
                done = True
                successful = False

        if self.is_fail(obs):
            done = True
            successful = False

        info['r'] = r
        info['d'] = done
        info['success'] = successful

        return obs, r, done, info

    def reset(self):
        self._step = 0
        self._robot.reset()
        self._active_task.reset()

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
        obs.append(self._goal[:3])
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

    def is_fail(self, obs):
        return self._active_task.is_fail(obs)

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

        if gripper_state != 0.0:
            print("****[has_peg] gripper_state: {}".format(gripper_state))
            return False

        peg_pos = self._world.get_peg_pose().position
        gripper_pos = self._robot.gripper_pose['position']
        max_xy_diff = 0.03
        max_z_diff = 0.04
        print("****[has_peg] gripper_state:{}, diff:{},{},{}".format(gripper_state, abs(peg_pos.x - gripper_pos.x), abs(peg_pos.y - gripper_pos.y), abs(peg_pos.z - gripper_pos.z)))
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
        self.ob_shape = {"joint": [4], "box": [7], "goal": [3]}
        self.ob_type = self.ob_shape.keys()
        self.control_mode = control_mode

    def act(self, ob):
        ob = self.get_obs()
        print("\tSawyerPick start")
        peg_position = copy.deepcopy(ob[4:11])
        print("\tpeg_position: "+str(peg_position))

        # gripper open
        self._robot._gripper_open()
        print("\tgripper open")

        # move to the peg position
        peg_position[2] += 0.07
        self._robot._move_to_target_position(peg_position)
        print("\tmove to the peg position")

        # go down
        peg_position[2] -= 0.07
        self._robot._move_to_target_position(peg_position)
        print("\tgo down")

        # gripper close
        self._robot._gripper_close()
        print("\tgripper close")

        # go up
        peg_position[2] = 0.25
        self._robot._move_to_target_position(peg_position)
        print("\tgo up")

        # move to the hole position
        hole_position = copy.deepcopy(self._goal)
        hole_position[2] += 0.07
        self._robot._move_to_target_position(hole_position)

        print("\thole_position: "+str(hole_position))
        print("\tmove to the hole position")
        print("\tSawyerPick end")

        return np.array([0.0, 0.0, 0.0, 0.0])


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
        self.control_mode = control_mode

    def act(self, ob):
        return np.array([0.0, 0.0, -0.02, 0.0])

    def reset(self, initial_obs):
        if self.has_peg():
            # move to the hole position
            hole_position = copy.deepcopy(self._goal)
            hole_position[2] += 0.07
            self._robot._move_to_target_position(hole_position)
            print("\tmove to the hole position")

            # move to the initial position
            init_position = copy.deepcopy(initial_obs[4:11])
            init_position[2] += 0.03
            self._robot._move_to_target_position(init_position)
            print("\tinit_position: ")

            # go down
            init_position[2] -= 0.035
            self._robot._move_to_target_position(init_position)
            print("\tgo down")

            # gripper open
            self._robot._gripper_open()
            print("\tgripper open")

        else:
            # ask peg is setted up
            ans = input('Has peg setted up? [y/n] :')
            if ans.lower() == 'y':
                # move to the hole position
                hole_position = copy.deepcopy(self._goal)
                hole_position[2] += 0.07
                self._robot._move_to_target_position(hole_position)

        return self.get_obs()


class TransitionPickAndPlaceEnv(TransitionEnv, Serializable):
    def __init__(self,
                 simulated=False,
                 collision_penalty=0.,
                 terminate_on_collision=True,
                 control_mode='effort'):
        TransitionEnv.__init__(self, simulated=simulated,
                               collision_penalty=collision_penalty,
                               terminate_on_collision=terminate_on_collision,
                               control_mode=control_mode)

        self._active_task = TransitionPickAndPlaceTask()
        self.reward_type = []
        self.ob_shape = {"joint": [4], "box": [7], "goal": [3]}
        self.ob_type = self.ob_shape.keys()
        self.control_mode = control_mode
        self.initial_obs = None

    def reset(self):
        print('Reset the environment')
        self._active_task.reset()

        if self.initial_obs is None:
            self.initial_obs = self.get_obs()

        if self.has_peg():
            # move to the hole position
            hole_position = copy.deepcopy(self._goal)
            hole_position[2] += 0.07
            self._robot._move_to_target_position(hole_position)
            print("\tmove to the hole position")

            # move to the initial position
            init_position = copy.deepcopy(self.initial_obs[4:11])
            init_position[2] += 0.03
            self._robot._move_to_target_position(init_position)
            print("\tinit_position: ")

            # go down
            init_position[2] -= 0.035
            self._robot._move_to_target_position(init_position)
            print("\tgo down")

            obs = self.get_obs()
            t = 0
            while obs[6] - self.initial_obs[6] > 0.02 and t < 5:
                print('adjust height of box', obs[6], self.initial_obs[6])
                init_position[2] -= 0.01
                self._robot._move_to_target_position(init_position)
                print("\tgo down")
                obs = self.get_obs()
                t += 1

            # gripper open
            self._robot._gripper_open()
            print("\tgripper open")

        init_position = copy.deepcopy(self.initial_obs[4:11])
        init_position[2] += 0.07
        self._robot._move_to_target_position(init_position)

        # ask peg is setted up
        print()
        print('*' * 80)
        ans = input('Set up the peg.')

        return self.get_obs()
