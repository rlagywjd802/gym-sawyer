import numpy as np

from sawyer.mujoco.tasks.base import ComposableTask


class TransitionPickTask(ComposableTask):
    """
    Task to pick up an object with the robot gripper.

    Success condition:
    - Object is grasped and has been lifted above the table
    """
    def __init__(self,
                 object_lift_target=0.3,
                 completion_bonus=0):
        self._obj_lift_target = object_lift_target
        self._completion_bonus = completion_bonus

    def compute_reward(self, obs, info):
        return 0

    def is_success(self, obs, info):
        box_pos = obs[4:7]
        box_z = box_pos[2]
        return box_z >= self._obj_lift_target

    def is_terminate(self, obs, init):
        box_pos = obs[4:7]
        box_z = box_pos[2]
        return box_z >= self._obj_lift_target

    @property
    def completion_bonus(self):
        return self._completion_bonus


class TransitionPlaceTask(ComposableTask):
    """
    Task to place object at a desired location.
    """
    def __init__(self,
                 success_thresh=0.045,
                 completion_bonus=0):
        self._success_thresh = success_thresh
        self._completion_bonus = completion_bonus

    def compute_reward(self, obs, info):
        return 0

    def is_success(self, obs, info):
        box_pos = obs[4:7]
        released = obs[3]
        goal = obs[11:14]
        d = np.linalg.norm(box_pos - goal, axis=-1)
        print("*****[is success] released:"+str(released)+", box_pos:"+str(box_pos)+", goal:"+str(goal)+", d:"+str(d))
        #return d < self._success_thresh

        max_xy_diff = 0.015
        max_z_diff = 0.025
        print("*****[is success] released:"+str(released)+", box_pos:"+str(box_pos)+", goal:"+str(goal))
        return ( abs(box_pos[0] - goal[0]) < max_xy_diff and
            abs(box_pos[1] - goal[1]) < max_xy_diff and
            abs(box_pos[2] - goal[2]) < max_z_diff )

    def is_terminate(self, obs, init):
        box_pos = obs[4:7]
        released = obs[3]
        goal = obs[11:14]
        d = np.linalg.norm(box_pos - goal, axis=-1)
        return released and d < self._success_thresh

    @property
    def completion_bonus(self):
        return self._completion_bonus


class TransitionPickAndPlaceTask(ComposableTask):
    """
    Task to pick up an object and place the object at a desired location.

    Success condition:
    - Object is grasped and has been lifted above the table
    """
    def __init__(self,
                 success_thresh=0.01,
                 completion_bonus=0):
        self._success_thresh = success_thresh
        self._completion_bonus = completion_bonus
        print("init")

    def compute_reward(self, obs, info):
        return 0

    def is_success(self, obs, info):
        box_pos = obs[4:7]
        released = obs[3]
        goal = obs[11:14]
        d = np.linalg.norm(box_pos - goal, axis=-1)
        return released and d < self._success_thresh

    def is_terminate(self, obs, init):
        box_pos = obs[4:7]
        released = obs[3]
        goal = obs[11:14]
        d = np.linalg.norm(box_pos - goal, axis=-1)
        return released and d < self._success_thresh

    def get_next_primitive(self, obs, prev_primitive):
        if prev_primitive == -1:
            return 'pick'
        return 'place'

    @property
    def completion_bonus(self):
        return self._completion_bonus
