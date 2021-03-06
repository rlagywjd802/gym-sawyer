import numpy as np

from sawyer.mujoco.tasks.base import ComposableTask


class PickTask(ComposableTask):
    """
    Task to pick up an object with the robot gripper.

    Reward function is based on the following heuristics:
    - Positive reward for moving gripper closer to object
    - Positive reward for grasping object
    - Positive reward for lifting the object

    Success condition:
    - Object is grasped and has been lifted above the table
    """
    def __init__(self,
                 pick_object,
                 never_done=False,
                 object_lift_target=0.3,
                 completion_bonus=0,
                 c_dist=1,
                 c_grasp=1,
                 c_lift=10):
        self._pick_object = pick_object
        self._never_done = never_done
        self._obj_lift_target = object_lift_target
        self._completion_bonus = completion_bonus
        self._c_dist = c_dist
        self._c_grasp = c_grasp
        self._c_lift = c_lift

        self._last_d_grip2obj = None
        self._last_dz_obj = None

    def compute_reward(self, obs, info):
        # Gather info variables
        gripper_pos = info['gripper_position']
        obj_pos = info['world_obs']['{}_position'.format(self._pick_object)]
        grasped = info['grasped_{}'.format(self._pick_object)]

        # Calculate reward measures
        d_grip2obj = np.linalg.norm(gripper_pos - obj_pos, axis=-1)
        if self._last_d_grip2obj is None:
            self._last_d_grip2obj = d_grip2obj
        dz_obj = obj_pos[2]
        if self._last_dz_obj is None:
            self._last_dz_obj = dz_obj

        # Calculate reward components
        r_dist = int(not grasped) * 100 * (self._last_d_grip2obj - d_grip2obj)
        r_grasp = 0
        r_lift = int(grasped) * 100 * (dz_obj - self._last_dz_obj)

        # Update stateful variables
        self._last_d_grip2obj = d_grip2obj
        self._last_dz_obj = dz_obj

        r = self._c_dist * r_dist + self._c_grasp * r_grasp + self._c_lift * r_lift
        return r

    def is_success(self, obs, info):
        if self._never_done:
            return False
        gripper_pos = info['gripper_position']
        obj_pos = info['world_obs']['{}_position'.format(self._pick_object)]
        grasped = info['grasped_{}'.format(self._pick_object)]
        
        # edited for transition policy project
        # return grasped and obj_pos[2] > self._obj_lift_target
        d_grip2obj = np.linalg.norm(gripper_pos[0:2] - obj_pos[0:2], axis=-1)
        threshold = 0.03
        print('d_grip2obj:'+str(d_grip2obj))
        print('gripper_pos:'+str(gripper_pos[0:2]))
        print('obj_pos:'+str(obj_pos[0:2]))
        if d_grip2obj < threshold:
            return True
        else:
            return False


    def done(self, obs, info):
        obj_pos = info['world_obs']['{}_position'.format(self._pick_object)]

        if obj_pos[2] < 0.:
            return True
        return False

    @property
    def completion_bonus(self):
        return self._completion_bonus


class PlaceTask(ComposableTask):
    """
    Task to place object at a desired location.

    Reward function is based on the following heuristics:
    - Positive reward for moving gripper closer to goal
    - Negative reward for releasing object before reaching the goal position
    - Positive reward for releasing object near goal position
    """
    def __init__(self,
                 place_object,
                 location,
                 success_thresh=0.01,
                 completion_bonus=0,
                 c_dist=1,
                 c_early_release=100,
                 c_release=10):
        self._place_object = place_object
        self._location = location
        self._success_thresh = success_thresh
        self._completion_bonus = completion_bonus
        self._c_dist = c_dist
        self._c_early_release = c_early_release
        self._c_release = c_release

    def compute_reward(self, obs, info):
        obj_pos = info['world_obs']['{}_position'.format(self._place_object)]
        released = not info['grasped_{}'.format(self._place_object)]
        d = np.linalg.norm(obj_pos - self._location, axis=-1)

        if d < self._success_thresh:
            return -self._c_dist * d + self._c_release * released
        else:
            return -self._c_dist * d - self._c_early_release * released

    def is_success(self, obs, info):
        obj_pos = info['world_obs']['{}_position'.format(self._place_object)]
        released = not info['grasped_{}'.format(self._place_object)]
        d = np.linalg.norm(obj_pos - self._location, axis=-1)

        return released and d < self._success_thresh

    @property
    def completion_bonus(self):
        return self._completion_bonus
