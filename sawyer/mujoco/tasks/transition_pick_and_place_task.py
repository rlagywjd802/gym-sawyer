import numpy as np

from sawyer.mujoco.tasks.base import ComposableTask


class TransitionTask(ComposableTask):
    """
    Task to pick up an object with the robot gripper.

    Success condition:
    - Object is grasped and has been lifted above the table
    """
    def __init__(self):
        pass

    def compute_reward(self, obs, info):
        return 0

    def is_success(self, obs, info=None, init=None):
        raise NotImplementedError

    def is_terminate(self, obs, init):
        return self.is_success(obs, init=init)

    def is_fail(self, obs):
        raise NotImplementedError

    def reset(self):
        pass

    @property
    def completion_bonus(self):
        return self._completion_bonus


class TransitionPickTask(TransitionTask):
    """
    Task to pick up an object with the robot gripper.

    Success condition:
    - Object is grasped and has been lifted above the table
    """
    def __init__(self,
                 success_thresh=0.05,
                 object_lift_target=0.3,
                 completion_bonus=0):
        self._success_thresh = success_thresh
        self._obj_lift_target = object_lift_target
        self._completion_bonus = completion_bonus
        self._t = 0

    def is_success(self, obs, info=None, init=None):
        return True
        if init:
            self.reset()
        goal = obs[11:14] + np.array([0, 0, 0.04])
        box_pos = obs[4:7]
        d = np.linalg.norm(box_pos - goal, axis=-1)
        print("****[pick/is success] box_pos:{}, goal:{}, d:{}".format(box_pos, goal, d))
        return d < self._success_thresh

    def is_fail(self, obs):
        self._t += 1
        if self._t >= 1 and not self.is_success(obs):
            return True
        return False

    def reset(self):
        self._t = 0


class TransitionPlaceTask(TransitionTask):
    """
    Task to place object at a desired location.
    """
    def __init__(self,
                 success_thresh=0.015,
                 completion_bonus=0):
        self._success_thresh = success_thresh
        self._completion_bonus = completion_bonus
        self._prev_box_pos = None

    def is_success(self, obs, info=None, init=None):
        if init:
            self.reset()
        box_pos = obs[4:7]
        goal = obs[11:14]

        max_xy_diff = 0.03
        abs_diff = abs(box_pos - goal)

        print("****[place/is success] abs_diff:{}".format(abs_diff))
        return ( abs_diff[0] < max_xy_diff and
            abs_diff[1] < max_xy_diff and
            box_pos[2] < 0.21 )

    def is_fail(self, obs):
        box_pos = obs[4:7]
        goal = obs[11:14]
        max_xy_diff = 0.03
        abs_diff = abs(box_pos - goal)

        if self._prev_box_pos is None:
            self._prev_box_pos = box_pos
        else:
            max_z_diff = 0.009
            z_diff = self._prev_box_pos[2] - box_pos[2]
            print("****[place/is_fail] z_diff:{}, box_pos_z:{}".format(z_diff, box_pos[2]))
            print(self._prev_box_pos[2], box_pos[2])
            if abs_diff[0] > max_xy_diff or abs_diff[1] > max_xy_diff or z_diff < max_z_diff:
                return True
            else:
                self._prev_box_pos = box_pos
        return False

    def reset(self):
        self._prev_box_pos = None


class TransitionPickAndPlaceTask(TransitionTask):
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
        self._prev_box_pos = None
        self._picked = False
        self._placing = False

    def is_success(self, obs, info=None, init=None):
        if init:
            self.reset()
        box_pos = obs[4:7]
        goal = obs[11:14]

        max_xy_diff = 0.02
        abs_diff = abs(box_pos - goal)

        print("****[pick&place/is success] abs_diff:{}, box_z:{}".format(abs_diff, box_pos[2]))
        return ( abs_diff[0] < max_xy_diff and
            abs_diff[1] < max_xy_diff and
            box_pos[2] < 0.22 )

    def is_fail(self, obs):
        box_pos = obs[4:7]
        goal = obs[11:14]
        abs_diff = abs(box_pos - goal)
        max_xy_diff = 0.03

        if self._picked:
            self._placing = True
            print("placing True")
        else:
            print("placing False")

        if self._picked and not self._placing:
            print("return True")
            return True

        self._picked = True

        if self._placing:
            if self._prev_box_pos is None:
                self._prev_box_pos = box_pos
            else:
                max_z_diff = 0.009
                z_diff = self._prev_box_pos[2] - box_pos[2]
                print("****[pick&place/is_fail] z_diff:{}, box_pos_z:{}".format(z_diff, box_pos[2]))
                print(self._prev_box_pos[2], box_pos[2])
                if box_pos[2] < 0.24 and (abs_diff[0] > max_xy_diff or abs_diff[1] > max_xy_diff or z_diff < max_z_diff):
                    print("return True")
                    return True
                else:
                    self._prev_box_pos = box_pos
        return False

    def get_next_primitive(self, obs, prev_primitive):
        if prev_primitive == -1:
            return 'pick'
        return 'place'

    def reset(self):
        self._picked = False
        self._placing = False
        self._prev_box_pos = None

