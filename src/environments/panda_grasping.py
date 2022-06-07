import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from collections import OrderedDict
import random
import numpy as np

"""Python Data Visualaize"""
import matplotlib.pyplot as plt
import cv2
import CNN as cnn
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import mujoco_py
from mujoco_py import MjRenderContextOffscreen, MjSim
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)

import utils.transform_utils as T
from utils.mjcf_utils import string_to_array
from utils.transform_utils import convert_quat
from environments.panda import PandaEnv

from models.arena import TableArena
from models.objects import CubeObject, Object1, Cyl2Object, Camera3rdview
from models.robot import Panda
from models.task import GraspingTask
import hjson
import os

class Pandagrasping(PandaEnv):

    def __init__(self, config):

        self.config = config
        self.table_full_size = config.table_full_size

        # Load the controller parameter configuration files
        controller_filepath = os.path.join(os.path.dirname(__file__), '..','config/controller_config.hjson')
        super().__init__(config, controller_config_file=controller_filepath)


    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([-0.5, 0, 0.913])

        # load model for table workspace
        self.mujoco_arena = TableArena(table_full_size=self.table_full_size)
        self.mujoco_arena.set_origin([0, 0, 0])

        # define mujoco objects
        Cube = CubeObject()
        self.mujoco_objects = OrderedDict([("Cube", Cube)])

        # task includes arena, robot, and objects of interest
        self.model = GraspingTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_objects)
        self.table_pos = string_to_array(self.model.table_body.get("pos"))


    def _get_reference(self):
        super()._get_reference()

        self.cube_body_id = self.sim.model.body_name2id("Cube") # 'cube': 23
        self.cube_geom_id = self.sim.model.geom_name2id("Cube") # 'cube': 41

        #information of objects
        self.object_names = list(self.mujoco_objects.keys())
        self.object_site_ids = [self.sim.model.site_name2id(ob_name) for ob_name in self.object_names]

        if self.has_gripper:
            self.finger_names = self.gripper.contact_geoms()
            self.l_finger_geom_ids = [self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms]
            self.r_finger_geom_ids = [self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms]

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        init_pos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi / 4])
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)

        """Cube"""
        self.sim.data.qpos[9:12] = np.array([0 ,0 ,0.75]) #0.75
        self.sim.data.qpos[12:16] = np.array([1 ,0 ,0, 0])

        self.phase = 0
        self.has_grasp = False

    def reward(self, action=None):

        # reaching reward
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cube_pos) # Euclidean distance b/w cube and grip site
        reaching_reward = 1 - np.tanh(10.0 * dist) # tanh function on distance

        vel = np.sum(abs(self.ee_v)) / 6  # robot last joint eef
        vel_reward = 1 - np.tanh(10.0 * vel)

        # Two phases of the task (0.Reaching and 1.Grasping)
        if self.phase == 0:

            reward = 0.8 * reaching_reward

            if dist < 0.08:
                reward += 0.3 * vel_reward

            # gripper open reward
            if action[-1] < 0:
                reward += 0.2 * abs(action[-1])

            if dist < 0.025:
                self.phase = 1

        elif self.phase == 1:

            reward = reaching_reward + vel_reward

            # gripper closing reward
            if action[-1] > 0:
                reward += 0.8 * action[-1]

            touch_left_finger = False
            touch_right_finger = False

            for i in range(self.sim.data.ncon):
                c = self.sim.data.contact[i]
                if c.geom1 in self.l_finger_geom_ids and c.geom2 == self.cube_geom_id:
                    touch_left_finger = True
                if c.geom1 == self.cube_geom_id and c.geom2 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom1 in self.r_finger_geom_ids and c.geom2 == self.cube_geom_id:
                    touch_right_finger = True
                if c.geom1 == self.cube_geom_id and c.geom2 in self.r_finger_geom_ids:
                    touch_right_finger = True

            self.has_grasp = touch_left_finger and touch_right_finger

            # grasping reward
            if self.has_grasp:
                reward += 0.5

        # success reward
        if self._check_success():
            reward += 30.0


        return reward

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cube_pos)

        return self.has_grasp and cube_pos[2] > 0.86


    def _get_observation(self):
        state = super()._get_observation()
        di = OrderedDict()

        # low-level object information
        # position and rotation of object
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        cube_quat = convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")
        di["cube_pos"] = cube_pos
        di["cube_quat"] = cube_quat

        gripper_site_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])
        # gripper to cube distance
        di["gripper_to_cube"] = gripper_site_pos - cube_pos

        #state["object-state"] = np.concatenate([cube_pos, cube_quat, di["gripper_to_cube"]]) #
        #print(state["object-state"])

        """ Random Object velocity, y-axis(가로)Translation + Rotation"""
        self.sim.data.qvel[9:12] = np.random.randn(3)*(0,0.5,0)
        self.sim.data.qvel[12:16] = np.random.randn(3)*0

        """Camera State RGB & Depth"""
        #if self.phase == 0:
        state["object-state"] = np.concatenate([cube_pos, cube_quat, di["gripper_to_cube"]])

        #elif self.phase == 1:
        #state["object-state"] = np.concatenate([cube_pos, cube_quat, di["gripper_to_cube"]])
            #rgb_feature_map = self.encoding_rgb()
            #depth_feature_map = self.encoding_depth()
            #state["camera_rgb-state"] = np.concatenate([rgb_feature_map])
            # #state["camera_depth-state"] = np.concatenate([depth_feature_map])

        return state


    """CNN Encoding"""

    def encoding_rgb(self):
        rgb = self.render_obs()
        # Form = ([1,84,84,3])
        #v = torch.tensor(rgb)
        #print(v.shape)

        rgb = rgb.squeeze(0)
        # Form = ([84,84,3])
        #v = torch.tensor(rgb)
        #print(v.shape)

        # Form = ([3,84,84])
        to_tensor = transforms.ToTensor()
        rgb = to_tensor(rgb)
        torch.permute(rgb, (1, 2, 0))
        rgb = rgb.unsqueeze(0)
        # Form = ([1,3,84,84])
        #v = torch.tensor(rgb)
        #print(v.shape)

        Class = cnn.CNN()
        rgb_data=Class.forward(rgb)

        rgb_data = rgb_data.squeeze(0)
        rgb_data=rgb_data.detach().numpy()

        #print(rgb_data)
        #print(rgb_data.shape)

        return np.asarray(rgb_data)

    #def encoding_depth(self):
        #depth = self.depth_obs()

        #Class = cnn.CNN()
        #depth_data = Class.forward(depth)
        #print(depth_data)

        #return np.asarray(depth_data)

    """Rendering"""

    def render_obs(self, mode=None, width=640, height=640):
        self._render_callback()
        rgb_data = []
        width1 = 640
        height1 = 640
        img1 = self.sim.render(width1, height1, camera_name='activecamera', depth=False)[::-1, :, :] # Rotation of Camera
        rgb = img1
        rgb = cv2.resize(rgb, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

        #print(rgb)
        print(rgb.shape)
        #plt.imshow(rgb)
        #plt.show()

        rgb_data.append(rgb)
        return np.asarray(rgb_data)

    def render(self, mode='human', width=640, height=640, depth=True):
        return super(Pandagrasping, self).render()

    def _render_callback(self):
        self.sim.forward()











