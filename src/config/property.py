
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import str2bool, str2intlist


def add_argument(parser):

    # training scene
    parser.add_argument('--mode', type=int, default=1)

    # mujoco simulation
    parser.add_argument('--table_full_size', type=float, default=(1.2, 1.3, 0.02),
                        help='x, y, and z dimensions of the table')
    parser.add_argument('--gripper_type', type=str, default='PandaGripper',
                        help='Gripper type of robot')
    parser.add_argument('--gripper_visualization', type=str2bool, default=True,
                        help='using gripper visualization')
    # rendering
    parser.add_argument('--render_collision_mesh', type=str2bool, default=False,
                        help='if rendering collision meshes in camera')
    parser.add_argument('--render_visual_mesh', type=str2bool, default=True,
                        help='if rendering visual meshes in camera')

    """ My Image Creation Config Part """
    # Image  #My changes for Detection
    parser.add_argument('--cameras', default=0, type=int) # activecamera
    parser.add_argument('--observation_type',default='image', type=str)
    parser.add_argument('--image_size', default=84, type=int)

    # episode settings
    parser.add_argument('--horizon', type=int, default=600,
                        help='Every episode lasts for exactly @horizon timesteps')
    parser.add_argument('--ignore_done', type=str2bool, default=True,
                        help='if never terminating the environment (ignore @horizon)')

    # controller
    parser.add_argument('--control_freq', type=int, default= 250,
                        help='control signals to receive in every simulated second, sets the amount of simulation time that passes between every action input')

def get_default_config():
    """
    Gets default configurations for the lift environment.
    """
    import argparse
    parser = argparse.ArgumentParser("Default Configuration for lift Environment")
    add_argument(parser)
    config = parser.parse_args([])
    return config
