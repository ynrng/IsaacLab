# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the cartpole balancing task."""

"""Launch Isaac Sim Simulator first."""

"""usage:
    ./isaaclab.sh -p source/playground/run_franka_teleop_env.py  --num_envs 1 --teleop_device keyboard
"""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""



import torch
import carb

from omni.isaac.lab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse
from omni.isaac.lab.envs import ManagerBasedRLEnv

from franka_teleop_env_cfg import FrankaTeleopEnvCfg, FrankaTeleopPDIKEnvCfg


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    # resolve gripper command
    gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
    gripper_vel[:] = -1.0 if gripper_command else 1.0
    # compute actions
    return torch.concat([delta_pose, gripper_vel], dim=1)



def main():
    """Main function."""
    # create environment configuration

    env_cfg = FrankaTeleopPDIKEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.terminations.time_out = None
    # env_cfg.terminations.object_dropping = None
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.005 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity
        )
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper for keyboard
    print(teleop_interface)

    # reset environment
    env.reset()
    teleop_interface.reset()


    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            delta_pose = delta_pose.astype("float32")
            # convert to torch
            delta_pose = torch.tensor(delta_pose, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            # pre-process actions
            actions = pre_process_actions(delta_pose, gripper_command)
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
