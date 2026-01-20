import os
import sys
import glob
import yaml
import select
import argparse
import numpy as np
import torch
import torch.jit
import mujoco, mujoco.viewer
from utils.model import *


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c

def mirror_obs(obs):
    """
    Mirror observations for left-right symmetry (numpy version).
    Supports both 1D (74,) and 2D (batch, 74) inputs.
    
    Joint order (indices 11-31 for pos, 32-52 for vel, 53-73 for actions):
    0: Left_Shoulder_Pitch    8: Waist              16: Right_Hip_Roll
    1: Left_Shoulder_Roll     9: Left_Hip_Pitch     17: Right_Hip_Yaw
    2: Left_Elbow_Pitch      10: Left_Hip_Roll      18: Right_Knee_Pitch
    3: Left_Elbow_Yaw        11: Left_Hip_Yaw       19: Right_Ankle_Pitch
    4: Right_Shoulder_Pitch  12: Left_Knee_Pitch    20: Right_Ankle_Roll
    5: Right_Shoulder_Roll   13: Left_Ankle_Pitch
    6: Right_Elbow_Pitch     14: Left_Ankle_Roll
    7: Right_Elbow_Yaw       15: Right_Hip_Pitch
    """
    mirrored = obs.copy()

    # Gravity (0:3) - mirror y component
    mirrored[..., 1] *= -1  # gy

    # Angular velocity (3:6) - mirror x and z components
    mirrored[..., 3] *= -1  # wx (roll rate)
    mirrored[..., 5] *= -1  # wz (yaw rate)

    # Commands (6:9) - mirror y velocity and yaw
    mirrored[..., 7] *= -1  # vy
    mirrored[..., 8] *= -1  # vyaw

    # Gait phase (9:11) - negate for phase shift by PI (swap stance/swing leg)
    mirrored[..., 9] *= -1   # cos_phase
    mirrored[..., 10] *= -1  # sin_phase

    # DOF positions (11:32), DOF velocities (32:53), Actions (53:74)
    # Each block has 21 DOFs
    for start_idx in [11, 32, 53]:
        # Swap left arm (0-3) with right arm (4-7)
        left_arm = mirrored[..., start_idx+0:start_idx+4].copy()
        right_arm = mirrored[..., start_idx+4:start_idx+8].copy()
        mirrored[..., start_idx+0:start_idx+4] = right_arm
        mirrored[..., start_idx+4:start_idx+8] = left_arm

        # Negate roll and yaw for arms (after swap)
        # Index 1: Shoulder_Roll, Index 3: Elbow_Yaw
        mirrored[..., start_idx+1] *= -1  # Left Shoulder Roll (was Right)
        mirrored[..., start_idx+3] *= -1  # Left Elbow Yaw (was Right)
        mirrored[..., start_idx+5] *= -1  # Right Shoulder Roll (was Left)
        mirrored[..., start_idx+7] *= -1  # Right Elbow Yaw (was Left)

        # Waist (index 8) - negate (yaw motion)
        mirrored[..., start_idx+8] *= -1

        # Swap left leg (9-14) with right leg (15-20)
        left_leg = mirrored[..., start_idx+9:start_idx+15].copy()
        right_leg = mirrored[..., start_idx+15:start_idx+21].copy()
        mirrored[..., start_idx+9:start_idx+15] = right_leg
        mirrored[..., start_idx+15:start_idx+21] = left_leg

        # Negate roll and yaw for legs (after swap)
        # Leg order: Hip_Pitch(0), Hip_Roll(1), Hip_Yaw(2), Knee_Pitch(3), Ankle_Pitch(4), Ankle_Roll(5)
        mirrored[..., start_idx+10] *= -1  # Left Hip Roll (was Right)
        mirrored[..., start_idx+11] *= -1  # Left Hip Yaw (was Right)
        mirrored[..., start_idx+14] *= -1  # Left Ankle Roll (was Right)
        mirrored[..., start_idx+16] *= -1  # Right Hip Roll (was Left)
        mirrored[..., start_idx+17] *= -1  # Right Hip Yaw (was Left)
        mirrored[..., start_idx+20] *= -1  # Right Ankle Roll (was Left)

    return mirrored


def mirror_act(actions):
    """
    Mirror actions (numpy version).
    Supports both 1D (21,) and 2D (batch, 21) inputs.
    
    Action order (21 DOFs):
    0-3:   Left arm  (Shoulder_P, Shoulder_R, Elbow_P, Elbow_Y)
    4-7:   Right arm (Shoulder_P, Shoulder_R, Elbow_P, Elbow_Y)
    8:     Waist
    9-14:  Left leg  (Hip_P, Hip_R, Hip_Y, Knee_P, Ankle_P, Ankle_R)
    15-20: Right leg (Hip_P, Hip_R, Hip_Y, Knee_P, Ankle_P, Ankle_R)
    """
    mirrored = actions.copy()

    # Swap left arm (0-3) with right arm (4-7)
    left_arm = mirrored[..., 0:4].copy()
    right_arm = mirrored[..., 4:8].copy()
    mirrored[..., 0:4] = right_arm
    mirrored[..., 4:8] = left_arm

    # Negate roll and yaw for arms (after swap)
    mirrored[..., 1] *= -1  # Left Shoulder Roll (was Right)
    mirrored[..., 3] *= -1  # Left Elbow Yaw (was Right)
    mirrored[..., 5] *= -1  # Right Shoulder Roll (was Left)
    mirrored[..., 7] *= -1  # Right Elbow Yaw (was Left)

    # Waist (index 8) - negate
    mirrored[..., 8] *= -1

    # Swap left leg (9-14) with right leg (15-20)
    left_leg = mirrored[..., 9:15].copy()
    right_leg = mirrored[..., 15:21].copy()
    mirrored[..., 9:15] = right_leg
    mirrored[..., 15:21] = left_leg

    # Negate roll and yaw for legs (after swap)
    mirrored[..., 10] *= -1  # Left Hip Roll (was Right)
    mirrored[..., 11] *= -1  # Left Hip Yaw (was Right)
    mirrored[..., 14] *= -1  # Left Ankle Roll (was Right)
    mirrored[..., 16] *= -1  # Right Hip Roll (was Left)
    mirrored[..., 17] *= -1  # Right Hip Yaw (was Left)
    mirrored[..., 20] *= -1  # Right Ankle Roll (was Left)

    return mirrored


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    parser.add_argument("--checkpoint", type=str, help="Path of model checkpoint to load. Overrides config file if provided.")
    args = parser.parse_args()
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    if args.checkpoint is not None:
        cfg["basic"]["checkpoint"] = args.checkpoint

    model = RMA(cfg["env"]["num_actions"], cfg["env"]["num_observations"], cfg["runner"]["num_stack"], cfg["env"]["num_privileged_obs"], cfg["algorithm"]["num_embedding"])
    if not cfg["basic"]["checkpoint"] or (cfg["basic"]["checkpoint"] == "-1") or (cfg["basic"]["checkpoint"] == -1):
        cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
    print("Loading model from {}".format(cfg["basic"]["checkpoint"]))
    checkpoint = torch.load(cfg["basic"]["checkpoint"], map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval() # Set the model to evaluation mode a

    mj_model = mujoco.MjModel.from_xml_path(cfg["asset"]["mujoco_file"])
    mj_model.opt.timestep = cfg["sim"]["dt"]
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    default_dof_pos = np.zeros(mj_model.nu, dtype=np.float32)
    dof_stiffness = np.zeros(mj_model.nu, dtype=np.float32)
    dof_damping = np.zeros(mj_model.nu, dtype=np.float32)
    base_mass_scaled = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) # no scaling
    for i in range(mj_model.nu):
        found = False
        for name in cfg["init_state"]["default_joint_angles"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                default_dof_pos[i] = cfg["init_state"]["default_joint_angles"][name]
                found = True
        if not found:
            default_dof_pos[i] = cfg["init_state"]["default_joint_angles"]["default"]

        found = False
        for name in cfg["control"]["stiffness"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                dof_stiffness[i] = cfg["control"]["stiffness"][name]
                dof_damping[i] = cfg["control"]["damping"][name]
                found = True
        if not found:
            raise ValueError(f"PD gain of joint {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)} were not defined")
    mj_data.qpos = np.concatenate(
        [
            np.array(cfg["init_state"]["pos"], dtype=np.float32),
            np.array(cfg["init_state"]["rot"][3:4] + cfg["init_state"]["rot"][0:3], dtype=np.float32),
            default_dof_pos,
        ]
    )
    mujoco.mj_forward(mj_model, mj_data)

    actions = np.zeros((cfg["env"]["num_actions"]), dtype=np.float32)
    dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)
    stacked_obs = np.zeros((1,cfg["runner"]["num_stack"], cfg["env"]["num_observations"]), dtype=np.float32)
    gait_frequency = gait_process = 0.0
    lin_vel_x = lin_vel_y = ang_vel_yaw = 0.0
    it = 0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.elevation = -20
        print(f"Set command (x, y, yaw): ")
        while viewer.is_running():
            if select.select([sys.stdin], [], [], 0)[0]:
                try:
                    parts = sys.stdin.readline().strip().split()
                    if len(parts) == 3:
                        lin_vel_x, lin_vel_y, ang_vel_yaw = map(float, parts)
                        if lin_vel_x == 0 and lin_vel_y == 0 and ang_vel_yaw == 0:
                            gait_frequency = 0
                        else:
                            gait_frequency = np.average(cfg["commands"]["gait_frequency"])
                        print(
                            f"Updated command to: x={lin_vel_x}, y={lin_vel_y}, yaw={ang_vel_yaw}\nSet command (x, y, yaw): ",
                            end="",
                        )
                    else:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Enter three numeric values.\nSet command (x, y, yaw): ", end="")
            dof_pos = mj_data.qpos.astype(np.float32)[7:]
            dof_vel = mj_data.qvel.astype(np.float32)[6:]
            quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
            base_ang_vel = mj_data.sensor("angular-velocity").data.astype(np.float32)
            base_linear_vel = mj_data.sensor("linear-velocity").data.astype(np.float32)
            projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
            if it % cfg["control"]["decimation"] == 0:
                obs = np.zeros(cfg["env"]["num_observations"], dtype=np.float32)
                obs[0:3] = projected_gravity * cfg["normalization"]["gravity"]
                obs[3:6] = base_ang_vel * cfg["normalization"]["ang_vel"]
                obs[6] = lin_vel_x * cfg["normalization"]["lin_vel"]
                obs[7] = lin_vel_y * cfg["normalization"]["lin_vel"]
                obs[8] = ang_vel_yaw * cfg["normalization"]["ang_vel"]
                obs[9] = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[10] = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[11:32] = (dof_pos - default_dof_pos) * cfg["normalization"]["dof_pos"]
                obs[32:53] = dof_vel * cfg["normalization"]["dof_vel"]
                obs[53:74] = actions

                if it == 0:
                    stacked_obs[:] = obs[np.newaxis, np.newaxis, :]
                stacked_obs[:, 1:, :] = stacked_obs[:, :-1, :]
                stacked_obs[:, 0, :] = obs

                mirrored_obs = mirror_obs(obs)
                mirrored_stacked_obs = mirror_obs(stacked_obs)

                dist, _ = model.act(torch.tensor(obs).unsqueeze(0), stacked_obs=torch.tensor(stacked_obs))
                mirrored_dist, _ = model.act(torch.tensor(mirrored_obs).unsqueeze(0), stacked_obs=torch.tensor(mirrored_stacked_obs))
                actions[:] = 0.5 * (dist.loc.detach().numpy() + mirror_act(mirrored_dist.loc.detach().numpy()))
                if hasattr(dist, "loc"):
                    actions[:] = dist.loc.detach().numpy()
                else:
                    actions[:] = dist.detach().numpy()

                actions[:] = np.clip(actions, -cfg["normalization"]["clip_actions"], cfg["normalization"]["clip_actions"])
                dof_targets[:] = default_dof_pos + cfg["control"]["action_scale"] * actions
            mj_data.ctrl = np.clip(
                dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel,
                mj_model.actuator_ctrlrange[:, 0],
                mj_model.actuator_ctrlrange[:, 1],
            )
            mujoco.mj_step(mj_model, mj_data)
            viewer.cam.lookat[:] = mj_data.qpos.astype(np.float32)[0:3]
            viewer.sync()
            it += 1
            gait_process = np.fmod(gait_process + cfg["sim"]["dt"] * gait_frequency, 1.0)



