import math
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, LeggedRobotCfgCTS, LeggedRobotCfgMoENGCTS, LeggedRobotCfgMCPCTS, LeggedRobotCfgACMoECTS, LeggedRobotCfgDualMoECTS, LeggedRobotCfgMoECTS

class BoyingCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.40] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint':  0.1,   # [rad]
            'RL_hip_joint':  0.1,   # [rad]
            'FR_hip_joint': -0.1,   # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.0,  # [rad]
            'RR_thigh_joint': 1.0,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }
        turn_over = False
        turn_over_proportions = [0.0, 0.2, 0.8]
        turn_over_init_heights = {
            'backflip': [0.10, 0.15],
            'sideflip': [0.16, 0.21],
        }

    class env(LeggedRobotCfg.env):
        num_envs = 8192
        num_observations = 45
        # obs(45) + base_lin_vel(3) + foot_contact(4) + torques(12) + dof_acc(12) + height_measurements(187)
        num_privileged_obs = 45 + 3 + 4 + 12 + 12 + 187  # 263
        episode_length_s = 25

    class domain_rand(LeggedRobotCfg.domain_rand):
        ### Robot properties ###
        randomize_friction = True
        friction_range = [0.0, 2.0]

        randomize_base_mass = True
        added_mass_range = [-1., 1.]

        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]

        randomize_base_com = True
        added_base_com_range = [-0.03, 0.03]

        randomize_restitution = True
        restitution_range = [0.0, 0.5]

        ### Environment reset ###
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]

        ### Environment step ###
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6

        randomize_action_delay = True

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 60.}  # [N*m/rad]
        damping = {'joint': 4.5}    # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain(LeggedRobotCfg.terrain):
        max_init_terrain_level = 5
        terrain_proportions = [0.05, 0.20, 0.05, 0.25, 0.10, 0.20, 0.0, 0.0, 0.15]
        slope_threshold = 1.5
        move_down_by_accumulated_xy_command = True

    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        resampling_time = 5.
        heading_command = False
        zero_command_curriculum = {'start_iter': 0, 'end_iter': 1500, 'start_value': 0.0, 'end_value': 0.1}
        limit_ang_vel_at_zero_command_prob = 0.2
        limit_vel_prob = 0.2
        limit_vel_invert_when_continuous = True
        limit_vel = {"lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]}
        stop_heading_at_limit = True
        dynamic_resample_commands = True
        command_range_curriculum = [{
            'iter': 20000,
            'lin_vel_x': [-1.0, 1.0],
            'lin_vel_y': [-1.0, 1.0],
            'ang_vel_yaw': [-1.5, 1.5],
            'heading': [-1.57, 1.57],
        }, {
            'iter': 50000,
            'lin_vel_x': [-2.0, 2.0],
            'lin_vel_y': [-1.0, 1.0],
            'ang_vel_yaw': [-2.0, 2.0],
            'heading': [-1.57, 1.57],
        }]
        turn_over_zero_time = {
            "backflip": 5.0,
            "sideflip": 3.0,
        }
        terrain_max_command_ranges = [
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # wave
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # slope
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # rough slope
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs up
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs down
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # obstacles
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stepping stones
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # gap
            {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0], 'heading': [-1.57, 1.57]},  # flat
        ]

        class ranges:
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-1.57, 1.57]

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/boying_description/urdf/boying_description_withouthm.urdf'
        name = "boying"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.36  # boying init height 0.40m, target ~4cm lower
        only_positive_rewards = False
        max_contact_force = 187.  # boying without hm weight ~19.1kg -> ~187N
        curriculum_rewards = [
            {'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
            {'reward_name': 'correct_base_height', 'start_iter': 0, 'end_iter': 5000, 'start_value': 1.0, 'end_value': 10.0},
        ]
        tracking_sigma = 0.25
        dynamic_sigma = {
            "min_lin_vel": 0.5,
            "max_lin_vel": 1.5,
            "min_ang_vel": 1.0,
            "max_ang_vel": 2.0,
            "max_sigma": [5/12, 1/4, 1/4, 1/2, 1/2, 3/4, 1, 1, 1/4]
        }
        min_legs_distance = 0.1
        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            dof_power = -2e-5
            torques = -3e-5
            correct_base_height = -1.0
            action_rate = -0.01
            action_smoothness = -0.01
            collision = -1.0
            dof_pos_limits = -2.0
            feet_regulation = -0.05
            hip_to_default = -0.05

        turn_over_roll_threshold = math.pi / 4
        class turn_over_scales:
            upright = 1.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True


class BoyingCfgMoECTS(LeggedRobotCfgMoECTS):
    class policy(LeggedRobotCfgMoECTS.policy):
        expert_num = 8

    class runner(LeggedRobotCfgMoECTS.runner):
        run_name = ''
        experiment_name = 'boying_moe_cts'
        max_iterations = 150000
        save_interval = 500
