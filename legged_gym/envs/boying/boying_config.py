# boying_config.py
# Boying 四足机器人的 IsaacGym 训练配置文件，用于 boying_moe_cts 任务。
# 继承自 LeggedRobotCfg（环境配置）和 LeggedRobotCfgMoECTS（算法配置）。
# 相比 Go2 的主要差异：质量 19.1kg（Go2 15kg）、Kp=60（Go2 Kp=20）、使用 withouthm URDF。

import math  # 用于 math.pi，计算翻转检测的 roll 角度阈值

# 导入所有可用的基类配置，BoyingCfg 继承 LeggedRobotCfg，BoyingCfgMoECTS 继承 LeggedRobotCfgMoECTS
from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,          # 通用腿足机器人环境配置基类
    LeggedRobotCfgPPO,       # PPO 算法配置（未使用，保留导入）
    LeggedRobotCfgCTS,       # CTS（并发师生）算法配置（未使用，保留导入）
    LeggedRobotCfgMoENGCTS,  # MoE-NoGoal-CTS 算法配置（未使用，保留导入）
    LeggedRobotCfgMCPCTS,    # MCP-CTS 算法配置（未使用，保留导入）
    LeggedRobotCfgACMoECTS,  # AC-MoE-CTS 算法配置（未使用，保留导入）
    LeggedRobotCfgDualMoECTS,# DualMoE-CTS 算法配置（未使用，保留导入）
    LeggedRobotCfgMoECTS,    # MoE-CTS 算法配置，boying_moe_cts 任务实际使用此基类
)


class BoyingCfg(LeggedRobotCfg):
    """
    Boying 机器人环境配置。
    覆写 LeggedRobotCfg 中与机器人本体相关的参数，其余继承基类默认值。
    """

    class init_state(LeggedRobotCfg.init_state):
        """机器人初始化状态配置。"""

        # 机器人质心初始位置 [x, y, z]，单位 m
        # Boying withouthm 实际站立高度约 0.336m（正向运动学估算），
        # 设 0.40m 提供约 6.4cm 缓冲，防止刚初始化时足端穿地
        pos = [0.0, 0.0, 0.40]

        # 动作为 0 时的目标关节角度（默认姿态），单位 rad
        # 髋关节：左腿 +0.1，右腿 -0.1（略微外展，对称分布）
        # 大腿：前腿 0.8 rad，后腿 1.0 rad（后腿略更弯曲，改善重心分布）
        # 小腿：全部 -1.5 rad（向后弯曲，形成稳定支撑三角）
        default_joint_angles = {
            'FL_hip_joint':  0.1,   # 左前髋，正值=内收方向，单位 rad
            'RL_hip_joint':  0.1,   # 左后髋，与左前髋对称
            'FR_hip_joint': -0.1,   # 右前髋，负值=内收方向（坐标系与左侧相反）
            'RR_hip_joint': -0.1,   # 右后髋，与右前髋对称

            'FL_thigh_joint': 0.8,  # 左前大腿，单位 rad
            'FR_thigh_joint': 0.8,  # 右前大腿
            'RL_thigh_joint': 1.0,  # 左后大腿，后腿比前腿多弯 0.2 rad
            'RR_thigh_joint': 1.0,  # 右后大腿

            'FL_calf_joint': -1.5,  # 左前小腿，负值=向后弯曲，单位 rad
            'FR_calf_joint': -1.5,  # 右前小腿
            'RL_calf_joint': -1.5,  # 左后小腿
            'RR_calf_joint': -1.5,  # 右后小腿
        }

        turn_over = False  # 不在训练中初始化翻倒姿态（Boying 暂不训练翻身恢复）

        # 翻倒初始化比例：[后空翻比例, 侧翻比例, 正常比例]
        # turn_over=False 时此参数不生效，保留供后续开启翻身训练使用
        turn_over_proportions = [0.0, 0.2, 0.8]

        # 各翻倒类型的初始高度范围 [min, max]，单位 m
        turn_over_init_heights = {
            'backflip': [0.10, 0.15],  # 后空翻初始高度范围
            'sideflip': [0.16, 0.21],  # 侧翻初始高度范围
        }

    class env(LeggedRobotCfg.env):
        """环境并行配置，控制仿真规模和观测空间维度。"""

        num_envs = 8192  # 并行仿真环境数，RTX 4090 Laptop (16GB) 推荐使用 4096

        # 学生策略观测维度（45维）：
        # 角速度(3) + 投影重力(3) + 指令(3) + 关节位置偏差(12) + 关节速度(12) + 上一步动作(12)
        num_observations = 45

        # 教师（特权）观测维度（263维）：
        # obs(45) + 真实线速度(3) + 足端接触力(4) + 关节力矩(12) + 关节加速度(12) + 地形高度图(187)
        num_privileged_obs = 45 + 3 + 4 + 12 + 12 + 187  # = 263

        episode_length_s = 25  # 单 episode 最大时长，单位秒（25s × 50Hz = 1250 步）

    class domain_rand(LeggedRobotCfg.domain_rand):
        """
        域随机化配置，用于提升 sim-to-real 迁移能力。
        通过随机化物理参数，训练出对真实世界不确定性鲁棒的策略。
        """

        ### 机器人本体属性随机化（每次 reset 时重新采样） ###

        randomize_friction = True           # 启用摩擦系数随机化
        friction_range = [0.0, 2.0]         # 摩擦系数范围，覆盖光滑地面到粗糙地面

        randomize_base_mass = True          # 启用机身附加质量随机化（模拟携带负载）
        added_mass_range = [-1., 1.]        # 附加质量范围，单位 kg（Boying 基础质量约 19.1kg）

        randomize_link_mass = True          # 启用连杆质量倍率随机化（模拟零件制造误差）
        multiplied_link_mass_range = [0.9, 1.1]  # 质量倍率范围（±10%）

        randomize_base_com = True           # 启用质心位置随机化（模拟载重偏心）
        added_base_com_range = [-0.03, 0.03]  # 质心偏移范围，单位 m（±3cm）

        randomize_restitution = True        # 启用恢复系数（弹性）随机化
        restitution_range = [0.0, 0.5]      # 恢复系数范围（0=完全非弹性，0.5=半弹性）

        ### 环境重置时的随机化 ###

        randomize_pd_gains = True                    # 启用 PD 增益随机化，模拟电机参数误差
        stiffness_multiplier_range = [0.9, 1.1]      # Kp 倍率范围（Boying Kp=60，±10% = 54~66）
        damping_multiplier_range = [0.9, 1.1]        # Kd 倍率范围（Boying Kd=4.5，±10% = 4.05~4.95）

        randomize_motor_zero_offset = True           # 启用电机零位偏移随机化（模拟零位标定误差）
        motor_zero_offset_range = [-0.035, 0.035]    # 零位偏移范围，单位 rad（约 ±2°）

        randomize_motor_strength = True              # 启用电机力矩强度随机化（模拟电机老化/温度影响）
        motor_strength_range = [0.8, 1.2]            # 力矩强度倍率范围（±20%）

        ### 仿真步内的随机化 ###

        push_robots = True          # 启用随机推力扰动，提升抗干扰能力
        push_interval_s = 4         # 推力间隔时间，单位秒（每 4 秒施加一次随机推力）
        max_push_vel_xy = 0.4       # 水平推力引起的最大速度变化，单位 m/s
        max_push_ang_vel = 0.6      # 旋转推力引起的最大角速度变化，单位 rad/s

        randomize_action_delay = True  # 启用动作延迟随机化（模拟通信延迟 0~20ms，4 decimation）

    class control(LeggedRobotCfg.control):
        """
        关节控制器配置（PD 位置控制）。
        Boying 使用比 Go2 刚度高 3 倍的参数（Kp=60 vs Go2 的 Kp=20），
        这直接导致相同动作下输出力矩约 3 倍，奖励函数中 torques/dof_acc/dof_power 系数需相应减小。
        """

        control_type = 'P'          # 控制模式：'P' = PD 位置控制（策略输出目标关节角度）
        stiffness = {'joint': 60.}  # 关节刚度 Kp，单位 N*m/rad。Boying=60，Go2=20，差 3 倍
        damping = {'joint': 4.5}    # 关节阻尼 Kd，单位 N*m*s/rad。Boying=4.5，Go2=0.5

        # 动作缩放：目标角度 = action_scale × action + default_angle
        # 0.25 rad ≈ 14.3°，限制单步最大关节位移，防止过激动作
        action_scale = 0.25

        # 控制频率细分：每个策略步执行 4 次物理仿真步
        # 仿真 dt=0.005s，策略 dt=0.005×4=0.02s（50Hz）
        decimation = 4

    class terrain(LeggedRobotCfg.terrain):
        """
        地形课程配置，控制训练地形类型和难度递进。
        """

        max_init_terrain_level = 5  # 训练初始阶段的最高地形难度等级（0~9，5为中等）

        # 各地形类型的采样比例，顺序：[wave, slope, rough_slope, stairs_up, stairs_down, obstacles, stepping_stones, gap, flat]
        # 当前配置偏向楼梯(0.25)和斜坡(0.20)，适合训练 Boying 的地形适应能力
        terrain_proportions = [0.05, 0.20, 0.05, 0.25, 0.10, 0.20, 0.0, 0.0, 0.15]

        # slope_threshold 控制地形生成时斜坡的最大坡度，值越大允许越陡的坡面
        slope_threshold = 1.5  # 较高值允许生成较陡的波浪地形和粗糙斜坡

        # True = 地形课程下降触发条件基于机器人累计水平位移（而非绝对位置）
        # 更公平地评估运动能力，避免原地踏步也能触发课程升级
        move_down_by_accumulated_xy_command = True

    class commands(LeggedRobotCfg.commands):
        """
        速度指令配置，控制训练中指令的采样方式和课程递进。
        """

        num_commands = 4        # 指令维度：lin_vel_x, lin_vel_y, ang_vel_yaw, heading（heading 模式关闭时不用）
        resampling_time = 5.    # 指令重采样间隔，单位秒（每 5 秒随机切换目标速度）
        heading_command = False  # 关闭朝向指令模式，直接使用 ang_vel_yaw（偏航角速度）

        # 零指令课程：训练初期(0→1500 iter)逐渐增加零速度指令概率从 0% 到 10%
        # 帮助机器人学习稳定站立，防止早期训练只追速度不学平衡
        zero_command_curriculum = {'start_iter': 0, 'end_iter': 1500, 'start_value': 0.0, 'end_value': 0.1}

        # 零指令时额外采样限制角速度的概率（20%），进一步强化站立稳定性训练
        limit_ang_vel_at_zero_command_prob = 0.2

        # 限速指令采样概率（20%），以一定概率采样极端速度（最大/最小），扩展速度分布
        limit_vel_prob = 0.2

        # True = 采样限速时使用连续模式的反转逻辑（避免极端速度过于集中）
        limit_vel_invert_when_continuous = True

        # 限速采样的速度选项：-1=最小值，0=零速，1=最大值
        limit_vel = {"lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]}

        # True = 在有限速指令时停止 heading 更新（防止限速和heading指令冲突）
        stop_heading_at_limit = True

        # True = 动态重采样：以较小幅度初始化指令并逐渐扩大，减少训练初期的指令跳变
        dynamic_resample_commands = True

        # 指令范围课程：随训练进度扩大速度指令范围，避免一开始就给出过大的速度目标
        command_range_curriculum = [
            {
                'iter': 20000,          # 在第 20000 步时扩大速度范围
                'lin_vel_x': [-1.0, 1.0],    # x 线速度范围扩大到 ±1 m/s
                'lin_vel_y': [-1.0, 1.0],    # y 线速度范围扩大到 ±1 m/s
                'ang_vel_yaw': [-1.5, 1.5],  # 偏航角速度扩大到 ±1.5 rad/s
                'heading': [-1.57, 1.57],    # 朝向范围 ±90°（heading 模式关闭时不生效）
            },
            {
                'iter': 50000,          # 在第 50000 步时进一步扩大速度范围
                'lin_vel_x': [-2.0, 2.0],    # x 线速度扩大到 ±2 m/s
                'lin_vel_y': [-1.0, 1.0],    # y 线速度保持 ±1 m/s
                'ang_vel_yaw': [-2.0, 2.0],  # 偏航角速度扩大到 ±2 rad/s
                'heading': [-1.57, 1.57],
            }
        ]

        # 翻转恢复后的零速保持时间（turn_over=False 时不生效）
        turn_over_zero_time = {
            "backflip": 5.0,   # 后空翻恢复后保持零速 5 秒
            "sideflip": 3.0,   # 侧翻恢复后保持零速 3 秒
        }

        # 各地形类型对应的最大速度指令限制
        # 复杂地形（楼梯/障碍）限速较低，平地允许更高速度
        terrain_max_command_ranges = [
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # wave（波浪地形）
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # slope（斜坡）
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # rough_slope（粗糙斜坡）
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs_up（上楼梯，限速）
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs_down（下楼梯，限速）
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # obstacles（障碍物，限速）
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stepping_stones（垫脚石，限速）
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # gap（沟壑，限速）
            {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0], 'heading': [-1.57, 1.57]},  # flat（平地，允许最高速度）
        ]

        class ranges:
            """训练初期的速度指令采样范围（较保守，后续由 command_range_curriculum 逐步扩大）。"""
            lin_vel_x = [-0.5, 0.5]     # 前后线速度初始范围，单位 m/s
            lin_vel_y = [-0.5, 0.5]     # 横向线速度初始范围，单位 m/s
            ang_vel_yaw = [-1.0, 1.0]   # 偏航角速度初始范围，单位 rad/s
            heading = [-1.57, 1.57]     # 朝向范围 ±π/2（heading 模式关闭时不生效）

    class asset(LeggedRobotCfg.asset):
        """机器人模型资产配置。"""

        # 使用不含 hm 配重模块的 URDF（withouthm 版本）
        # 相比 with_hm 版本（22.5kg），不含配重质量约 19.1kg，更轻
        # 足端碰撞球位于关节原点(0,0,0)，无偏移（with_hm 版本有偏移）
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/boying_description/urdf/boying_description_withouthm.urdf'

        name = "boying"  # 资产名称，用于 IsaacGym 内部标识

        foot_name = "foot"  # 足端 link 名称，用于识别接触点和计算足端奖励

        # 接触惩罚的 link 名称（大腿/小腿碰地则惩罚，鼓励机器人抬腿而非拖腿）
        penalize_contacts_on = ["thigh", "calf"]

        # 触发 episode 终止的 link 名称（机身碰地 = 摔倒 = 终止并重置）
        terminate_after_contacts_on = ["base"]

        # 自碰撞过滤：0=启用自碰撞检测（相邻link间除外），1=完全禁用自碰撞
        self_collisions = 0

    class rewards(LeggedRobotCfg.rewards):
        """
        奖励函数配置。
        所有惩罚项系数均根据 Boying 与 Go2 的物理参数差异进行了比例调整：
          - torques/dof_acc/dof_power: Kp 差 3 倍 → 系数缩小 3-5 倍
          - ang_vel_xy: 质量更重 → 系数适度缩小
          - feet_regulation: withouthm 足端在关节原点 → 系数适度增大
          - hip_to_default: 关节范围更窄 → 系数适度缩小
        """

        soft_dof_pos_limit = 0.9  # 关节位置软限制比例，超过 90% 的关节范围时触发惩罚

        # 机身目标高度，单位 m
        # Boying withouthm 正向运动学估算站立高度约 0.336m，0.36m 略高于实际提供余量
        base_height_target = 0.36

        only_positive_rewards = False  # False = 允许总奖励为负（更严格的惩罚约束）

        # 足端接触力惩罚阈值，单位 N
        # Boying withouthm 质量约 19.1kg，重力约 187N，超过此值视为冲击过大
        # Go2 对应值为 147N（15kg × 9.8）
        max_contact_force = 187.

        # 奖励课程：随训练迭代步数线性插值调整奖励系数
        curriculum_rewards = [
            # lin_vel_z 惩罚：训练初期(0→1500 iter)系数从 1.0 线性降至 0.0
            # 初期需要较强惩罚防止机器人跳跃，后期可放松让步态更自然
            {'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
            # correct_base_height 惩罚：训练初期(0→5000 iter)系数从 1.0 线性增至 10.0
            # 先用较小系数让机器人学会站立，再逐步加强高度精度要求
            {'reward_name': 'correct_base_height', 'start_iter': 0, 'end_iter': 5000, 'start_value': 1.0, 'end_value': 10.0},
        ]

        # 速度跟踪奖励的高斯核宽度：reward = exp(-error² / sigma)
        # sigma=0.25 时，0.5m/s 的跟踪误差对应约 37% 的奖励衰减
        tracking_sigma = 0.25

        # 动态 sigma：根据指令速度大小和地形类型动态调整跟踪奖励的宽松程度
        # 高速/复杂地形时放宽 sigma（允许更大误差），鼓励机器人尝试
        dynamic_sigma = {
            "min_lin_vel": 0.5,   # 低于此线速度时使用默认 sigma
            "max_lin_vel": 1.5,   # 高于此线速度时使用 max_sigma
            "min_ang_vel": 1.0,   # 低于此角速度时使用默认 sigma
            "max_ang_vel": 2.0,   # 高于此角速度时使用 max_sigma
            # 各地形类型的最大 sigma 值（顺序：wave/slope/rough_slope/stairs_up/stairs_down/obstacles/stepping_stones/gap/flat）
            # 楼梯/障碍地形 sigma 更大（更宽松），平地 sigma 更小（更严格）
            "max_sigma": [5/12, 1/4, 1/4, 1/2, 1/2, 3/4, 1, 1, 1/4]
        }

        min_legs_distance = 0.1  # 判定腿部碰撞的最小腿间距，单位 m（小于此值视为腿部干涉）

        class scales:
            """
            各奖励项的系数（正值=奖励，负值=惩罚）。
            所有系数均考虑了 Boying 与 Go2 的物理参数差异。
            """

            # ── 跟踪奖励（正值）──────────────────────────────────────────
            tracking_lin_vel = 1.0   # 线速度跟踪奖励，高斯形式：exp(-||v_cmd - v||² / sigma)
            tracking_ang_vel = 0.5   # 偏航角速度跟踪奖励，权重为线速度的一半

            # ── 运动质量惩罚（负值）─────────────────────────────────────
            lin_vel_z = -2.0         # 垂直线速度惩罚，抑制跳跃/弹跳行为
            ang_vel_xy = -0.02       # 横滚/俯仰角速度惩罚，鼓励机身保持水平
                                     # Go2=-0.05；Boying 质量更重，惯性大，适度放宽为 -0.02

            dof_acc = -5e-8          # 关节加速度惩罚，抑制抖振和不平滑动作
                                     # Go2=-2.5e-7；Boying Kp=60，高刚度导致加速度量级更大，缩小 5 倍

            dof_power = -1e-5        # 关节功率惩罚（力矩 × 速度），鼓励节能运动
                                     # Go2=-2e-5；Boying 高力矩导致功率量级更大，缩小约 2 倍

            torques = -3e-5          # 关节力矩惩罚，防止过大电机输出
                                     # Go2=-1e-4；Boying Kp=60 vs Go2 Kp=20，力矩约大 3 倍，缩小 ~3.3 倍

            correct_base_height = -1.0  # 机身高度偏差惩罚，配合 base_height_target=0.36m 使用
                                        # 实际系数由 curriculum_rewards 在 0→5000 iter 从 1.0 增至 10.0

            action_rate = -0.01      # 动作变化率惩罚，抑制相邻步之间动作的剧烈变化
            action_smoothness = -0.01  # 动作平滑性惩罚，进一步抑制高频颤振

            collision = -1.0         # 碰撞惩罚（大腿/小腿接触地面），鼓励正常抬腿步态

            dof_pos_limits = -2.0    # 关节位置越限惩罚，超过 soft_dof_pos_limit=0.9 时触发
                                     # Boying 关节范围：髋 ±0.681rad，大腿 -0.175~4.712rad，小腿 -2.748~-0.927rad

            feet_regulation = -0.1   # 足端调节惩罚（防止足端过度内/外偏）
                                     # Go2=-0.05；withouthm 足端在关节原点，自然位置信号弱，适度增大为 -0.1

            hip_to_default = -0.03   # 髋关节归位惩罚，惩罚髋关节偏离默认角度
                                     # Go2=-0.05；Boying 髋关节范围 ±0.681rad（±39°），比 Go2 ±0.872rad 窄，缩小为 -0.03
                                     # 对应 boying_env.py 中的 _reward_hip_to_default 方法

            # x_command_hip_regular 未启用（boying_env.py 中有实现，但此处未列出）
            # 若需启用：x_command_hip_regular = -0.5（有前进指令时惩罚髋关节左右不对称）

        turn_over_roll_threshold = math.pi / 4  # 翻倒检测 roll 角阈值（45°），超过则切换为翻身恢复奖励

        class turn_over_scales:
            """翻身恢复时替换使用的奖励系数（turn_over=False 时不生效）。"""
            upright = 1.0  # 直立奖励，鼓励机器人从翻倒状态恢复到站立

    class noise(LeggedRobotCfg.noise):
        """观测噪声配置。"""
        add_noise = True  # 启用观测噪声注入，具体噪声幅度在 boying_env._get_noise_scale_vec 中定义


class BoyingCfgMoECTS(LeggedRobotCfgMoECTS):
    """
    Boying 机器人的 MoE-CTS 算法配置（boying_moe_cts 任务）。

    MoE-CTS = Mixture of Experts + Concurrent Teacher-Student：
      - Teacher（教师）：使用 privileged_obs(263维) 训练，能看到地形高度图和真实速度
      - Student（学生）：使用 obs(45维) 训练，通过模仿教师的潜空间表示来蒸馏知识
      - MoE（混合专家）：学生网络由 8 个专家子网络组成，由门控网络动态路由，
        不同专家自发专注于不同运动模式（快走/慢走/转弯/爬坡等）
    """

    class policy(LeggedRobotCfgMoECTS.policy):
        expert_num = 8  # MoE 专家数量，8 个专家分别处理不同运动场景

    class runner(LeggedRobotCfgMoECTS.runner):
        run_name = ''                       # 运行名称后缀（空字符串则自动使用时间戳）
        experiment_name = 'boying_moe_cts'  # 实验名称，对应日志目录 logs/boying_moe_cts/
        max_iterations = 150000             # 最大训练迭代步数（约等于 Go2 的训练量）
        save_interval = 500                 # 每 500 步保存一次 checkpoint（model_500.pt, model_1000.pt ...）
