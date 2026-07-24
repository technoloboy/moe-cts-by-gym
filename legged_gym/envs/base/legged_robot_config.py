# legged_robot_config.py
# 腿足机器人训练的所有配置基类，定义了环境、地形、奖励、噪声、算法等全部超参数。
# 具体机器人（Go2、Boying 等）通过继承并覆写相应字段来定制化配置。
# 算法配置分支：PPO（基础）→ CTS（师生）→ MoECTS（混合专家师生，boying_moe_cts 使用）

import math          # 用于 math.pi 计算翻倒检测阈值
from .base_config import BaseConfig  # 基类，提供 __init_subclass__ 等通用配置工具

class LeggedRobotCfg(BaseConfig):
    """
    腿足机器人环境配置基类。
    所有具体机器人配置（BoyingCfg、GO2Cfg 等）均继承此类，
    只覆写与本体相关的字段（资产路径、PD 增益、奖励系数等）。
    """

    class env:
        """并行仿真环境的基本规模和观测空间配置。"""
        num_envs = 4096           # 并行环境数量，默认 4096；8192 适用于 24GB+ 显存
        num_observations = 48     # 学生策略观测维度（基类默认值，子类通常覆写为 45）
        num_privileged_obs = None # 教师特权观测维度；None=不使用特权观测（对称训练）
                                  # 不为 None 时 step() 返回 privileged_obs_buf（CTS/MoE 训练用）
        num_actions = 12          # 动作维度，四足机器人 = 12（每条腿 3 个关节）
        env_spacing = 3.          # 环境间距（米），仅用于平面/无地形模式，trimesh 模式不使用
        send_timeouts = True      # True = 将 episode 超时信号发送给算法（用于正确处理 episode 边界）
        episode_length_s = 20     # 单 episode 最大时长（秒），超时后自动重置
        test = False              # True = 测试模式（关闭随机化，用于部署验证）

    class terrain:
        """
        仿真地形配置，控制地形类型、分辨率、课程难度递进。
        地形课程（curriculum）是腿足机器人 RL 的关键：机器人表现好则升到更难地形，否则降级。
        """
        mesh_type = 'trimesh'     # 地形网格类型：'none'=无地形, 'plane'=平面, 'heightfield'=高度场, 'trimesh'=三角网格
                                  # trimesh 最真实，支持楼梯/斜坡/障碍；plane 最快，用于快速验证
        horizontal_scale = 0.1   # 地形水平分辨率（米/格），影响地形细节精度
        vertical_scale = 0.005   # 地形垂直分辨率（米），高度精度
        border_size = 25         # 地形边界缓冲区大小（米），防止机器人走出地形范围
        curriculum = True        # True = 启用地形课程，机器人随训练进度升/降难度等级
        static_friction = 1.0    # 静摩擦系数（基础值，域随机化会在此基础上随机扰动）
        dynamic_friction = 1.0   # 动摩擦系数
        restitution = 0.         # 地形弹性恢复系数（0=完全非弹性，无弹跳）

        # 地形高度测量（用于特权观测中的高度图）
        measure_heights = True   # True = 在机器人周围采样地形高度，作为教师特权观测输入
        # x 方向高度采样点（相对机器人基座），共 17 个点，覆盖前后各 0.8m
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # y 方向高度采样点，共 11 个点，覆盖左右各 0.5m；17×11=187 个采样点 = privileged_obs 中的高度图维度
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        selected = False         # True = 强制选择单一地形类型（配合 terrain_kwargs 使用）
        terrain_kwargs = None    # selected=True 时传入的地形参数字典

        max_init_terrain_level = 5  # 训练初始阶段机器人的最高起始地形等级（0~num_rows-1）
        terrain_length = 8.      # 单块地形长度（米）
        terrain_width = 8.       # 单块地形宽度（米）
        num_rows = 10            # 地形课程行数（难度等级数），行越高越难
        num_cols = 20            # 地形列数（类型变体数），每行 20 个随机变体
        terrain_spacing = 0.5   # 不同地形块间的间隔（米），防止机器人滑出边界

        # 各地形类型的采样比例，顺序固定：
        # [wave, slope, rough_slope, stairs_up, stairs_down, obstacles, stepping_stones, gap, flat]
        terrain_proportions = [0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.0]

        # trimesh 专用：超过此坡度阈值的斜坡会被修正为垂直面（防止机器人穿透陡坡）
        slope_threshold = 0.75

        # 地形难度整体缩放系数（0~1），统一缩放所有地形参数（坡度/台阶/障碍/波浪）
        # 默认 1.0 = 原始 IS_HARD 难度；各任务可覆写以降低整体地形难度
        difficulty_scale = 1.0

        # True = 地形课程下降触发基于累计水平位移；False = 基于绝对位置
        # True 更公平：即使机器人在原地踏步也不会误触发课程升级
        move_down_by_accumulated_xy_command = False

    class commands:
        """
        速度指令配置，控制机器人的目标运动速度采样方式和课程安排。
        指令在每个 episode 内每隔 resampling_time 秒随机重采样一次。
        """
        # 指令维度：lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        # heading 模式开启时 ang_vel_yaw 由朝向误差自动计算，否则直接使用采样的 ang_vel_yaw
        num_commands = 4
        resampling_time = 10.     # 指令重采样间隔（秒），每隔此时间随机切换目标速度
        heading_command = False   # False = 直接使用 ang_vel_yaw 指令；True = 从朝向误差计算偏航速度

        # 零速度指令课程：逐步增加零速指令的采样概率，帮助机器人学习站立平衡
        # None = 不使用零速课程（始终按 ranges 随机采样）
        # 格式：{'start_iter': 0, 'end_iter': 1500, 'start_value': 0.0, 'end_value': 0.1}
        zero_command_curriculum = None

        # 零速指令时额外采样限制角速度的概率（防止零速时仍然旋转）
        limit_ang_vel_at_zero_command_prob = 0.0

        # 以此概率采样极端限速指令（最大/最小），拓展训练分布
        limit_vel_prob = 0.0

        # True = 连续限速采样时使用反转逻辑（避免极端速度采样过于集中）
        limit_vel_invert_when_continuous = True

        # 限速采样的速度选项映射：-1=最小值，0=零速，1=最大值
        limit_vel = {"lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]}

        # True = 有限速指令时暂停 heading 更新（防止限速和 heading 指令冲突）
        stop_heading_at_limit = True

        # True = 动态重采样：以较小初始幅度采样并逐渐扩大，减少训练初期的指令跳变
        dynamic_resample_commands = False

        # 指令范围课程列表：在指定训练迭代步数时扩大速度指令范围
        # 空列表 = 始终使用 ranges 中的固定范围
        # 格式：[{'iter': 20000, 'lin_vel_x': [-1.0, 1.0], ...}]
        command_range_curriculum = []

        # 翻转恢复后机器人保持零速的等待时间（turn_over=False 时不生效）
        turn_over_zero_time = {
            "backflip": 5.0,  # 后空翻恢复后零速保持时间（秒）
            "sideflip": 3.0,  # 侧翻恢复后零速保持时间（秒）
        }

        # 各地形类型对应的最大速度指令限制（防止在困难地形上给出过高速度目标）
        # 顺序：[wave, slope, rough_slope, stairs_up, stairs_down, obstacles, stepping_stones, gap, flat]
        terrain_max_command_ranges = [
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.5, 1.5], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # wave（波浪）
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.5, 1.5], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # slope（斜坡）
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.5, 1.5], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # rough_slope（粗糙斜坡）
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs_up（上楼梯，限速）
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs_down（下楼梯，限速）
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # obstacles（障碍物，限速）
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stepping_stones（垫脚石，限速）
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # gap（沟壑，限速）
            {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.5, 1.5], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # flat（平地，允许最高速度）
        ]

        class ranges:
            """训练初始阶段的速度指令采样范围（由 command_range_curriculum 逐步扩大）。"""
            lin_vel_x = [-1.0, 1.0]    # 前后线速度范围（米/秒）
            lin_vel_y = [-0.5, 0.5]    # 横向线速度范围（米/秒）
            ang_vel_yaw = [-1, 1]      # 偏航角速度范围（弧度/秒）
            heading = [-3.14, 3.14]    # 朝向角范围（弧度），仅 heading_command=True 时有效

    class init_state:
        """机器人初始化状态，定义 episode 开始时的位置、姿态和关节角度。"""
        pos = [0.0, 0.0, 1.]          # 初始质心位置 [x, y, z]，单位米；z 需高于实际站立高度防穿地
        rot = [0.0, 0.0, 0.0, 1.0]   # 初始四元数姿态 [x, y, z, w]，[0,0,0,1] = 无旋转
        lin_vel = [0.0, 0.0, 0.0]    # 初始线速度 [x, y, z]，单位 m/s，通常为零
        ang_vel = [0.0, 0.0, 0.0]    # 初始角速度 [x, y, z]，单位 rad/s，通常为零
        default_joint_angles = {      # 动作为 0 时的目标关节角度（默认姿态），单位 rad
            "joint_a": 0.,            # 基类占位符，子类覆写为实际关节名称和角度
            "joint_b": 0.}
        turn_over = False             # False = 正常初始化；True = 以翻倒姿态初始化（训练翻身恢复）
        turn_over_proportions = [0.0, 0.2, 0.8]  # 翻倒类型比例 [后空翻, 侧翻, 正常]
        turn_over_init_heights = {    # 各翻倒类型的初始高度范围 [min, max]，单位米
            'backflip': [0.10, 0.15], # 后空翻初始高度
            'sideflip': [0.16, 0.21], # 侧翻初始高度
        }

    class control:
        """
        关节控制器配置（PD 位置控制）。
        策略输出目标关节角度偏置（action），经 PD 控制器转换为力矩施加到仿真中。
        实际力矩 = Kp × (target_pos - current_pos) - Kd × current_vel
        target_pos = action_scale × action + default_angle
        """
        control_type = 'P'   # 控制类型：'P'=位置控制（PD）, 'V'=速度控制, 'T'=力矩控制
        # PD 增益（子类覆写为实际机器人参数）
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # Kp，位置增益，单位 N*m/rad
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # Kd，速度阻尼，单位 N*m*s/rad
        # 动作缩放：target_angle = action_scale × action + default_angle
        # 限制策略单步最大关节位移，防止过激动作导致仿真不稳定
        action_scale = 0.5
        # 控制频率细分：每个策略步执行 decimation 次物理仿真步
        # 仿真 dt=0.005s，decimation=4 → 策略频率 = 1/(0.005×4) = 50Hz
        decimation = 4

    class asset:
        """
        机器人模型资产配置，控制 URDF 加载方式和物理属性。
        """
        file = ""              # URDF 文件路径（必须在子类中覆写）
        name = "legged_robot"  # IsaacGym 中的 actor 名称标识符
        foot_name = "None"     # 足端 link 名称，用于索引足端接触力和足端状态张量
        penalize_contacts_on = []           # 接触惩罚 link 列表（如 ["thigh", "calf"]）
        terminate_after_contacts_on = []    # 触发终止的 link 列表（如 ["base"]，机身触地=摔倒=重置）
        disable_gravity = False            # True = 关闭重力（调试用）
        # True = 合并固定关节连接的 body（减少自由度，提升仿真速度）
        # 特定固定关节可通过在 URDF 中添加 dont_collapse="true" 属性来保留
        # Boying withouthm 的 foot 关节使用了 dont_collapse="true"
        collapse_fixed_joints = True
        fix_base_link = False              # True = 固定机器人基座（用于调试关节行为）
        # 关节驱动模式（见 GymDofDriveModeFlags）：0=无驱动, 1=位置目标, 2=速度目标, 3=力矩驱动
        # 通常设为 3（力矩驱动），由控制器计算力矩后施加
        default_dof_drive_mode = 3
        self_collisions = 0                # 自碰撞过滤：0=启用自碰撞, 1=完全禁用自碰撞
        # True = 将碰撞圆柱体替换为胶囊体，仿真速度更快且更稳定
        replace_cylinder_with_capsule = True
        # True = 将视觉网格从 y-up 翻转为 z-up（部分 .obj 文件需要）
        flip_visual_attachments = True

        density = 0.001               # 默认密度（kg/m³），通常由 URDF 中的惯量参数覆盖
        angular_damping = 0.          # 角速度阻尼（用于模拟空气阻力等，通常设为 0）
        linear_damping = 0.           # 线速度阻尼
        max_angular_velocity = 1000.  # 最大允许角速度（rad/s），超过则被截断
        max_linear_velocity = 1000.   # 最大允许线速度（m/s）
        armature = 0.                 # 关节惯量（kg*m²），模拟电机转子惯量，通常设为 0
        thickness = 0.01              # 碰撞体厚度（米），影响接触检测灵敏度

    class domain_rand:
        """
        域随机化（Domain Randomization）配置。
        通过在仿真中随机化物理参数，训练出对真实世界参数不确定性鲁棒的策略，
        是 sim-to-real 迁移的核心手段。
        """
        ### 机器人本体属性随机化（每次 episode reset 时重采样） ###

        robot_properties_update = None  # 机器人属性更新计划，格式：{'start_iter': 5000, 'interval': 5000}
                                        # 控制何时开始和多久更新一次域随机化范围

        randomize_friction = True       # 启用地面摩擦系数随机化
        friction_range = [0.2, 1.25]    # 摩擦系数采样范围（覆盖光滑到粗糙地面）

        randomize_base_mass = True      # 启用机身附加质量随机化（模拟携带不同重量负载）
        added_mass_range = [-1., 1.]    # 附加质量范围（千克），负值=减重，正值=加重

        randomize_link_mass = True      # 启用连杆质量倍率随机化（模拟制造误差）
        multiplied_link_mass_range = [0.9, 1.1]  # 连杆质量倍率范围（±10%）

        randomize_base_com = True       # 启用质心位置随机化（模拟负载偏心、重心偏移）
        added_base_com_range = [-0.03, 0.03]  # 质心偏移范围（米），各轴独立随机

        randomize_restitution = False   # 启用弹性恢复系数随机化（机器人连杆的弹性）
        restitution_range = [0.0, 0.2]  # 恢复系数范围（0=非弹性）

        ### 环境重置时的参数随机化 ###

        randomize_pd_gains = True                   # 启用 PD 增益随机化（模拟电机参数误差）
        stiffness_multiplier_range = [0.9, 1.1]     # Kp 倍率范围（±10%）
        damping_multiplier_range = [0.9, 1.1]       # Kd 倍率范围（±10%）

        randomize_motor_zero_offset = True          # 启用电机零位偏移随机化（模拟标定误差）
        motor_zero_offset_range = [-0.035, 0.035]   # 零位偏移范围（弧度，约 ±2°）

        randomize_motor_strength = False            # 启用电机力矩强度随机化（模拟老化/温度效应）
        motor_strength_range = [0.8, 1.2]           # 力矩强度倍率范围（±20%）

        ### 仿真运行过程中的随机扰动 ###

        push_robots = True          # 启用随机推力扰动，提升策略对外力干扰的鲁棒性
        push_interval_s = 4         # 施加推力的时间间隔（秒）
        max_push_vel_xy = 0.4       # 水平推力引起的最大速度变化（m/s）
        max_push_ang_vel = 0.6      # 旋转推力引起的最大角速度变化（rad/s）

        # 启用动作延迟随机化：以 0~20ms 延迟使用上一步动作（模拟真实控制通信延迟）
        # decimation=4 时，最大延迟 = 1 个控制步（20ms）
        randomize_action_delay = False

    class rewards:
        """
        奖励函数配置基类，定义默认奖励项系数和相关超参数。
        总奖励 = Σ(scale_i × reward_i)，scale 为负 = 惩罚项。
        子类覆写 scales 中的字段来启用/调整各奖励项。
        """
        class scales:
            """默认奖励系数（基类），子类（BoyingCfg/GO2Cfg）通过覆写来定制。"""
            termination = -0.0          # episode 异常终止惩罚（如摔倒）
            tracking_lin_vel = 1.0      # 线速度跟踪奖励：exp(-||v_cmd - v||² / sigma)
            tracking_ang_vel = 0.5      # 偏航角速度跟踪奖励
            lin_vel_z = -2.0            # 垂直线速度惩罚，抑制跳跃弹跳
            ang_vel_xy = -0.05          # 横滚/俯仰角速度惩罚，鼓励机身水平
            orientation = -0.           # 机身朝向惩罚（0=禁用）
            torques = -0.00001          # 关节力矩惩罚，鼓励节能
            dof_vel = -0.               # 关节速度惩罚（0=禁用）
            dof_acc = -2.5e-7           # 关节加速度惩罚，抑制高频抖振
            base_height = -0.           # 机身高度惩罚（0=禁用，由 correct_base_height 替代）
            feet_air_time = 1.0         # 足端腾空时间奖励，鼓励迈步而非拖步
            collision = -1.             # 碰撞惩罚（大腿/小腿接触地面）
            feet_stumble = -0.0         # 脚部绊倒惩罚（0=禁用）
            action_rate = -0.01         # 动作变化率惩罚，抑制相邻步动作剧烈变化
            stand_still = -0.           # 静止惩罚（0=禁用）

        class turn_over_scales:
            """翻身恢复模式下替换使用的奖励系数（turn_over=True 时生效）。"""
            upright = 1.0               # 直立奖励，引导机器人从翻倒状态恢复站立

        # True = 总奖励负值时截断为 0（避免早期终止时的梯度问题）
        # False = 允许负总奖励（更严格的惩罚约束，boying/go2 均使用 False）
        only_positive_rewards = True

        # 速度跟踪奖励的高斯核宽度：reward = exp(-error² / sigma)
        # sigma 越小要求越精确；0.25 时 0.5m/s 误差对应约 37% 奖励衰减
        tracking_sigma = 0.25

        # 关节位置软限制：超过 URDF 关节范围的此比例时开始惩罚
        soft_dof_pos_limit = 1.   # 1.0 = 在 100% 范围处惩罚（即不惩罚）；0.9 = 超过 90% 时惩罚
        soft_dof_vel_limit = 1.   # 关节速度软限制比例
        soft_torque_limit = 1.    # 关节力矩软限制比例

        base_height_target = 1.           # 机身目标高度（米），子类覆写为实际值（Boying=0.36）
        max_contact_force = 100.          # 足端接触力惩罚阈值（牛顿），超过则惩罚（Boying=187N）

        # 奖励课程：在训练过程中线性调整奖励系数
        # None = 不使用课程；子类设置列表来启用
        # 格式：[{'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0}]
        curriculum_rewards = None

        # 动态 sigma：根据指令速度和地形类型动态调整跟踪奖励宽松程度
        # None = 使用固定 tracking_sigma；子类设置 dict 来启用
        # 需要地形课程先运行（**Must start terrain curriculum first**）
        dynamic_sigma = None

        # 翻倒检测的 roll 角阈值（弧度）：超过此值切换为翻身恢复奖励
        turn_over_roll_threshold = math.pi / 4  # 45°

        # 腿间距最小值（米）：两腿距离小于此值视为腿部干涉/绊倒
        min_legs_distance = 0.1

    class normalization:
        """观测归一化和动作截断配置。"""
        class obs_scales:
            """各观测分量的缩放因子，将原始物理量映射到约 [-1, 1] 范围，提升训练稳定性。"""
            lin_vel = 2.0               # 线速度缩放：1 m/s × 2.0 = 2.0（观测值）
            ang_vel = 0.25              # 角速度缩放：4 rad/s × 0.25 = 1.0
            dof_pos = 1.0               # 关节位置缩放：通常偏差在 ±1 rad 内，无需缩放
            dof_vel = 0.05              # 关节速度缩放：20 rad/s × 0.05 = 1.0
            height_measurements = 2.5  # 高度图缩放：0.4m × 2.5 = 1.0
        clip_observations = 100.       # 观测值截断范围（±100），防止异常值破坏训练
        clip_actions = 100.            # 动作值截断范围（±100），通常远大于实际动作范围

    class noise:
        """训练时注入到学生观测中的噪声配置，模拟真实传感器误差。"""
        add_noise = True               # True = 启用噪声注入（训练时开启，部署时可关闭）
        noise_level = 1.0              # 全局噪声强度缩放因子（1.0=正常，可在课程中动态调整）
        class noise_scales:
            """各观测分量的噪声幅度（乘以 noise_level 后为最终噪声范围）。"""
            dof_pos = 0.01             # 关节位置噪声（弧度），约 ±0.6°
            dof_vel = 1.5              # 关节速度噪声（rad/s），较大以模拟真实编码器噪声
            lin_vel = 0.1              # 线速度噪声（m/s），模拟速度估计误差
            ang_vel = 0.2              # 角速度噪声（rad/s），模拟 IMU 陀螺仪噪声
            gravity = 0.05            # 投影重力向量噪声，模拟 IMU 加速度计噪声
            height_measurements = 0.1 # 高度图噪声（米），模拟激光雷达/深度相机测量误差

    class viewer:
        """IsaacGym 可视化窗口相机配置（仅用于调试可视化，不影响训练）。"""
        ref_env = 0           # 相机跟随的参考环境索引（通常跟随第 0 号环境）
        pos = [10, 0, 6]      # 相机位置 [x, y, z]，单位米
        lookat = [11., 5, 3.] # 相机注视点 [x, y, z]，单位米

    class sim:
        """IsaacGym 物理仿真参数配置。"""
        dt = 0.005        # 仿真时间步长（秒），200Hz；策略步长 = dt × decimation = 0.02s（50Hz）
        substeps = 1      # 每个仿真步的子步数（>1 可提升接触稳定性，但降低速度）
        gravity = [0., 0., -9.81]  # 重力加速度向量（m/s²），z 轴向下
        up_axis = 1       # 上方向轴：0=y 轴向上，1=z 轴向上（IsaacGym 默认 z 向上）

        class physx:
            """PhysX 物理引擎配置（IsaacGym 使用 NVIDIA PhysX）。"""
            num_threads = 10              # CPU 物理计算线程数
            solver_type = 1              # 约束求解器：0=PGS（投影高斯-赛德尔），1=TGS（时序高斯-赛德尔，更稳定）
            num_position_iterations = 4  # 位置约束迭代次数，越多越精确但越慢
            num_velocity_iterations = 0  # 速度约束迭代次数（0 对腿足机器人通常足够）
            contact_offset = 0.01        # 接触检测偏移（米），值越大接触检测越早触发
            rest_offset = 0.0            # 静止偏移（米），接触分离的最小距离
            bounce_threshold_velocity = 0.5  # 弹跳速度阈值（m/s），低于此值的碰撞不产生弹跳
            max_depenetration_velocity = 1.0 # 最大穿透恢复速度（m/s），防止穿透后过快弹出
            max_gpu_contact_pairs = 2**23    # GPU 最大接触对数；8000+ 环境可能需要 2**24
            default_buffer_size_multiplier = 5  # PhysX 内部缓冲区大小倍率
            contact_collection = 2       # 接触力收集时机：0=不收集, 1=最后子步, 2=所有子步（默认）

class LeggedRobotCfgPPO(BaseConfig):
    """
    PPO（近端策略优化）算法配置。
    最基础的策略梯度算法，适合入门和基准对比。
    boying_moe_cts 不使用此配置（使用 MoECTS），保留供其他任务使用。
    """
    seed = 1                               # 随机种子，保证实验可复现
    runner_class_name = 'OnPolicyRunner'   # runner 类名，由 task_registry 动态实例化

    class policy:
        """Actor-Critic 网络结构配置。"""
        init_noise_std = 1.0               # 动作分布初始标准差（探索噪声大小）
        actor_hidden_dims = [512, 256, 128]   # Actor 网络隐藏层维度
        critic_hidden_dims = [512, 256, 128]  # Critic 网络隐藏层维度
        activation = 'elu'                 # 激活函数：elu/relu/selu/crelu/lrelu/tanh/sigmoid

    class algorithm:
        """PPO 算法超参数。"""
        value_loss_coef = 1.0              # Critic 值函数损失系数
        use_clipped_value_loss = True      # True = 使用截断的值函数损失（稳定训练）
        clip_param = 0.2                   # PPO 截断参数 ε，限制策略更新幅度
        entropy_coef = 0.01               # 熵正则化系数，鼓励探索
        num_learning_epochs = 5            # 每次数据收集后的学习轮数
        num_mini_batches = 4               # mini-batch 数量；batch_size = num_envs × num_steps / num_mini_batches
        learning_rate = 1.e-3              # 学习率
        schedule = 'adaptive'             # 学习率调度：'adaptive'=根据 KL 散度自适应调整，'fixed'=固定
        gamma = 0.99                       # 折扣因子，控制未来奖励的权重
        lam = 0.95                         # GAE（广义优势估计）的 λ 参数
        desired_kl = 0.01                  # 目标 KL 散度（adaptive 调度时使用）
        max_grad_norm = 1.                 # 梯度裁剪阈值，防止梯度爆炸

    class runner:
        """PPO 训练 runner 配置。"""
        policy_class_name = 'ActorCritic'     # 策略类名
        algorithm_class_name = 'PPO'          # 算法类名
        num_steps_per_env = 24                # 每次迭代每个环境收集的步数
        max_iterations = 1500                 # 最大策略更新次数

        # 日志和保存
        save_interval = 50                    # 每隔多少次迭代检查是否保存 checkpoint
        experiment_name = 'test'              # 实验名称（对应 logs/ 下的目录）
        run_name = ''                         # 运行名称后缀（空 = 使用时间戳）

        # 断点续训
        resume = False                        # True = 从 checkpoint 恢复训练
        load_run = -1                         # 要加载的运行目录；-1 = 最新一次运行
        checkpoint = -1                       # 要加载的 checkpoint 编号；-1 = 最新 checkpoint
        resume_path = None                    # 由 load_run 和 checkpoint 自动构建的完整路径

    class robogauge:
        """RoboGauge 评估接口配置（用于在线性能评估）。"""
        enabled = False   # False = 禁用 RoboGauge（默认关闭）
        port = 9973        # RoboGauge 服务端口

class LeggedRobotCfgCTS(BaseConfig):
    """
    CTS（Concurrent Teacher-Student，并发师生训练）算法配置。
    在同一次仿真中同时训练教师策略（使用特权观测）和学生策略（使用真实可观测量）。
    学生通过模仿教师的潜空间表示进行知识蒸馏，无需单独预训练教师。
    boying_moe_cts 基于此类（通过 LeggedRobotCfgMoECTS 继承）。
    """
    seed = 0                                    # 随机种子
    runner_class_name = "OnPolicyRunnerCTS"     # CTS 专用 runner 类名
    history_length = 5                          # 历史观测长度（用于时序编码器）

    class policy:
        """CTS 策略网络结构配置。"""
        init_noise_std = 1.0                           # 动作分布初始标准差
        actor_hidden_dims = [512, 256, 128]            # Actor（策略）网络隐藏层
        critic_hidden_dims = [512, 256, 128]           # Critic（值函数）网络隐藏层
        teacher_encoder_hidden_dims = [512, 256]       # 教师编码器隐藏层（将特权观测编码为潜向量）
        student_encoder_hidden_dims = [512, 256]       # 学生编码器隐藏层（将历史普通观测编码为潜向量）
        activation = 'elu'                             # 激活函数
        latent_dim = 32                                # 师生共享潜空间维度（编码器输出维度）
        norm_type = 'l2norm'                           # 编码器输出归一化方式：'l2norm'=L2归一化，'simnorm'=SimNorm

    class algorithm:
        """CTS 算法超参数（继承 PPO 大部分参数）。"""
        value_loss_coef = 1.0              # Critic 损失系数
        use_clipped_value_loss = True      # 使用截断值函数损失
        clip_param = 0.2                   # PPO 截断参数
        entropy_coef = 0.01               # 熵正则化系数
        num_learning_epochs = 5            # 每迭代学习轮数
        num_mini_batches = 4               # mini-batch 数量
        learning_rate = 1.e-3             # Actor-Critic 学习率
        student_encoder_learning_rate = 1e-3  # 学生编码器独立学习率（可与 Actor 解耦调整）
        schedule = 'adaptive'             # 学习率调度策略
        gamma = 0.99                       # 折扣因子
        lam = 0.95                         # GAE λ 参数
        desired_kl = 0.01                  # 目标 KL 散度
        max_grad_norm = 1.                 # 梯度裁剪阈值
        # 教师环境比例：75% 的环境分配给教师策略（使用特权观测），25% 分配给学生
        # 更多教师环境 → 教师学得更好 → 学生有更好的蒸馏目标
        teacher_env_ratio = 0.75

    class runner:
        """CTS runner 配置。"""
        policy_class_name = 'ActorCriticCTS'     # CTS Actor-Critic 类名
        algorithm_class_name = 'CTS'             # CTS 算法类名
        num_steps_per_env = 24                   # 每迭代每环境收集步数
        max_iterations = 1500                    # 最大迭代次数

        save_interval = 50                       # checkpoint 保存间隔
        experiment_name = 'test'                 # 实验名称
        run_name = ''                            # 运行名称后缀

        resume = False                           # 是否从 checkpoint 恢复
        load_run = -1                            # 加载的运行目录（-1=最新）
        checkpoint = -1                          # 加载的 checkpoint 编号（-1=最新）
        resume_path = None                       # 完整 checkpoint 路径（自动构建）

    class robogauge:
        """RoboGauge 评估接口。"""
        enabled = False   # 禁用
        port = 9973

class LeggedRobotCfgMoENGCTS(LeggedRobotCfgCTS):
    """
    MoE-NoGoal-CTS 算法配置。
    在 CTS 基础上加入 MoE（混合专家）结构，且学生编码器不感知速度指令（NoGoal）。
    通过 obs_no_goal_mask 屏蔽指令信息，强迫学生专注于运动状态而非目标。
    """
    class policy(LeggedRobotCfgCTS.policy):
        # 屏蔽观测中的速度指令维度的掩码（None=不屏蔽）
        # 格式：与 obs_buf 等长的 bool 列表，True=保留，False=屏蔽
        obs_no_goal_mask = None
        student_expert_num = 8   # 学生 MoE 专家数量

    class algorithm(LeggedRobotCfgCTS.algorithm):
        load_balance_coef = 0.01  # 专家负载均衡损失系数，防止所有样本集中到少数专家

    class runner(LeggedRobotCfgCTS.runner):
        policy_class_name = 'ActorCriticMoENGCTS'   # MoE-NoGoal-CTS 策略类名
        algorithm_class_name = 'MoENGCTS'            # 算法类名


class LeggedRobotCfgMCPCTS(LeggedRobotCfgCTS):
    """
    MCP-CTS（Multiplicative Compositional Policies + CTS）算法配置。
    MCP 将动作分解为多个原语策略的乘积，每个原语专注于不同运动子任务。
    """
    class policy(LeggedRobotCfgCTS.policy):
        obs_no_goal_mask = None       # 观测屏蔽掩码（同 MoENGCTS）
        student_expert_num = 8        # 原语策略（专家）数量

    class runner(LeggedRobotCfgCTS.runner):
        policy_class_name = 'ActorCriticMCPCTS'   # MCP-CTS 策略类名
        algorithm_class_name = 'MCPCTS'            # 算法类名


class LeggedRobotCfgACMoECTS(LeggedRobotCfgCTS):
    """
    AC-MoE-CTS（Actor-Critic MoE + CTS）算法配置。
    MoE 结构同时应用于 Actor 和 Critic 网络。
    """
    class policy(LeggedRobotCfgCTS.policy):
        expert_num = 8   # Actor 和 Critic 的专家数量

    class runner(LeggedRobotCfgCTS.runner):
        policy_class_name = 'ActorCriticACMoECTS'   # AC-MoE-CTS 策略类名
        algorithm_class_name = 'ACMoECTS'            # 算法类名


class LeggedRobotCfgDualMoECTS(LeggedRobotCfgCTS):
    """
    DualMoE-CTS（双重 MoE + CTS）算法配置。
    在教师和学生两侧都使用 MoE 结构；学生编码器网络更宽（额外一层 256）。
    """
    class policy(LeggedRobotCfgCTS.policy):
        expert_num = 8                                 # MoE 专家数量
        student_encoder_hidden_dims = [512, 256, 256]  # 学生编码器更宽（3层 vs 基类的2层）

    class runner(LeggedRobotCfgCTS.runner):
        policy_class_name = 'ActorCriticDualMoECTS'   # DualMoE-CTS 策略类名
        algorithm_class_name = 'DualMoECTS'            # 算法类名


class LeggedRobotCfgMoECTS(LeggedRobotCfgCTS):
    """
    MoE-CTS（Mixture of Experts + CTS）算法配置基类。
    boying_moe_cts 和 go2_moe_cts 均继承此类。

    架构：
      - 教师：普通 Actor-Critic + 特权观测编码器（单一策略）
      - 学生：MoE Actor（8 个专家 + 门控网络）+ 历史观测编码器
      - 蒸馏：学生编码器模仿教师编码器的输出（潜空间对齐）

    与 DualMoECTS 的区别：只有学生端使用 MoE，教师仍是单一策略，
    更轻量且在实践中效果相当。
    """
    class policy(LeggedRobotCfgCTS.policy):
        expert_num = 8                                 # 学生 MoE 专家数量（子类可覆写）
        student_encoder_hidden_dims = [512, 256, 256]  # 学生编码器（3层，比基类多1层 256）

    class algorithm(LeggedRobotCfgCTS.algorithm):
        load_balance_coef = 0.01  # 专家负载均衡系数，防止专家坍塌（所有样本路由到同一专家）

    class runner(LeggedRobotCfgCTS.runner):
        policy_class_name = 'ActorCriticMoECTS'   # MoE-CTS 策略类名
        algorithm_class_name = 'MoECTS'            # 算法类名
