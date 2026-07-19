# boying_env.py
# Boying 四足机器人的 IsaacGym 训练环境，继承自通用腿足机器人基类 LeggedRobot。
# 本文件定义了 boying_moe_cts 任务专用的观测计算和奖励函数。
# Boying 机器人特点：质量约 19.1kg（不含hm配重），Kp=60，Kd=4.5，比 Go2（Kp=20）刚度高 3 倍。

from legged_gym.envs.base.legged_robot import LeggedRobot  # 导入通用腿足机器人基类，包含 PPO/CTS 训练的全部基础逻辑

from isaacgym.torch_utils import *   # 导入 IsaacGym 的 torch 工具函数（quat_rotate_inverse 等四元数/坐标变换工具）
from isaacgym import gymtorch, gymapi, gymutil  # gymtorch: tensor API；gymapi: 场景/资产管理；gymutil: 辅助工具
import torch  # PyTorch，用于所有张量计算


class BoyingRobot(LeggedRobot):
    """
    Boying 四足机器人环境类（boying_moe_cts 任务）。

    继承 LeggedRobot 的全部训练逻辑（物理仿真、地形课程、CTS/MoE 算法接口），
    仅覆写以下三个方法以适配 Boying 的关节布局和奖励设计：
      - _get_noise_scale_vec: 观测噪声向量（45维obs）
      - compute_observations:  学生/特权观测计算（45维 / 263维）
      - _reward_hip_to_default: 髋关节偏离默认角度的惩罚
      - _reward_x_command_hip_regular: 有前进指令时的髋关节左右对称性奖励
    """

    def _get_noise_scale_vec(self, cfg):
        """
        构建观测噪声缩放向量，与 obs_buf 的维度和顺序严格对应（共 45 维）。
        每个观测分量乘以对应噪声幅度后，在 compute_observations 中以均匀分布随机注入。
        噪声 = noise_scale_vec * (2 * rand - 1)，模拟传感器误差和通信延迟。
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])   # 初始化零向量，shape = (45,)，与单环境观测维度一致

        self.add_noise = self.cfg.noise.add_noise        # 从配置中读取是否启用噪声（BoyingCfg.noise.add_noise = True）
        noise_scales = self.cfg.noise.noise_scales       # 各分量的噪声系数对象（ang_vel/gravity/dof_pos/dof_vel）
        noise_level = self.cfg.noise.noise_level         # 全局噪声强度缩放因子，可在训练中动态调整

        # obs[0:3]   — 机身角速度（roll/pitch/yaw rate），单位 rad/s，已乘 obs_scale 缩放
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel

        # obs[3:6]   — 投影重力向量（projected gravity），表示机身相对世界重力的朝向，无量纲
        noise_vec[3:6] = noise_scales.gravity * noise_level

        # obs[6:9]   — 速度指令（lin_vel_x, lin_vel_y, ang_vel_yaw），指令本身不加噪声，设为 0
        noise_vec[6:9] = 0.  # commands

        # obs[9:21]  — 关节位置偏差（dof_pos - default_dof_pos），12个关节，单位 rad，已乘 obs_scale
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos

        # obs[21:33] — 关节速度（dof_vel），12个关节，单位 rad/s，已乘 obs_scale
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel

        # obs[33:45] — 上一步动作（previous actions），作为历史信息输入，不加噪声
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0.  # previous actions

        return noise_vec  # 返回 shape=(45,) 的噪声向量，被 LeggedRobot 保存为 self.noise_scale_vec

    def compute_observations(self):
        """
        计算每个仿真步的观测值，分为两路：
          - obs_buf (45维):         学生策略的输入，仅含可在真实机器人上获取的传感器信息（无特权信息）
          - privileged_obs_buf (263维): 教师策略（专家）的输入，包含仿真特权信息（真实线速度、地形高度图等）

        CTS（Concurrent Teacher-Student）框架中：
          教师用 privileged_obs 学习高质量动作，学生通过模仿教师的潜空间表示来蒸馏知识。
        MoE（Mixture of Experts）在学生网络中用 8 个专家处理不同运动模式。

        obs_buf 维度分解（共 45 维）：
          [0:3]   base_ang_vel       — 机身角速度 (3)
          [3:6]   projected_gravity  — 投影重力向量 (3)
          [6:9]   commands           — 速度指令 (3): lin_vel_x, lin_vel_y, ang_vel_yaw
          [9:21]  dof_pos_offset     — 关节位置偏差 (12)
          [21:33] dof_vel            — 关节速度 (12)
          [33:45] actions            — 上一步动作 (12)

        privileged_obs_buf 维度分解（共 263 维）：
          [0:3]    base_lin_vel      — 真实线速度 (3)，仿真特权，真机无法直接测量
          [3:6]    base_ang_vel      — 机身角速度 (3)
          [6:9]    projected_gravity — 投影重力向量 (3)
          [9:12]   commands          — 速度指令 (3)
          [12:24]  dof_pos_offset    — 关节位置偏差 (12)
          [24:36]  dof_vel           — 关节速度 (12)
          [36:48]  actions           — 上一步动作 (12)
          [48:52]  foot_contact      — 足端接触力范数 (4)，单位 kN
          [52:64]  torques_norm      — 归一化关节力矩 (12)，除以力矩限制
          [64:76]  dof_acc           — 关节加速度估计 (12)，单位 rad/s²×1e-4
          [76:263] heights           — 地形高度图 (187)，相对机身高度的扫描值
        """

        # ── 学生观测（45维） ──────────────────────────────────────────────────
        self.obs_buf = torch.cat((
            self.base_ang_vel  * self.obs_scales.ang_vel,        # 机身角速度，乘缩放因子归一化
            self.projected_gravity,                               # 投影重力向量（quat_rotate_inverse 计算），表示机身倾斜程度
            self.commands[:, :3] * self.commands_scale,          # 速度指令前3维（x/y线速度 + yaw角速度），乘命令缩放因子
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 关节位置相对默认姿态的偏差
            self.dof_vel * self.obs_scales.dof_vel,              # 关节速度，乘缩放因子
            self.actions,                                         # 上一步输出的12维动作（目标关节角度偏置）
        ), dim=-1)

        # ── 地形高度图处理 ───────────────────────────────────────────────────
        # 相对高度 = 机身z坐标 - 0.5（参考偏置）- 各采样点地形高度，clip 到 [-1, 1]
        # 0.5 是参考基准，使高度值围绕 0 分布；Boying 初始高度 0.40m，站立高度约 0.336m
        heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
            -1, 1.0
        ) * self.obs_scales.height_measurements  # 乘高度缩放因子

        # ── 教师特权观测（263维） ────────────────────────────────────────────
        self.privileged_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,          # 真实线速度（特权：真机用速度估计器近似）
            self.base_ang_vel  * self.obs_scales.ang_vel,         # 机身角速度（与学生obs相同）
            self.projected_gravity,                               # 投影重力向量（与学生obs相同）
            self.commands[:, :3] * self.commands_scale,          # 速度指令（与学生obs相同）
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 关节位置偏差（与学生obs相同）
            self.dof_vel * self.obs_scales.dof_vel,              # 关节速度（与学生obs相同）
            self.actions,                                         # 上一步动作（与学生obs相同）
            torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) * 1e-3,  # 4个足端接触力范数，单位kN（特权）
            self.torques / self.torque_limits,                   # 12个关节力矩归一化值，反映电机负载程度（特权）
            (self.last_dof_vel - self.dof_vel) / self.dt * 1e-4, # 关节加速度估计 = Δvel/dt，乘1e-4缩放（特权）
            heights,                                              # 187点地形高度图，提供局部地形信息给教师（特权）
        ), dim=-1)

        # ── 噪声注入（仅对学生obs） ──────────────────────────────────────────
        if self.add_noise:
            # 均匀分布噪声 [-1, 1] × noise_scale_vec，模拟真实传感器误差
            # 仅加到 obs_buf（学生），privileged_obs_buf 不加噪声（教师在仿真中可获得精确值）
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_hip_to_default(self):
        """
        髋关节归位奖励（惩罚项，scale = -0.03）。

        惩罚所有髋关节偏离默认角度的绝对偏差之和，引导机器人保持自然站姿，
        防止 CTS 训练倾向于将髋关节大幅外展/内收来获取其他奖励。

        Boying URDF 关节顺序（与 Go2 相同）：
          index 0: FL_hip_joint  （左前髋，默认 +0.1 rad）
          index 3: FR_hip_joint  （右前髋，默认 -0.1 rad）
          index 6: RL_hip_joint  （左后髋，默认 +0.1 rad）
          index 9: RR_hip_joint  （右后髋，默认 -0.1 rad）

        Boying 髋关节范围 ±0.681 rad（约 ±39°），比 Go2 的 ±0.872 rad 窄，
        故 scale 设为 -0.03（低于 Go2 的 -0.05），避免对合理偏差过度惩罚。

        Returns:
            (num_envs,) 张量，每个环境4个髋关节绝对偏差之和，值越小越好（返回正值，scale为负）
        """
        hip_dof_indices = [0, 3, 6, 9]                          # 4个髋关节在 dof_pos 中的索引
        hip_pos = self.dof_pos[:, hip_dof_indices]               # shape: (num_envs, 4)，4个髋关节当前角度
        default_hip_pos = self.default_dof_pos[:, hip_dof_indices]  # shape: (num_envs, 4)，默认角度（来自 BoyingCfg）
        return torch.sum(torch.abs(hip_pos - default_hip_pos), dim=1)  # 对4个关节求绝对偏差之和，shape: (num_envs,)

    def _reward_x_command_hip_regular(self):
        """
        前进指令下的髋关节对称性奖励（惩罚项）。

        当机器人接收到前进/后退（x方向）速度指令时，惩罚左右前髋、左右后髋不对称的情况。
        目的：防止机器人在直线行走时髋关节左右偏斜，改善步态美观性和实机部署稳定性。

        对称性检测逻辑：
          - 前腿对：FL_hip(idx=0) + FR_hip(idx=1)。正常直行时两者符号相反、绝对值相近，
            其和接近 0；若差异大则和偏离 0，产生惩罚。
          - 后腿对：RL_hip(idx=2) + RR_hip(idx=3)，同理。

        x_command_ratio 调制：
          仅在 x 指令分量占总指令比例较大时（接近纯前进/后退）才施加惩罚，
          纯旋转或横移时比例接近 0，惩罚自动减弱，避免干扰侧移和转向行为。

        注：该奖励函数在 boying_config.py 的 scales 中未启用（未列出），
        实际训练中不生效，保留供后续实验使用。

        Returns:
            (num_envs,) 张量，不对称程度 × x指令比例，值越小越好（scale为负时惩罚）
        """
        hip_dof_indices = [0, 3, 6, 9]                          # 4个髋关节索引：FL/FR/RL/RR
        hip_pos = self.dof_pos[:, hip_dof_indices]               # shape: (num_envs, 4)

        # x方向指令占总指令的比例，用于调制惩罚强度
        # commands[:,0] = lin_vel_x；commands[:,:3] 包含 x/y 线速度 + yaw 角速度
        x_command_ratio = torch.abs(self.commands[:, 0]) / torch.norm(self.commands[:, :3], dim=1)
        # shape: (num_envs,)，值域 [0, 1]，纯x指令时为 1，纯旋转时趋近 0

        # 前腿对称性：|FL_hip + FR_hip|；后腿对称性：|RL_hip + RR_hip|
        # hip_pos[:,0]=FL, hip_pos[:,1]=FR（注意：这里用的是4个髋关节列，idx依次为 FL/FR/RL/RR）
        rew = torch.abs(hip_pos[:, 0] + hip_pos[:, 1]) + torch.abs(hip_pos[:, 2] + hip_pos[:, 3])
        # shape: (num_envs,)，两对髋关节不对称度之和

        return rew * x_command_ratio  # 乘以 x 指令比例，纯前进时全力惩罚，旋转/侧移时自动减弱
