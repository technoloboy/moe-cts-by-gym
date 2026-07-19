from itertools import product  # 用于生成速度限制组合的笛卡尔积 (limit_vel_comb)
from legged_gym import LEGGED_GYM_ROOT_DIR, envs  # 导入工程根目录路径和环境注册表
import time  # 用于测试模式下的实时仿真同步睡眠
from warnings import WarningMessage  # 警告消息类 (当前未直接使用但保留导入)
import numpy as np  # 数值计算库, 用于地形创建、角度计算等
import os  # 操作系统接口, 用于解析URDF资产路径

from isaacgym.torch_utils import *  # 导入IsaacGym的torch工具函数(quat_rotate_inverse等)
from isaacgym import gymtorch, gymapi, gymutil  # IsaacGym核心API: Tensor接口/仿真API/工具函数

import torch  # PyTorch深度学习框架, RL训练的核心计算库
from torch import Tensor  # Tensor类型注解, 提升代码可读性
from typing import Tuple, Dict  # 类型注解工具, 用于函数签名

from legged_gym import LEGGED_GYM_ROOT_DIR  # 再次导入根目录 (用于asset路径格式化)
from legged_gym.envs.base.base_task import BaseTask  # 父类: 提供gym/sim/viewer基础设施
from legged_gym.utils.math import wrap_to_pi, quat_apply_yaw  # 数学工具: 角度归一化/偏航旋转
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor  # 四元数转欧拉角(tensor版)
from legged_gym.utils.isaacgym_utils import sample_disjoint_intervals, sample_single_interval  # 指令采样工具函数
from legged_gym.utils.helpers import class_to_dict  # 将配置类转为字典, 方便奖励缩放初始化
from .legged_robot_config import LeggedRobotCfg  # 四足机器人的配置数据类
from legged_gym.utils.terrain import Terrain  # 地形生成类: heightfield/trimesh

# LeggedRobot继承自BaseTask, 是四足机器人IsaacGym训练环境的核心类
# 负责仿真循环、奖励计算、观测生成、域随机化和地形课程
class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg  # 保存环境配置对象(LeggedRobotCfg), 包含所有超参数
        self.sim_params = sim_params  # 保存物理仿真参数(时间步长、重力等)
        self.height_samples = None  # 地形高度采样数组, _create_heightfield/_create_trimesh时填充
        self.debug_viz = False  # 调试可视化开关, 控制是否渲染调试信息
        self.init_done = False  # 初始化完成标志, 防止_update_terrain_curriculum在初始化时误触发
        self._parse_cfg(self.cfg)  # 解析配置: 计算dt、episode长度、奖励缩放等派生参数
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)  # 调用父类创建gym/sim/viewer

        if not self.headless:  # 如果需要渲染(非无头模式)
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)  # 设置观察相机位置和朝向
        self._init_buffers()  # 初始化所有PyTorch张量缓冲区(状态、动作、奖励等)
        self._prepare_reward_function()  # 准备奖励函数列表, 过滤零权重项
        self.init_done = True  # 标记初始化完成, 允许地形课程更新

        self.reward_curriculum_scales = {}  # 奖励课程缩放字典: {奖励名称: 当前缩放值}
        self.reward_curriculum_configs = []  # 奖励课程配置列表, 控制训练过程中奖励权重的变化
        if hasattr(self.cfg.rewards, "curriculum_rewards") and self.cfg.rewards.curriculum_rewards is not None:  # 检查配置中是否定义了奖励课程
            self.reward_curriculum_configs = self.cfg.rewards.curriculum_rewards  # 加载课程配置列表
            for config in self.reward_curriculum_configs:  # 遍历每个奖励课程配置
                self.reward_curriculum_scales[config['reward_name']] = config['start_value']  # 初始化为起始值
        self.num_steps_per_env = 24  # PPO default num_steps_per_env  # PPO每个环境的默认步数, 用于计算训练迭代次数

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions  # 动作裁剪范围 (通常为100或1.0)
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)  # 裁剪并移到GPU, 防止过大动作破坏仿真
        # step physics and render each frame  # 执行物理仿真步骤并渲染每一帧
        self.render()  # 渲染当前帧 (无头模式下为空操作)
        if self.cfg.domain_rand.randomize_action_delay:  # 如果启用动作延迟随机化(模拟真实机器人通信延迟)
            actions_start_decimation = torch.randint(0, self.cfg.control.decimation+1, (self.num_envs, 1), device=self.device)  # 为每个环境随机采样延迟步数(0到decimation之间)
        for i in range(self.cfg.control.decimation):  # 每个控制步内执行decimation个仿真步(控制频率降采样)
            if self.cfg.domain_rand.randomize_action_delay:  # 如果启用延迟随机化
                use_actions = (i >= actions_start_decimation).float()  # 当前子步是否已超过延迟阈值, 决定是否使用新动作
                input_actions = (1 - use_actions) * self.last_actions + use_actions * self.actions  # 延迟期间使用上一步动作, 之后切换到当前动作
            else:  # 不使用延迟随机化
                input_actions = self.actions  # 直接使用当前动作
            self.torques = self._compute_torques(input_actions).view(self.torques.shape)  # PD控制器计算力矩, 并reshape到正确形状
            if self.cfg.domain_rand.randomize_motor_strength:  # 如果启用电机强度随机化(模拟电机老化/差异)
                self.torques *= self.motor_strengths  # 按随机电机强度系数缩放力矩(每个关节独立)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))  # 将力矩张量写入IsaacGym仿真
            self.gym.simulate(self.sim)  # 推进物理仿真一步(PhysX计算碰撞/接触/动力学)
            if self.cfg.env.test:  # 如果是测试模式(非训练), 需要实时仿真速度
                elapsed_time = self.gym.get_elapsed_time(self.sim)  # 获取实际经过的墙钟时间
                sim_time = self.gym.get_sim_time(self.sim)  # 获取仿真时间
                if sim_time-elapsed_time>0:  # 如果仿真超前于现实时间
                    time.sleep(sim_time-elapsed_time)  # 睡眠等待, 保持实时1:1比例

            if self.device == 'cpu':  # CPU模式下需要显式拉取仿真结果
                self.gym.fetch_results(self.sim, True)  # 等待并获取PhysX计算结果
            self.gym.refresh_dof_state_tensor(self.sim)  # 刷新关节位置/速度张量(dof_state)
        self.post_physics_step()  # 执行物理步骤后处理: 计算观测/奖励/重置等

        # return clipped obs, clipped states (None), rewards, dones and infos  # 返回裁剪后的观测、奖励、终止标志和额外信息
        clip_obs = self.cfg.normalization.clip_observations  # 观测裁剪范围
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)  # 裁剪观测, 防止数值异常传入策略网络
        if self.privileged_obs_buf is not None:  # 如果存在特权观测(教师网络输入)
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)  # 同样裁剪特权观测
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras  # 返回(观测, 特权观测, 奖励, 重置标志, 额外信息)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)  # 刷新根状态张量(位置/方向/速度)
        self.gym.refresh_net_contact_force_tensor(self.sim)  # 刷新接触力张量(用于碰撞检测和奖励计算)
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # 刷新刚体状态张量(脚部位置/速度)

        self.episode_length_buf += 1  # 每个环境的当前episode步数计数器加1
        self.common_step_counter += 1  # 全局训练步数计数器加1(用于课程学习和奖励课程)
        self.commands_resampling_step -= 1  # 指令重采样倒计时减1
        if self.cfg.init_state.turn_over:  # 如果启用翻转恢复训练模式
            self.turn_over_timer = (self.turn_over_timer - self.dt).clip(min=0.0)  # 翻转后零指令计时器倒计时(确保机器人有时间站立)

        self.update_reward_curriculum()  # 根据当前训练进度更新奖励课程缩放值
        # prepare quantities  # 计算基础运动学量
        self.base_pos[:] = self.root_states[:, 0:3]  # 提取机身位置 (x, y, z)
        self.base_quat[:] = self.root_states[:, 3:7]  # 提取机身四元数方向 (qx, qy, qz, qw)
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])  # 将四元数转换为欧拉角 (roll, pitch, yaw)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])  # 将世界系线速度转换到机身坐标系
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])  # 将世界系角速度转换到机身坐标系
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)  # 将重力向量投影到机身坐标系(用于方向奖励)
        self.max_move_distance = self.max_move_distance.maximum(torch.norm(self.root_states[:, :2] - self.env_origins[:, :2], dim=1))  # 更新每个环境中机器人离起点的最大水平位移(用于地形课程)

        self._post_physics_step_callback()  # 子类回调: 计算高度图等感知输入

        # compute observations, rewards, resets, ...  # 计算观测值、奖励和重置条件
        self.check_termination()  # 检查碰撞/超时等终止条件, 更新reset_buf
        self.compute_reward()  # 计算所有奖励项之和, 更新rew_buf
        # resample commands must after reward computing  # 指令重采样必须在奖励计算之后(否则新指令影响本步奖励)
        resampling_env_ids = ((self.commands_resampling_step <= 0.0) * (self.episode_length_buf < self.max_episode_length - 1)).nonzero(as_tuple=False).flatten()  # 找到需要重采样指令的环境(计时到期且episode未结束)
        self._resample_commands(resampling_env_ids)  # 为指定环境重新采样速度指令
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()  # 找到需要重置的环境id
        self.reset_idx(env_ids)  # 重置指定环境(域随机化+状态初始化)

        if self.cfg.domain_rand.push_robots:  # 如果启用随机推力扰动
            self._push_robots()  # 随机给机器人施加冲击速度(模拟外部干扰)

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)  # 计算观测向量(某些观测需要在重置后重新计算)

        self.last_last_actions[:] = self.last_actions[:]  # 保存前前步动作(用于action_smoothness奖励)
        self.last_actions[:] = self.actions[:]  # 更新上一步动作缓冲区(用于动作平滑度奖励和延迟随机化)
        self.last_dof_vel[:] = self.dof_vel[:]  # 保存上一步关节速度(用于加速度惩罚奖励)
        self.last_root_vel[:] = self.root_states[:, 7:13]  # 保存上一步根状态速度(线速度+角速度)
    
    def update_reward_curriculum(self, force_update: bool = False):
        # update reward curriculum  # 奖励课程更新函数: 随训练进度动态调整奖励权重(例如逐渐减小某项惩罚)
        if self.reward_curriculum_configs:  # 如果存在奖励课程配置
            if self.common_step_counter % self.num_steps_per_env == 0 or force_update:  # 每个PPO迭代或强制更新时才计算(避免过于频繁)

                for config in self.reward_curriculum_configs:  # 遍历所有奖励课程配置项
                    current_scale = self.get_current_scale(config)  # 根据当前训练进度计算该奖励的缩放值
                    reward_name = config['reward_name']  # 获取奖励函数名称
                    self.reward_curriculum_scales[reward_name] = current_scale  # 更新该奖励的当前缩放值

    def get_current_scale(self, config):
        """ config: Dict
            {'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0}
        """
        current_iter = self.common_step_counter // self.num_steps_per_env  # 当前PPO迭代次数(将步数转换为迭代数)
        cfg_start_iter = config['start_iter']  # 课程开始迭代次数
        cfg_end_iter = config['end_iter']  # 课程结束迭代次数
        cfg_start_val = config['start_value']  # 课程起始缩放值
        cfg_end_val = config['end_value']  # 课程结束缩放值

        percentage = (current_iter - cfg_start_iter) / (cfg_end_iter - cfg_start_iter)  # 计算当前进度百分比(0~1之间)
        percentage = max(min(percentage, 1.0), 0.0)  # 将进度限制在[0,1]范围内(防止越界)

        current_scale = (1.0 - percentage) * cfg_start_val + percentage * cfg_end_val  # 线性插值计算当前缩放值
        return current_scale  # 返回当前奖励的课程缩放系数

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 初始化重置缓冲区为全False(无环境需重置)
        if not self.cfg.init_state.turn_over:  # 如果不是翻转恢复训练模式
            self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)  # 检测躯干/大腿等终止接触体是否受到>1N的接触力(摔倒检测)
        # self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)  # 备用: 通过欧拉角检测倾倒(当前注释掉)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs  # 检测episode是否超时(超时不给终止惩罚)
        self.reset_buf |= self.time_out_buf  # 超时也触发重置(OR操作合并超时和碰撞终止)

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids),
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:  # 如果没有环境需要重置, 直接返回
            return

        ### Domain randomizations ###  # 域随机化: 每次重置时随机化机器人参数, 提高sim-to-real泛化
        # randomization of the motor strength  # 随机化电机强度(模拟不同电机性能/磨损程度)
        if self.cfg.domain_rand.randomize_motor_strength:  # 如果启用电机强度随机化
            rng = self.cfg.domain_rand.motor_strength_range  # 获取电机强度范围[min, max]
            self.motor_strengths[env_ids] = torch_rand_float(
                rng[0], rng[1], (len(env_ids), self.num_actions), device=self.device
            )  # 为每个关节独立采样强度系数
        # randomization of the motor zero calibration for real machine  # 随机化电机零位偏置(模拟真实机器人的关节零位误差)
        if self.cfg.domain_rand.randomize_motor_zero_offset:  # 如果启用电机零位偏置随机化
            self.motor_zero_offsets[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_zero_offset_range[0], self.cfg.domain_rand.motor_zero_offset_range[1], (len(env_ids), self.num_actions), device=self.device)  # 为每个关节采样随机零位偏置
        # randomization of the motor pd gains  # 随机化PD增益(模拟控制参数不确定性)
        if self.cfg.domain_rand.randomize_pd_gains:  # 如果启用PD增益随机化
            self.p_gains_multiplier[env_ids] = torch_rand_float(self.cfg.domain_rand.stiffness_multiplier_range[0], self.cfg.domain_rand.stiffness_multiplier_range[1], (len(env_ids), self.num_actions), device=self.device)  # 随机化Kp增益乘数
            self.d_gains_multiplier[env_ids] =  torch_rand_float(self.cfg.domain_rand.damping_multiplier_range[0], self.cfg.domain_rand.damping_multiplier_range[1], (len(env_ids), self.num_actions), device=self.device)  # 随机化Kd增益乘数

        # update terrain curriculum before reset root states  # 在重置根状态前先更新地形课程(这样env_origins已更新)
        if self.cfg.terrain.curriculum:  # 如果启用地形课程
            self._update_terrain_curriculum(env_ids)  # 根据机器人表现升/降地形难度

        # reset robot states  # 重置机器人物理状态
        self._reset_dofs(env_ids)  # 重置关节角度和速度到初始值附近
        self._reset_root_states(env_ids)  # 重置机身位置/方向/速度(包含翻转初始化)

        # reset buffers  # 重置各缓冲区到初始状态
        self.actions[env_ids] = 0.  # 清零动作缓冲区
        self.last_actions[env_ids] = 0.  # 清零上一步动作缓冲区
        self.last_last_actions[env_ids] = 0.  # 清零前前步动作缓冲区
        self.last_dof_vel[env_ids] = 0.  # 清零上一步关节速度
        self.feet_air_time[env_ids] = 0.  # 清零脚部离地计时器
        self.episode_length_buf[env_ids] = 0  # 重置episode步数计数器
        self.reset_buf[env_ids] = 1  # 标记这些环境已完成重置
        self.commands_resampling_step[env_ids] = self.cfg.commands.resampling_time / self.dt  # 重置指令重采样倒计时
        self.commands_xy_accumulation[env_ids] = 0.0  # 清零xy方向指令累积量(用于动态指令重采样)
        self._resample_commands(env_ids)  # 为重置后的环境重新采样速度指令
        # fill extras  # 填充日志信息
        self.extras["episode"] = {}  # 初始化episode信息字典
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:  # 如果使用复杂地形
            self.extras["episode"]['terrain_level_all'] = torch.mean(self.terrain_levels.float())  # 记录所有环境的平均地形难度
            for name, cols in self.terrain.name2cols.items():  # 遍历每种地形类型
                if isinstance(cols, set):  # 如果cols是集合类型
                    cols = self.terrain.name2cols[name] = torch.tensor(list(cols), device=self.device)  # 转换为tensor并缓存
                self.extras["episode"]['terrain_level_' + name] = torch.mean(self.terrain_levels[torch.isin(self.terrain_types, cols)].float())  # 记录该类型地形的平均难度
        else:  # 平地训练
            self.extras["episode"]['terrain_level_all'] = 0.0  # 平地无地形难度级别
        for key in self.episode_sums.keys():  # 遍历所有奖励项
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s  # 计算每秒平均奖励(便于不同时长episode对比)
            self.episode_sums[key][env_ids] = 0.  # 重置episode奖励累积
        self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]  # 记录当前最大x方向速度指令(跟踪课程进度)
        # send timeout info to the algorithm  # 将超时信息传递给算法(用于正确处理truncated episodes)
        if self.cfg.env.send_timeouts:  # 如果需要向算法发送超时信息
            self.extras["time_outs"] = self.time_out_buf  # 发送超时标志(PPO需要区分done和truncated)
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.  # 将奖励缓冲区清零, 准备累加各项奖励
        for i in range(len(self.reward_functions)):  # 遍历所有非零权重的奖励函数
            name = self.reward_names[i]  # 获取当前奖励函数的名称
            raw_rew = self.reward_functions[i]()  # 调用奖励函数计算原始奖励值(未乘以scale)
            rew = raw_rew * self.reward_scales.get(name, 0.0)  # 乘以配置的scale(已乘以dt使奖励与时间步无关)
            if self.cfg.init_state.turn_over:  # 如果启用翻转恢复模式
                turn_over_rew = raw_rew * self.reward_turn_over_scales.get(name, 0.0)  # 计算翻转模式下的奖励(使用不同的scale)
            if name in self.reward_curriculum_scales:  # 如果该奖励有课程缩放
                rew *= self.reward_curriculum_scales[name]  # 乘以当前课程缩放系数(随训练进度变化)
                if self.cfg.init_state.turn_over:  # 翻转模式下也应用课程缩放
                    turn_over_rew *= self.reward_curriculum_scales[name]  # 翻转奖励乘以课程缩放
            if self.cfg.init_state.turn_over:  # 翻转模式: 根据是否需要翻转选择不同奖励
                need_turn_over = self.rpy[:, 0].abs() > self.cfg.rewards.turn_over_roll_threshold  # 判断当前roll角是否超过阈值(需要翻转)
                rew = torch.where(need_turn_over, turn_over_rew, rew)  # 翻转状态下使用翻转奖励, 否则使用正常奖励
            self.rew_buf += rew  # 累加到总奖励缓冲区
            self.episode_sums[name] += rew  # 累加到episode统计(用于日志记录)
        if self.cfg.rewards.only_positive_rewards:  # 如果配置为只使用正奖励
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)  # 裁剪掉负奖励(适合早期训练)
        # add termination reward after clipping  # 终止奖励在裁剪后单独添加
        if "termination" in self.reward_scales:  # 如果配置了终止奖励(通常为负值作为惩罚)
            rew = self._reward_termination() * self.reward_scales["termination"]  # 计算终止惩罚(仅在非超时终止时触发)
            self.rew_buf += rew  # 将终止惩罚加入总奖励
            self.episode_sums["termination"] += rew  # 记录终止惩罚累计值
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,  # 机身线速度(机体系) * 缩放因子, 共3维
                                    self.base_ang_vel  * self.obs_scales.ang_vel,  # 机身角速度(机体系) * 缩放因子, 共3维
                                    self.projected_gravity,  # 重力向量在机体系的投影, 共3维(表征机体倾斜)
                                    self.commands[:, :3] * self.commands_scale,  # 速度指令[vx,vy,wz] * 缩放, 共3维
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 关节位置偏差(相对默认姿态) * 缩放, 共12维
                                    self.dof_vel * self.obs_scales.dof_vel,  # 关节速度 * 缩放因子, 共12维
                                    self.actions  # 上一步动作(闭环观测), 共12维
                                    ),dim=-1)  # 拼接成观测向量: 3+3+3+3+12+12+12=48维
        # add perceptive inputs if not blind  # 如果有感知输入(高度图等)则添加
        # add noise if needed  # 如果需要则添加观测噪声
        if self.add_noise:  # 如果启用观测噪声(训练时增强鲁棒性)
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec  # 添加均匀噪声[-1,1]*noise_scale

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly  # 设置上轴索引: 2表示Z轴朝上(IsaacGym标准)
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)  # 创建IsaacGym仿真实例(GPU设备、渲染设备、PhysX引擎、仿真参数)

        mesh_type = self.cfg.terrain.mesh_type  # 获取地形网格类型配置(plane/heightfield/trimesh/None)
        if mesh_type in ['heightfield', 'trimesh']:  # 如果使用复杂地形
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)  # 创建地形对象(生成地形网格数据)
        if mesh_type=='plane':  # 平面地形
            self._create_ground_plane()  # 创建简单平面地形(最快, 用于调试)
        elif mesh_type=='heightfield':  # 高度场地形
            self._create_heightfield()  # 创建高度场地形(基于网格高度数据)
        elif mesh_type=='trimesh':  # 三角网格地形
            self._create_trimesh()  # 创建三角网格地形(最精确, 支持复杂地形)
        elif mesh_type is not None:  # 未知地形类型
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")  # 抛出错误提示合法类型

        self._create_envs()  # 创建所有并行仿真环境(加载URDF、设置属性、实例化actor)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])  # 将相机位置列表转为gymapi.Vec3格式
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])  # 将相机朝向目标点转为gymapi.Vec3格式
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)  # 设置查看器相机的位置和注视点

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:  # 如果启用摩擦系数随机化(关键sim-to-real域随机化参数)
            if env_id==0:  # 只在第一个环境时初始化摩擦桶(所有环境共享同一批随机值)
                # prepare friction randomization  # 准备摩擦系数随机化
                friction_range = self.cfg.domain_rand.friction_range  # 获取摩擦系数范围[min, max]
                num_buckets = 64  # 使用64个桶离散化摩擦分布(避免每个env完全独立导致内存浪费)
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))  # 为每个环境随机分配一个桶
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')  # 生成64个随机摩擦系数
                self.friction_coeffs = friction_buckets[bucket_ids]  # 为每个环境查表获取摩擦系数

            for s in range(len(props)):  # 遍历该机器人的所有碰撞形状
                props[s].friction = self.friction_coeffs[env_id]  # 将该环境的摩擦系数应用到所有形状

        if self.cfg.domain_rand.randomize_restitution:  # 如果启用弹性系数随机化(模拟不同地面弹性)
            rand_restitution = np.random.uniform(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1])  # 随机采样弹性系数
            for s in range(len(props)):  # 遍历所有碰撞形状
                props[s].restitution = rand_restitution  # 应用弹性系数
        return props  # 返回修改后的刚体形状属性

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:  # 只在第一个环境时初始化关节限制(所有环境使用相同的URDF定义限制)
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)  # 初始化关节位置限制张量: shape=(num_dof, 2) [下限, 上限]
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)  # 初始化关节速度限制张量
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)  # 初始化力矩限制张量
            for i in range(len(props)):  # 遍历所有DOF
                self.dof_pos_limits[i, 0] = props["lower"][i].item()  # 读取URDF定义的关节角度下限
                self.dof_pos_limits[i, 1] = props["upper"][i].item()  # 读取URDF定义的关节角度上限
                self.dof_vel_limits[i] = props["velocity"][i].item()  # 读取URDF定义的关节速度限制
                self.torque_limits[i] = props["effort"][i].item()  # 读取URDF定义的力矩限制
                # soft limits  # 软限制: 使实际使用范围略小于硬限制, 避免关节卡到物理边界
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2  # 计算关节角度范围的中点
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]  # 计算关节角度范围的宽度
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit  # 软下限: 中点减去(soft_factor*范围/2)
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit  # 软上限: 中点加上(soft_factor*范围/2)

        return props  # 返回DOF属性(本函数主要作用是提取限制, 不修改属性)

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:  # 调试代码: 打印每个刚体的质量(已注释)
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass  # 随机化机身质量(模拟携带不同载荷)
        if self.cfg.domain_rand.randomize_base_mass:  # 如果启用机身质量随机化
            rng = self.cfg.domain_rand.added_mass_range  # 获取附加质量范围(如[-1, 3]kg)
            props[0].mass += np.random.uniform(rng[0], rng[1])  # 给机身(index=0)添加随机质量偏置

        # randomize link masses  # 随机化连杆质量(模拟URDF质量参数不确定性)
        if self.cfg.domain_rand.randomize_link_mass:  # 如果启用连杆质量随机化
            self.multiplied_link_masses_ratio = torch_rand_float(self.cfg.domain_rand.multiplied_link_mass_range[0], self.cfg.domain_rand.multiplied_link_mass_range[1], (1, self.num_bodies-1), device=self.device)  # 为每个连杆(除机身外)采样质量乘数
            for i in range(1, len(props)):  # 遍历所有连杆(跳过机身index=0)
                props[i].mass *= self.multiplied_link_masses_ratio[0,i-1]  # 按随机乘数缩放连杆质量

        # randomize base com  # 随机化机身质心位置(模拟实际机器人重心偏移)
        if self.cfg.domain_rand.randomize_base_com:  # 如果启用质心偏移随机化
            self.added_base_com = torch_rand_float(self.cfg.domain_rand.added_base_com_range[0], self.cfg.domain_rand.added_base_com_range[1], (1, 3), device=self.device)  # 随机采样质心偏移量(x,y,z方向)
            props[0].com += gymapi.Vec3(self.added_base_com[0, 0], self.added_base_com[0, 1],
                                    self.added_base_com[0, 2])  # 将随机质心偏移叠加到机身质心位置
        return props  # 返回修改后的刚体属性
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute measured terrain heights and randomly push robots
        """
        if self.cfg.terrain.measure_heights:  # 如果需要测量地形高度(感知输入/高度估计)
            self.measured_heights = self._get_heights()  # 采样机器人周围地形高度点

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:  # 如果没有环境需要重采样指令, 直接返回
            return
        self.stop_heading[env_ids] = False  # 重置停止朝向标志(允许根据目标朝向控制偏航)
        # update command curriculum with train steps  # 根据训练步数更新指令范围课程
        if len(self.cfg.commands.command_range_curriculum):  # 如果存在指令范围课程配置
            current_iter = self.common_step_counter // self.num_steps_per_env  # 当前PPO迭代次数
            for i in range(len(self.cfg.commands.command_range_curriculum)-1, -1, -1):  # iterate backwards to be able to pop entries  # 倒序遍历以便安全删除已触发的课程条目
                cfg = self.cfg.commands.command_range_curriculum[i]  # 获取当前课程配置
                if current_iter >= cfg["iter"]:  # 如果已达到该课程的触发迭代次数
                    self.command_ranges["lin_vel_x"] = cfg["lin_vel_x"]  # 更新x方向速度指令范围
                    self.command_ranges["lin_vel_y"] = cfg["lin_vel_y"]  # 更新y方向速度指令范围
                    self.command_ranges["ang_vel_yaw"] = cfg["ang_vel_yaw"]  # 更新偏航角速度指令范围
                    self.command_ranges["heading"] = cfg["heading"]  # 更新朝向指令范围
                    self.max_lin_vel = max(abs(self.command_ranges["lin_vel_x"][0]), abs(self.command_ranges["lin_vel_x"][1]),
                                           abs(self.command_ranges["lin_vel_y"][0]), abs(self.command_ranges["lin_vel_y"][1]))  # 更新最大线速度(用于动态指令重采样)
                    self.cfg.commands.command_range_curriculum.pop(i)  # 移除已触发的课程条目(避免重复触发)
                    self._update_env_command_ranges()  # 更新每个环境的具体指令范围(考虑地形类型限制)
                    print(f"Command range updated at iter {current_iter}: {self.command_ranges}")  # 打印课程更新日志
        remaining_dist = torch.clip(0.625 * self.cfg.terrain.terrain_length - torch.norm(self.commands_xy_accumulation[env_ids], dim=1) * self.cfg.commands.resampling_time, 0.0)  # 计算剩余地形距离(0.625倍地形长度减去已累积的指令位移)
        self.commands_resampling_step[env_ids] = self.cfg.commands.resampling_time / self.dt  # 重置指令重采样倒计时(单位:步数)
        if self.cfg.commands.dynamic_resample_commands:  # 如果启用动态指令重采样(考虑剩余地形距离)
            # arrive at boundary 0.625 times the width of the remaining distance  # 目标: 在剩余距离内以0.625倍宽度到达边界
            if ((self.max_episode_length - self.episode_length_buf[env_ids]) == 0).any():  # 检查是否有环境剩余步数为0(防止除零)
                raise ValueError("Some envs have zero remaining episode length during command resampling")  # 抛出错误
            vel_low_bound = torch.clip(remaining_dist / ((self.max_episode_length - self.episode_length_buf[env_ids] + 1e-9) * self.dt), 0.0)  # 计算速度下界: 确保能在剩余时间内走完剩余距离
            self.commands[env_ids, 0] = sample_disjoint_intervals(  # 采样x方向速度指令(确保与零速区间不重叠)
                env_ids,  # 目标环境ID
                vel_low_bound,  # 速度最小绝对值(保证机器人能走到地形边界)
                self.env_command_ranges["lin_vel_x"][env_ids, 0],  # 速度范围下限
                self.env_command_ranges["lin_vel_x"][env_ids, 1],  # 速度范围上限
                self.device  # 计算设备
            )
            self.commands[env_ids, 1] = sample_disjoint_intervals(  # 采样y方向速度指令
                env_ids,  # 目标环境ID
                vel_low_bound,  # 速度最小绝对值
                self.env_command_ranges["lin_vel_y"][env_ids, 0],  # 速度范围下限
                self.env_command_ranges["lin_vel_y"][env_ids, 1],  # 速度范围上限
                self.device  # 计算设备
            )
            if self.cfg.commands.heading_command:  # 如果使用目标朝向控制模式
                r = torch.rand(len(env_ids), device=self.device)  # 在[0,1]均匀采样随机数
                lower = self.env_command_ranges["heading"][env_ids, 0]  # 朝向指令下限
                upper = self.env_command_ranges["heading"][env_ids, 1]  # 朝向指令上限
                self.commands[env_ids, 3] = (upper - lower) * r + lower  # 均匀采样目标朝向角
            else:  # 使用直接角速度控制模式
                r = torch.rand(len(env_ids), device=self.device)  # 均匀采样随机数
                lower = self.env_command_ranges["ang_vel_yaw"][env_ids, 0]  # 偏航角速度下限
                upper = self.env_command_ranges["ang_vel_yaw"][env_ids, 1]  # 偏航角速度上限
                self.commands[env_ids, 2] = (upper - lower) * r + lower  # 均匀采样偏航角速度指令
        else:  # 静态指令重采样(标准均匀采样)
            self.commands[env_ids, 0] = sample_single_interval(  # 采样x方向速度指令(单区间均匀分布)
                env_ids,  # 目标环境ID
                self.env_command_ranges["lin_vel_x"][env_ids, 0],  # 速度范围下限
                self.env_command_ranges["lin_vel_x"][env_ids, 1],  # 速度范围上限
                self.device  # 计算设备
            )
            self.commands[env_ids, 1] = sample_single_interval(  # 采样y方向速度指令
                env_ids,  # 目标环境ID
                self.env_command_ranges["lin_vel_y"][env_ids, 0],  # 速度范围下限
                self.env_command_ranges["lin_vel_y"][env_ids, 1],  # 速度范围上限
                self.device  # 计算设备
            )
            if self.cfg.commands.heading_command:  # 如果使用目标朝向控制
                self.commands[env_ids, 3] = sample_single_interval(  # 采样目标朝向角
                    env_ids,  # 目标环境ID
                    self.env_command_ranges["heading"][env_ids, 0],  # 朝向下限
                    self.env_command_ranges["heading"][env_ids, 1],  # 朝向上限
                    self.device  # 计算设备
                )
            else:  # 直接角速度控制
                self.commands[env_ids, 2] = sample_single_interval(  # 采样偏航角速度
                    env_ids,  # 目标环境ID
                    self.env_command_ranges["ang_vel_yaw"][env_ids, 0],  # 角速度下限
                    self.env_command_ranges["ang_vel_yaw"][env_ids, 1],  # 角速度上限
                    self.device  # 计算设备
                )

            # set small commands to zero  # 将过小的速度指令置零(避免机器人微动)
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)  # xy速度小于0.2m/s的置零

        rand_prob = torch.rand(len(env_ids), device=self.device)  # 为每个重置环境采样一个均匀随机数, 用于选择特殊指令模式
        min_prob, max_prob = 0.0, 0.0  # 初始化概率区间边界(用于分段概率采样)
        # set limitation lin vel  # 设置极限速度指令(让机器人学习边界速度行为)
        if self.limit_vel_prob > 0.0:  # 如果极限速度概率大于零
            max_prob += self.limit_vel_prob  # 更新概率区间上界
            lim_mask = (rand_prob >= min_prob) * (rand_prob < max_prob)  # 找到落在极限速度概率区间的环境
            lim_env_ids = env_ids[lim_mask]  # 提取需要使用极限速度指令的环境ID
            if len(lim_env_ids) > 0:  # 如果有环境需要极限速度指令
                change_lim_env_ids = lim_env_ids  # 默认所有极限速度环境都要更新速度
                if self.cfg.commands.limit_vel_invert_when_continuous:  # 如果启用连续极限速度反转(让机器人学习来回走)
                    was_limited = self.last_is_limit_vel[lim_env_ids]  # 检查上次是否也是极限速度模式
                    invert_env_ids = lim_env_ids[was_limited]  # 连续极限速度的环境需要反转方向
                    self.commands[invert_env_ids, 0] *= -1.0  # 反转x方向速度
                    self.commands[invert_env_ids, 1] *= -1.0  # 反转y方向速度
                    self.commands[invert_env_ids, 2] *= -1.0  # 反转偏航角速度
                    change_lim_env_ids = lim_env_ids[~was_limited]  # 只有新进入极限速度模式的环境才重新采样
                vel_idx = torch.randint(0, self.limit_vel_comb.shape[0], (len(change_lim_env_ids),), device=self.device)  # 随机选择一种速度组合(笛卡尔积中的一种)
                lin_vel_x_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 0] == -1,  # 如果x方向为负向极限
                    self.env_command_ranges["lin_vel_x"][change_lim_env_ids, 0],  # 使用x速度下限(最大负速)
                    self.env_command_ranges["lin_vel_x"][change_lim_env_ids, 1],  # 使用x速度上限(最大正速)
                )
                lin_vel_x_lim[self.limit_vel_comb[vel_idx, 0] == 0] = 0.0  # x方向为零时将速度设为0
                lin_vel_y_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 1] == -1,  # 如果y方向为负向极限
                    self.env_command_ranges["lin_vel_y"][change_lim_env_ids, 0],  # 使用y速度下限
                    self.env_command_ranges["lin_vel_y"][change_lim_env_ids, 1]  # 使用y速度上限
                )
                lin_vel_y_lim[self.limit_vel_comb[vel_idx, 1] == 0] = 0.0  # y方向为零时将速度设为0
                ang_vel_z_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 2] == -1,  # 如果偏航方向为负向极限
                    self.env_command_ranges["ang_vel_yaw"][change_lim_env_ids, 0],  # 使用偏航角速度下限
                    self.env_command_ranges["ang_vel_yaw"][change_lim_env_ids, 1]  # 使用偏航角速度上限
                )
                ang_vel_z_lim[self.limit_vel_comb[vel_idx, 2] == 0] = 0.0  # 偏航方向为零时将角速度设为0
                self.commands[change_lim_env_ids, 0] = lin_vel_x_lim  # 设置x方向极限速度指令
                self.commands[change_lim_env_ids, 1] = lin_vel_y_lim  # 设置y方向极限速度指令
                self.commands[change_lim_env_ids, 2] = ang_vel_z_lim  # 设置偏航角速度极限指令
                if self.cfg.commands.heading_command and self.cfg.commands.stop_heading_at_limit:  # 如果使用朝向控制且需要在极限速度时停止朝向
                    self.stop_heading[lim_env_ids] = True # stop heading to current heading  # 停止朝向更新(保持当前朝向不再转向)
                self.last_is_limit_vel[env_ids] = False  # 清除所有重置环境的上次极限速度标志
                self.last_is_limit_vel[lim_env_ids] = True  # 标记当前极限速度环境
            else:  # 没有环境需要极限速度指令
                self.last_is_limit_vel[env_ids] = False  # 清除所有环境的极限速度标志
            min_prob += self.limit_vel_prob  # 更新概率区间下界

        # set all commands to zero with some probability  # 以一定概率将所有指令置零(训练机器人站立/原地保持)
        if self.cfg.commands.zero_command_curriculum is not None:  # 如果存在零指令概率课程配置
            self.zero_command_proba = self.get_current_scale(self.cfg.commands.zero_command_curriculum)  # 根据训练进度动态计算零指令概率
        if self.zero_command_proba > 0.0:  # 如果零指令概率大于零
            max_prob += self.zero_command_proba  # 更新概率区间上界
            next_resampling_step = torch.clip(
                self.max_episode_length - self.episode_length_buf[env_ids] - (remaining_dist / (0.8 * self.max_lin_vel * self.dt + 1e-9)),  # 计算下次重采样时机: 剩余步数减去走完剩余距离所需步数
                min=0.0,  # 不能为负数
                max=self.cfg.commands.resampling_time / self.dt,  # 不超过最大重采样间隔
            )  # 零指令持续步数: 让机器人在到达目标区域后原地站立
            zero_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) * (next_resampling_step > 0.0)  # 落在零指令概率区间且有足够剩余时间的环境
            zero_env_ids = env_ids[zero_mask]  # 提取需要零指令的环境ID
            if len(zero_env_ids) > 0:  # 如果有环境需要零指令
                self.commands[zero_env_ids, :2] = 0.0  # 将xy线速度指令置零
                self.commands_resampling_step[zero_env_ids] = next_resampling_step[zero_mask]  # 设置到下次重采样的步数
                if self.cfg.commands.limit_ang_vel_at_zero_command_prob > 0.0:  # 如果需要在零指令时保留角速度指令
                    ang_vel_rand = torch.rand(len(zero_env_ids), device=self.device) # independent distribution  # 独立采样角速度随机数
                    add_ang_mask = ang_vel_rand < self.cfg.commands.limit_ang_vel_at_zero_command_prob  # 部分零指令环境添加角速度(原地旋转)
                    add_ang_env_ids = zero_env_ids[add_ang_mask]  # 需要添加角速度的环境ID
                    if len(add_ang_env_ids) > 0:  # 如果有环境需要角速度
                        direction_rand = torch.rand(len(add_ang_env_ids), device=self.device)  # 随机决定旋转方向
                        self.commands[add_ang_env_ids, 2] = torch.where(
                            direction_rand < 0.5,  # 50%概率正转
                            self.env_command_ranges["ang_vel_yaw"][add_ang_env_ids, 0],  # 负向极限角速度(左转)
                            self.env_command_ranges["ang_vel_yaw"][add_ang_env_ids, 1]  # 正向极限角速度(右转)
                        )  # 设置原地旋转角速度
                        if self.cfg.commands.heading_command:  # 如果使用朝向控制
                            self.stop_heading[add_ang_env_ids] = True  # 停止朝向控制(避免与角速度指令冲突)
            min_prob += self.zero_command_proba  # 更新概率区间下界

        # turn over zero command time  # 翻转恢复期间保持零速度指令
        if self.cfg.init_state.turn_over and (self.turn_over_timer[env_ids] > 0).any():  # 如果启用翻转恢复且有环境仍在翻转计时
            zero_mask = self.turn_over_timer[env_ids] > 0  # 找到仍在翻转计时的环境
            zero_env_ids = env_ids[zero_mask]  # 提取这些环境ID
            self.commands[zero_env_ids, :3] = 0.0  # 将vx,vy,wz指令全部置零(让机器人专注翻身)
            self.stop_heading[zero_env_ids] = True  # 停止朝向控制

        self.commands_xy_accumulation[env_ids] += self.commands[env_ids, :2]  # 累积xy速度指令(用于计算预期行进距离)

        if self.cfg.commands.heading_command:  # 如果使用目标朝向控制模式
            heading_env_ids = env_ids[self.stop_heading[env_ids] == 0.0]  # 找到没有停止朝向控制的环境
            if len(heading_env_ids) > 0:  # 如果有需要朝向控制的环境
                forward = quat_apply(self.base_quat[heading_env_ids], self.forward_vec[heading_env_ids])  # 计算机器人前向方向(世界坐标系)
                heading = torch.atan2(forward[:, 1], forward[:, 0])  # 计算当前朝向角(偏航角)
                self.commands[heading_env_ids, 2] = torch.clip(
                    0.5 * wrap_to_pi(self.commands[heading_env_ids, 3] - heading),  # P控制: 目标朝向误差乘以0.5转为角速度指令
                    self.env_command_ranges["ang_vel_yaw"][heading_env_ids, 0],  # 限制在角速度下限
                    self.env_command_ranges["ang_vel_yaw"][heading_env_ids, 1]  # 限制在角速度上限
            )  # 朝向控制: 将目标朝向转换为角速度指令

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller  # PD控制器: 将位置目标/速度目标转换为关节力矩
        actions_scaled = actions * self.cfg.control.action_scale  # 将归一化动作缩放到实际关节角度偏移量
        control_type = self.cfg.control.control_type  # 获取控制类型(P=位置控制, V=速度控制, T=力矩控制)
        p_gains = self.p_gains * self.p_gains_multiplier  # 实际Kp = 标称Kp * 随机化乘数(域随机化)
        d_gains = self.d_gains * self.d_gains_multiplier  # 实际Kd = 标称Kd * 随机化乘数(域随机化)
        if control_type=="P":  # 位置控制模式(最常用, Go2 Kp=60)
            torques = p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_zero_offsets) - d_gains*self.dof_vel  # 力矩=Kp*(目标位置-当前位置)-Kd*速度
        elif control_type=="V":  # 速度控制模式
            torques = p_gains*(actions_scaled - self.dof_vel) - d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt  # 力矩=Kp*(目标速度-当前速度)-Kd*加速度
        elif control_type=="T":  # 直接力矩控制模式
            torques = actions_scaled  # 直接使用缩放后的动作作为力矩
        else:  # 未知控制类型
            raise NameError(f"Unknown controller type: {control_type}")  # 抛出错误
        return torch.clip(torques, -self.torque_limits, self.torque_limits)  # 裁剪力矩到关节力矩限制范围内

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)  # 随机初始化关节角度: 默认角度的0.5~1.5倍(增加初始状态多样性)
        self.dof_vel[env_ids] = 0.  # 将关节速度初始化为零

        env_ids_int32 = env_ids.to(dtype=torch.int32)  # 将环境ID转换为int32(IsaacGym API要求)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),  # 完整DOF状态张量
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))  # 只更新指定环境的关节状态
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        if self.cfg.init_state.turn_over:  # 如果启用翻转恢复训练
            self.turn_over_timer[env_ids] = 0.0  # 重置翻转计时器为0
        # base position  # 设置机身初始位置和方向
        random_yaw = torch_rand_float(-np.pi, np.pi, (len(env_ids), 1), device=self.device).squeeze(1)  # 随机初始偏航角(全方向均匀分布)
        def get_quat(target_yaws, roll: float):  # 内部函数: 根据偏航角和翻滚角生成四元数
            roll_tensor = torch.full((len(target_yaws),), roll, device=self.device)  # 生成统一翻滚角张量
            pitch_tensor = torch.zeros((len(target_yaws),), device=self.device)  # 俯仰角为零
            quat = quat_from_euler_xyz(roll_tensor, pitch_tensor, target_yaws)  # 从欧拉角生成四元数
            return quat  # 返回四元数

        base_init_state = self.base_init_state.reshape(1, -1).repeat(len(env_ids), 1)  # 复制基础初始状态模板(位置+方向+速度)
        if self.cfg.init_state.turn_over:  # 如果启用翻转初始化(训练翻身恢复)
            rand_prob = torch.rand(len(env_ids), device=self.device)  # 为每个环境采样随机数, 决定初始翻转类型
            proportions = self.cfg.init_state.turn_over_proportions  # 各翻转类型的概率分配[backflip, sideflip, noflip]
            init_heights = self.cfg.init_state.turn_over_init_heights  # 各翻转类型的初始高度范围

            min_prob, max_prob = 0.0, proportions[0]  # 第一区间: 后翻(backflip)
            back_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) # backflip  # 找到需要后翻初始化的环境
            if back_mask.any():  # 如果有后翻环境
                heights = torch_rand_float(init_heights['backflip'][0], init_heights['backflip'][1], (torch.sum(back_mask), 1), device=self.device).squeeze(1)  # 随机采样后翻初始高度
                base_init_state[back_mask, 2] = heights # z  # 设置初始高度(z坐标)
                base_init_state[back_mask, 3:7] = get_quat(random_yaw[back_mask], np.pi)  # 设置初始方向(翻转180度=roll=pi)
                if self.cfg.init_state.turn_over:  # 设置翻转计时器
                    self.turn_over_timer[env_ids[back_mask]] = self.cfg.commands.turn_over_zero_time['backflip']  # 后翻计时器(这段时间内速度指令为零)

            min_prob = max_prob  # 更新区间下界
            max_prob += proportions[1]  # 更新区间上界
            side_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) # sideflip  # 找到需要侧翻初始化的环境
            if side_mask.any():  # 如果有侧翻环境
                side_ids = torch.nonzero(side_mask, as_tuple=False).flatten()  # 获取侧翻环境的局部索引
                heights = torch_rand_float(init_heights['sideflip'][0], init_heights['sideflip'][1], (len(side_ids), 1), device=self.device).squeeze(1)  # 随机采样侧翻初始高度
                base_init_state[side_mask, 2] = heights # z  # 设置初始高度
                side_rand_prob = torch.rand(len(side_ids), device=self.device)  # 随机决定左侧翻还是右侧翻
                pos_side_mask = side_rand_prob < 0.5  # 50%概率正向侧翻
                neg_side_mask = ~pos_side_mask  # 50%概率负向侧翻
                if pos_side_mask.any():  # 正向侧翻(左侧)
                    pos_ids = side_ids[pos_side_mask]  # 获取正向侧翻环境索引
                    base_init_state[pos_ids, 3:7] = get_quat(random_yaw[pos_ids], np.pi/2)  # 设置90度滚转方向
                if neg_side_mask.any():  # 负向侧翻(右侧)
                    neg_ids = side_ids[neg_side_mask]  # 获取负向侧翻环境索引
                    base_init_state[neg_ids, 3:7] = get_quat(random_yaw[neg_ids], -np.pi/2)  # 设置-90度滚转方向
                if self.cfg.init_state.turn_over:  # 设置侧翻计时器
                    self.turn_over_timer[env_ids[side_mask]] = self.cfg.commands.turn_over_zero_time['sideflip']  # 侧翻计时器

            min_prob = max_prob  # 更新区间下界
            max_prob += proportions[2]  # 更新区间上界
            noflip_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) # noflip  # 找到正常初始化(不翻转)的环境
            if noflip_mask.any():  # 如果有正常初始化环境
                noflip_indices = torch.nonzero(noflip_mask, as_tuple=False).flatten()  # 获取正常初始化环境的局部索引
                base_init_state[noflip_mask, 3:7] = get_quat(random_yaw[noflip_indices], 0.0)  # 设置正常站立方向(roll=0)
        else:  # 不启用翻转初始化, 正常随机偏航初始化
            base_init_state[:, 3:7] = get_quat(random_yaw, 0.0)  # 设置随机偏航角, 翻滚角为0
                
        if self.custom_origins:  # 如果使用地形自定义原点(复杂地形模式)
            self.root_states[env_ids] = base_init_state  # 设置初始状态
            self.root_states[env_ids, :3] += self.env_origins[env_ids]  # 加上地形平台原点偏移
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center  # 在平台中心1m范围内随机偏移xy位置
        else:  # 平地或网格模式
            self.root_states[env_ids] = base_init_state  # 设置初始状态
            self.root_states[env_ids, :3] += self.env_origins[env_ids]  # 加上环境原点(网格偏移)
        # base velocities  # 设置初始速度
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel  # 随机初始化线速度和角速度(±0.5 m/s和rad/s)
        env_ids_int32 = env_ids.to(dtype=torch.int32)  # 将环境ID转为int32(IsaacGym要求)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),  # 完整根状态张量
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))  # 只更新指定环境的根状态

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        env_ids = torch.arange(self.num_envs, device=self.device)  # 生成所有环境ID
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]  # 找到当前步应该推动的环境(按固定间隔推动)
        if len(push_env_ids) == 0:  # 如果没有环境需要推动, 直接返回
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy  # 最大线速度推力(m/s)
        max_push_ang = self.cfg.domain_rand.max_push_ang_vel  # 最大角速度推力(rad/s)
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y  # 随机设置xy方向线速度(模拟水平冲击)
        self.root_states[:, 10:13] = torch_rand_float(-max_push_ang, max_push_ang, (self.num_envs, 3), device=self.device) # ang vel x/y/z  # 随机设置角速度(模拟倾覆冲击)

        env_ids_int32 = push_env_ids.to(dtype=torch.int32)  # 转为int32(IsaacGym要求)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),  # 完整根状态张量
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))  # 只更新被推动环境的状态
   
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])  # 创建与观测向量同形状的噪声缩放向量, 初始化为零
        self.add_noise = self.cfg.noise.add_noise  # 是否添加观测噪声的全局开关
        noise_scales = self.cfg.noise.noise_scales  # 各观测量的噪声缩放配置
        noise_level = self.cfg.noise.noise_level  # 整体噪声水平缩放系数(训练时可调整)
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel  # 线速度噪声: 噪声幅度*等级*观测缩放
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel  # 角速度噪声
        noise_vec[6:9] = noise_scales.gravity * noise_level  # 重力投影噪声(IMU测量误差)
        noise_vec[9:12] = 0. # commands  # 速度指令无噪声(已知量)
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos  # 关节位置噪声(编码器误差)
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # 关节速度噪声(微分计算误差)
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions  # 上一步动作无噪声(已知量)

        return noise_vec  # 返回噪声缩放向量, 在compute_observations中乘以均匀分布[-1,1]添加到观测

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors  # 从IsaacGym获取GPU状态张量(零拷贝访问)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  # 获取根状态张量(所有actor的位置/方向/速度)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # 获取DOF状态张量(关节位置/速度)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)  # 获取净接触力张量(用于碰撞检测)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  # 获取刚体状态张量(用于脚部位置计算)
        self.gym.refresh_dof_state_tensor(self.sim)  # 刷新关节状态(确保初始值正确)
        self.gym.refresh_actor_root_state_tensor(self.sim)  # 刷新根状态
        self.gym.refresh_net_contact_force_tensor(self.sim)  # 刷新接触力
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # 刷新刚体状态

        # create some wrapper tensors for different slices  # 创建张量视图(零内存拷贝地访问不同状态量)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)  # 根状态: shape=(num_envs, 13) [pos(3)+quat(4)+lin_vel(3)+ang_vel(3)]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)  # DOF状态: shape=(num_envs*num_dof, 2) [pos, vel]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]  # 提取关节位置: shape=(num_envs, num_dof)
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]  # 提取关节速度: shape=(num_envs, num_dof)
        self.base_quat = self.root_states[:, 3:7]  # 机身四元数视图: shape=(num_envs, 4)
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)  # 计算初始欧拉角: shape=(num_envs, 3)
        self.base_pos = self.root_states[:self.num_envs, 0:3]  # 机身位置视图: shape=(num_envs, 3)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis  # 接触力: shape=(num_envs, num_bodies, 3)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)  # 刚体状态: shape=(num_envs*num_bodies, 13)
        if self.cfg.terrain.measure_heights:  # 如果需要高度感知
            self.height_points = self._init_height_points()  # 初始化高度采样点(机体坐标系下的网格点)
            x_points = self.height_points[0, :, 0]  # 获取第一个环境的x坐标
            y_points = self.height_points[0, :, 1]  # 获取第一个环境的y坐标
            x_mask = (x_points >= -0.2) & (x_points <= 0.2)  # 0.4m length  # 筛选x方向[-0.2, 0.2]范围内的点(机身正下方区域)
            y_mask = (y_points >= -0.15) & (y_points <= 0.15)  # 0.3m width  # 筛选y方向[-0.15, 0.15]范围内的点
            self.base_height_scan_mask = (x_mask & y_mask).float()  # 生成机身下方区域掩码(用于高度估计)
            self.num_base_height_scan_points = self.base_height_scan_mask.sum()  # 统计掩码内的采样点数量
            assert self.num_base_height_scan_points > 0, "No height scan points within the specified area."  # 确保有足够采样点
        self.measured_heights = 0  # 初始化高度测量值为0

        # initialize some data used later on  # 初始化训练过程中使用的各种数据缓冲区
        self.common_step_counter = 0  # 全局步数计数器(用于课程学习进度判断)
        self.extras = {}  # 额外信息字典(传递给算法的日志和调试信息)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)  # 初始化观测噪声缩放向量
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))  # 重力单位向量(Z轴负方向), shape=(num_envs, 3)
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))  # 机器人前向单位向量(X轴正方向), shape=(num_envs, 3)
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # 关节力矩缓冲区: shape=(num_envs, num_actions)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # PD控制器比例增益Kp: shape=(num_actions,)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # PD控制器微分增益Kd: shape=(num_actions,)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # 当前步动作: shape=(num_envs, num_actions)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # 上一步动作(用于动作平滑度奖励)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # 前前步动作(用于二阶平滑度奖励)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)  # 上一步关节速度(用于加速度惩罚)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])  # 上一步根状态速度
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading  # 速度指令向量: [vx, vy, wz, heading]
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this  # 指令缩放系数(与观测缩放对应)
        self.commands_resampling_step = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)  # 指令重采样倒计时步数
        self.commands_xy_accumulation = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)  # xy方向累积指令(用于动态指令重采样)
        self.zero_command_proba = 0.0  # 零指令概率初始化(后续可通过课程更新)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)  # 脚部离地时间计时器: shape=(num_envs, 4)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)  # 上一步脚部接触状态(用于首次接触检测)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])  # 机身线速度(机体坐标系): shape=(num_envs, 3)
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])  # 机身角速度(机体坐标系): shape=(num_envs, 3)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)  # 重力在机体坐标系的投影: shape=(num_envs, 3)
        self.max_move_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)  # 每个环境episode内的最大水平移动距离(用于地形课程)
        self.stop_heading = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)  # 是否停止朝向控制的标志
        self.last_is_limit_vel = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)  # 上次指令是否为极限速度的标志
        self.motor_strengths = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # 电机强度系数(1.0=全强度, <1.0=电机退化)
        self.limit_vel_prob = self.cfg.commands.limit_vel_prob  # 极限速度指令出现的概率
        self.limit_vel_comb = torch.tensor(list(product(  # 生成速度组合笛卡尔积(每个方向取-1/0/1)
            self.cfg.commands.limit_vel["lin_vel_x"],  # x方向速度符号列表
            self.cfg.commands.limit_vel["lin_vel_y"],  # y方向速度符号列表
            self.cfg.commands.limit_vel["ang_vel_yaw"]  # 偏航角速度符号列表
        )), device=self.device, requires_grad=False)  # 速度组合张量(排除全零组合)
        self.last_robot_props_update_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)  # 上次机器人属性更新步数(预留扩展)
        self.turn_over_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)  # 翻转恢复零指令计时器(翻转后保持零速度的倒计时)
        self.env_command_ranges = {  # 每个环境的具体指令范围(受地形类型限制)
            'lin_vel_x': torch.tensor(self.command_ranges['lin_vel_x'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),  # x速度范围: shape=(num_envs, 2)
            'lin_vel_y': torch.tensor(self.command_ranges['lin_vel_y'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),  # y速度范围: shape=(num_envs, 2)
            'ang_vel_yaw': torch.tensor(self.command_ranges['ang_vel_yaw'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),  # 偏航角速度范围: shape=(num_envs, 2)
            'heading': torch.tensor(self.command_ranges['heading'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),  # 朝向范围: shape=(num_envs, 2)
        }
        self._update_env_command_ranges()  # 根据地形类型更新每个环境的指令范围上限

        # joint positions offsets and PD gains  # 初始化关节默认角度和PD增益
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)  # 默认关节角度(站立姿态): shape=(num_dof,)
        for i in range(self.num_dofs):  # 遍历所有关节
            name = self.dof_names[i]  # 获取关节名称(如 FL_hip_joint)
            angle = self.cfg.init_state.default_joint_angles[name]  # 从配置中读取该关节的默认角度
            self.default_dof_pos[i] = angle  # 设置默认关节角度
            found = False  # 标记是否找到对应的PD增益配置
            for dof_name in self.cfg.control.stiffness.keys():  # 遍历所有配置的PD增益关键字
                if dof_name in name:  # 如果关键字是关节名称的子串(如 "hip" 匹配 "FL_hip_joint")
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]  # 设置比例增益Kp(boying: 60 N·m/rad)
                    self.d_gains[i] = self.cfg.control.damping[dof_name]  # 设置微分增益Kd
                    found = True  # 标记已找到
            if not found:  # 如果没有找到对应的增益配置
                self.p_gains[i] = 0.  # 未配置的关节Kp设为0
                self.d_gains[i] = 0.  # 未配置的关节Kd设为0
                if self.cfg.control.control_type in ["P", "V"]:  # 如果使用PD控制(非力矩控制)
                    print(f"PD gain of joint {name} were not defined, setting them to zero")  # 打印警告
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)  # 增加batch维度: shape=(1, num_dof), 便于广播到(num_envs, num_dof)
    
    def _update_env_command_ranges(self):
        """ Update environment-wise command ranges based on current command ranges and terrain type """
        if not hasattr(self, 'terrain_ids'):  # 如果地形ID尚未初始化(初始化阶段)
            self.env_command_ranges = {  # 所有环境使用相同的全局指令范围
                'lin_vel_x': torch.tensor(self.command_ranges['lin_vel_x'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),  # x速度范围
                'lin_vel_y': torch.tensor(self.command_ranges['lin_vel_y'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),  # y速度范围
                'ang_vel_yaw': torch.tensor(self.command_ranges['ang_vel_yaw'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),  # 偏航角速度范围
                'heading': torch.tensor(self.command_ranges['heading'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),  # 朝向范围
            }
            return  # 直接返回, 不需要根据地形类型限制
        for terrain_id, terrain_command_ranges in enumerate(self.cfg.commands.terrain_max_command_ranges):  # 遍历每种地形类型的最大指令范围配置
            env_ids = (self.terrain_ids == terrain_id).nonzero(as_tuple=False).flatten()  # 找到属于该地形类型的环境ID
            if len(env_ids) == 0:  # 如果没有环境在该地形类型
                continue  # 跳过
            self.env_command_ranges['lin_vel_x'][env_ids, 0] = max(  # 更新x速度下限(取配置限制和当前全局范围的最大值)
                terrain_command_ranges['lin_vel_x'][0],  # 地形限制下限
                self.command_ranges['lin_vel_x'][0],  # 全局范围下限
            )
            self.env_command_ranges['lin_vel_x'][env_ids, 1] = min(  # 更新x速度上限(取配置限制和当前全局范围的最小值)
                terrain_command_ranges['lin_vel_x'][1],  # 地形限制上限
                self.command_ranges['lin_vel_x'][1]  # 全局范围上限
            )
            self.env_command_ranges['lin_vel_y'][env_ids, 0] = max(  # 更新y速度下限
                terrain_command_ranges['lin_vel_y'][0],  # 地形y速度下限
                self.command_ranges['lin_vel_y'][0]  # 全局y速度下限
            )
            self.env_command_ranges['lin_vel_y'][env_ids, 1] = min(  # 更新y速度上限
                terrain_command_ranges['lin_vel_y'][1],  # 地形y速度上限
                self.command_ranges['lin_vel_y'][1]  # 全局y速度上限
            )
            self.env_command_ranges['ang_vel_yaw'][env_ids, 0] = max(  # 更新偏航角速度下限
                terrain_command_ranges['ang_vel_yaw'][0],  # 地形偏航角速度下限
                self.command_ranges['ang_vel_yaw'][0]  # 全局偏航角速度下限
            )
            self.env_command_ranges['ang_vel_yaw'][env_ids, 1] = min(  # 更新偏航角速度上限
                terrain_command_ranges['ang_vel_yaw'][1],  # 地形偏航角速度上限
                self.command_ranges['ang_vel_yaw'][1]  # 全局偏航角速度上限
            )
            if self.cfg.commands.heading_command:  # 如果使用朝向控制
                self.env_command_ranges['heading'][env_ids, 0] = max(  # 更新朝向下限
                    terrain_command_ranges['heading'][0],  # 地形朝向下限
                    self.command_ranges['heading'][0]  # 全局朝向下限
                )
                self.env_command_ranges['heading'][env_ids, 1] = min(  # 更新朝向上限
                    terrain_command_ranges['heading'][1],  # 地形朝向上限
                    self.command_ranges['heading'][1]  # 全局朝向上限
                )

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt  # 过滤零权重奖励, 将非零权重乘以dt(使奖励与控制频率无关)
        def update_scales(scales):  # 内部函数: 更新奖励缩放字典
            for key in list(scales.keys()):  # 遍历所有奖励名称
                scale = scales[key]  # 获取当前奖励的缩放系数
                if scale==0:  # 如果缩放为零
                    scales.pop(key)  # 移除该奖励(零权重无需计算)
                else:  # 非零缩放
                    scales[key] *= self.dt  # 乘以时间步长dt(将每步奖励转为每秒奖励)
        update_scales(self.reward_scales)  # 更新正常奖励缩放
        if self.cfg.init_state.turn_over:  # 如果启用翻转模式
            update_scales(self.reward_turn_over_scales)  # 更新翻转模式奖励缩放
        # prepare list of functions  # 准备奖励函数列表
        self.reward_functions = []  # 存储奖励函数对象的列表
        self.reward_names = []  # 存储奖励函数名称的列表
        names = set()  # 用集合收集所有奖励名称(去重)
        names.update(list(self.reward_scales.keys()))  # 从正常奖励缩放中收集名称
        if self.cfg.init_state.turn_over:  # 如果启用翻转模式
            names.update(list(self.reward_turn_over_scales.keys()))  # 也收集翻转奖励名称
        for name in names:  # 遍历所有奖励名称
            if name=="termination":  # 终止奖励单独处理(在奖励裁剪后添加)
                continue  # 跳过termination
            self.reward_names.append(name)  # 添加到名称列表
            name = '_reward_' + name  # 构造函数名(如 "_reward_tracking_lin_vel")
            self.reward_functions.append(getattr(self, name))  # 动态查找并添加对应的奖励方法

        # reward episode sums  # 初始化episode奖励累计字典
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in names}  # 每个奖励项的episode累计: shape=(num_envs,)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()  # 创建平面参数对象
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)  # 平面法向量(Z轴正方向, 即水平面)
        plane_params.static_friction = self.cfg.terrain.static_friction  # 静摩擦系数(来自配置)
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction  # 动摩擦系数(来自配置)
        plane_params.restitution = self.cfg.terrain.restitution  # 弹性恢复系数(0=完全非弹性)
        self.gym.add_ground(self.sim, plane_params)  # 将地面平面添加到仿真中

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)  # 格式化资产文件路径(替换根目录占位符)
        asset_root = os.path.dirname(asset_path)  # 提取资产文件所在目录
        asset_file = os.path.basename(asset_path)  # 提取资产文件名(如 boying.urdf)

        asset_options = gymapi.AssetOptions()  # 创建资产加载选项对象
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode  # 默认DOF驱动模式(位置/速度/力矩)
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints  # 是否合并固定关节(减少自由度数量)
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule  # 是否用胶囊体替换圆柱体碰撞(提高稳定性)
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments  # 是否翻转视觉附件(某些URDF需要)
        asset_options.fix_base_link = self.cfg.asset.fix_base_link  # 是否固定基座(测试时可用)
        asset_options.density = self.cfg.asset.density  # 默认密度(未在URDF中指定时使用)
        asset_options.angular_damping = self.cfg.asset.angular_damping  # 角阻尼系数
        asset_options.linear_damping = self.cfg.asset.linear_damping  # 线性阻尼系数
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity  # 最大角速度限制
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity  # 最大线速度限制
        asset_options.armature = self.cfg.asset.armature  # 电枢惯量(模拟电机转动惯量)
        asset_options.thickness = self.cfg.asset.thickness  # 碰撞体厚度(避免穿透)
        asset_options.disable_gravity = self.cfg.asset.disable_gravity  # 是否禁用重力(测试用)

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)  # 加载机器人URDF资产
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)  # 获取关节自由度数量
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)  # 获取刚体数量
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)  # 获取URDF中定义的DOF属性(限制/增益等)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)  # 获取刚体形状属性(摩擦/弹性等)

        # save body names from the asset  # 保存资产中的刚体名称
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)  # 获取所有刚体名称列表
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)  # 获取所有DOF名称列表
        self.num_bodies = len(body_names)  # 更新刚体数量
        self.num_dofs = len(self.dof_names)  # 更新DOF数量
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]  # 筛选脚部刚体名称(用于接触检测)
        hip_names = [s for s in self.dof_names if 'hip' in s]  # 筛选髋关节DOF名称
        penalized_contact_names = []  # 需要惩罚碰撞的刚体名称列表
        for name in self.cfg.asset.penalize_contacts_on:  # 遍历配置的惩罚接触刚体
            penalized_contact_names.extend([s for s in body_names if name in s])  # 找到匹配的刚体名称
        termination_contact_names = []  # 触发终止的接触刚体名称列表
        for name in self.cfg.asset.terminate_after_contacts_on:  # 遍历配置的终止接触刚体
            termination_contact_names.extend([s for s in body_names if name in s])  # 找到匹配的刚体名称

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel  # 合并初始状态: 位置+四元数+线速度+角速度
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)  # 转为tensor: shape=(13,)
        start_pose = gymapi.Transform()  # 创建初始位姿对象
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])  # 设置初始位置(x, y, z)

        # domain rand  # 域随机化初始化
        self.motor_zero_offsets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # 电机零位偏置: shape=(num_envs, num_actions), 初始化为0
        self.p_gains_multiplier = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # Kp增益乘数: shape=(num_envs, num_actions), 初始化为1
        self.d_gains_multiplier = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # Kd增益乘数: shape=(num_envs, num_actions), 初始化为1
        if self.cfg.rewards.dynamic_sigma:  # 如果启用动态追踪sigma(根据速度和地形难度调整奖励精度)
            self.dynamic_sigma_cfg = self.cfg.rewards.dynamic_sigma  # 保存动态sigma配置
            self.terrain_max_sigmas = torch.tensor(self.dynamic_sigma_cfg["max_sigma"], device=self.device, requires_grad=False)  # 各地形类型对应的最大sigma值

        self._get_env_origins()  # 计算每个环境的原点坐标(地形平台中心或网格位置)
        env_lower = gymapi.Vec3(0., 0., 0.)  # 环境空间下界(所有actor共享空间, 设为0)
        env_upper = gymapi.Vec3(0., 0., 0.)  # 环境空间上界(设为0意味着IsaacGym自动管理)
        self.actor_handles = []  # 存储所有actor句柄的列表
        self.envs = []  # 存储所有环境句柄的列表
        for i in range(self.num_envs):  # 遍历创建每个并行环境
            # create env instance  # 创建环境实例
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))  # 创建环境(排列为近似正方形网格)
            pos = self.env_origins[i].clone()  # 复制该环境的原点坐标
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)  # 在原点附近±1m范围内随机偏移xy位置
            start_pose.p = gymapi.Vec3(*pos)  # 设置actor的初始位置

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)  # 处理刚体形状属性(摩擦系数随机化等)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)  # 将修改后的属性应用到资产
            actor_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)  # 创建机器人actor实例
            dof_props = self._process_dof_props(dof_props_asset, i)  # 处理DOF属性(提取关节限制)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)  # 设置actor的DOF属性
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)  # 获取actor的刚体属性
            if i == 0:  # 只保存第一个环境的默认刚体属性(用于对比)
                self.default_body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)  # 保存默认刚体属性
            body_props = self._process_rigid_body_props(body_props, i)  # 处理刚体属性(质量随机化等)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)  # 设置刚体属性(重新计算惯性张量)
            self.envs.append(env_handle)  # 保存环境句柄
            self.actor_handles.append(actor_handle)  # 保存actor句柄

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)  # 初始化脚部刚体索引
        for i in range(len(feet_names)):  # 遍历所有脚部名称
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])  # 查找脚部刚体的全局索引

        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)  # 初始化髋关节DOF索引
        for i in range(len(hip_names)):  # 遍历所有髋关节名称
            self.hip_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], hip_names[i])  # 查找髋关节DOF的全局索引

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)  # 初始化惩罚接触刚体索引
        for i in range(len(penalized_contact_names)):  # 遍历惩罚接触刚体名称
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])  # 查找刚体全局索引

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)  # 初始化终止接触刚体索引
        for i in range(len(termination_contact_names)):  # 遍历终止接触刚体名称
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])  # 查找刚体全局索引

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:  # 如果使用复杂地形
            self.custom_origins = True  # 标记使用地形自定义原点
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)  # 初始化环境原点: shape=(num_envs, 3)
            # put robots at the origins defined by the terrain  # 将机器人放置在地形平台定义的原点
            max_init_level = self.cfg.terrain.max_init_terrain_level  # 初始最大地形难度级别(课程学习起始难度)
            if not self.cfg.terrain.curriculum:  # 如果不使用地形课程
                max_init_level = self.cfg.terrain.num_rows - 1  # 使用最大行数作为初始难度上限

            # random choice terrain levels and types for each env  # 以下两行被注释: 随机分配地形级别和类型
            # self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            # self.terrain_types = torch.randint(0, self.cfg.terrain.num_cols, (self.num_envs,), device=self.device)

            # levels and types in a round robin manner  # 使用轮询方式分配地形级别和类型(更均匀的初始分布)
            self.terrain_levels = torch.fmod(torch.arange(self.num_envs, device=self.device), max_init_level + 1)  # 地形难度级别: 0到max_init_level循环
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs / self.cfg.terrain.num_cols), rounding_mode="floor").to(torch.long)  # 地形类型: 均匀分配到所有列
            self.terrain_cols2id = torch.tensor(self.terrain.cols2id, device=self.device)  # 地形列到类型ID的映射
            if len(self.terrain_cols2id):  # 如果存在类型映射
                self.terrain_ids = self.terrain_cols2id[self.terrain_types]  # 为每个环境分配地形类型ID(用于指令范围限制)

            self.max_terrain_level = self.cfg.terrain.num_rows  # 最大地形难度级别(总行数)
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)  # 从numpy数组加载地形平台原点坐标
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]  # 根据难度级别和类型为每个环境赋值原点

        else:  # 平面地形: 创建网格状分布
            self.custom_origins = False  # 标记不使用自定义原点
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)  # 初始化环境原点
            # create a grid of robots  # 创建机器人网格排列
            num_cols = np.floor(np.sqrt(self.num_envs))  # 列数: 取环境数量平方根的下取整
            num_rows = np.ceil(self.num_envs / num_cols)  # 行数: 向上取整确保覆盖所有环境
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))  # 生成行列网格坐标
            spacing = self.cfg.env.env_spacing  # 环境间距(米)
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]  # 设置x方向原点坐标
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]  # 设置y方向原点坐标
            self.env_origins[:, 2] = 0.  # z方向原点为0(平地)

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt  # 控制时间步长 = 仿真时间步长 * 降采样倍数(如4*0.005=0.02s=50Hz)
        self.obs_scales = self.cfg.normalization.obs_scales  # 观测量归一化缩放配置(各量的期望范围)
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)  # 将奖励缩放配置类转为字典
        self.reward_turn_over_scales = class_to_dict(self.cfg.rewards.turn_over_scales)  # 将翻转模式奖励缩放配置类转为字典
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)  # 将速度指令范围配置类转为字典
        self.max_lin_vel = max(abs(self.command_ranges["lin_vel_x"][0]), abs(self.command_ranges["lin_vel_x"][1]),
                               abs(self.command_ranges["lin_vel_y"][0]), abs(self.command_ranges["lin_vel_y"][1]))  # 计算最大线速度(取所有方向绝对值的最大值)
        self.cfg.commands.command_range_curriculum = sorted(self.cfg.commands.command_range_curriculum, key=lambda x: x['iter'], reverse=True)  # 按迭代次数从大到小排序课程(便于倒序遍历时安全删除)

        self.max_episode_length_s = self.cfg.env.episode_length_s  # episode最大时长(秒)
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)  # episode最大步数(向上取整)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)  # 推力间隔(步数) = 时间间隔/控制步长

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()  # 创建高度场地形参数对象
        hf_params.column_scale = self.terrain.cfg.horizontal_scale  # 列方向水平缩放(每格对应的米数)
        hf_params.row_scale = self.terrain.cfg.horizontal_scale  # 行方向水平缩放(每格对应的米数)
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale  # 垂直缩放(高度单位到米的转换)
        hf_params.nbRows = self.terrain.tot_cols  # 高度场行数(注意IsaacGym行列定义可能与直觉相反)
        hf_params.nbColumns = self.terrain.tot_rows  # 高度场列数
        hf_params.transform.p.x = -self.terrain.cfg.border_size  # x方向偏移(将地形中心对齐到原点)
        hf_params.transform.p.y = -self.terrain.cfg.border_size  # y方向偏移
        hf_params.transform.p.z = 0.0  # z方向偏移为0
        hf_params.static_friction = self.cfg.terrain.static_friction  # 地形静摩擦系数
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction  # 地形动摩擦系数
        hf_params.restitution = self.cfg.terrain.restitution  # 地形弹性系数

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)  # 将高度场添加到仿真
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)  # 将高度采样数据转为GPU张量

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()  # 创建三角网格地形参数对象
        tm_params.nb_vertices = self.terrain.vertices.shape[0]  # 设置顶点数量
        tm_params.nb_triangles = self.terrain.triangles.shape[0]  # 设置三角形数量

        tm_params.transform.p.x = -self.terrain.cfg.border_size  # x方向偏移(对齐到原点)
        tm_params.transform.p.y = -self.terrain.cfg.border_size  # y方向偏移
        tm_params.transform.p.z = 0.0  # z方向无偏移
        tm_params.static_friction = self.cfg.terrain.static_friction  # 静摩擦系数
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction  # 动摩擦系数
        tm_params.restitution = self.cfg.terrain.restitution  # 弹性系数
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  # 将三角网格添加到仿真(行主序展开)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)  # 加载高度采样数据(用于_get_heights查询)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum  # 实现游戏启发式地形课程: 表现好的机器人升级到更难地形
        if not self.init_done or self.cfg.terrain.mesh_type == 'plane':  # 如果初始化未完成或使用平面地形
            # don't change on initial reset  # 初始化重置时不改变地形级别
            return  # 直接返回
        # distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)  # 备用: 直接计算当前位移(已改为最大位移)
        distance = self.max_move_distance[env_ids]  # 使用episode内的最大水平移动距离(更稳健的性能指标)
        # robots that walked far enough progress to harder terains  # 走得足够远的机器人升级到更难地形
        move_up = distance > self.terrain.env_length / 2  # 移动超过地形格子长度一半则升级
        if self.cfg.terrain.move_down_by_accumulated_xy_command:  # 如果使用累积指令计算降级标准
            move_down = (distance < torch.norm(self.commands_xy_accumulation[env_ids], dim=1) * (self.cfg.commands.resampling_time * (1 - self.zero_command_proba)) * 0.5) * ~move_up  # 移动不足累积指令的50%则降级
        else:  # 使用当前指令计算降级标准
            # robots that walked less than half of their required distance go to simpler terrains  # 未走完要求距离一半的机器人降级
            move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~move_up  # 移动不足指令速度*时长*50%则降级

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down  # 升级+1, 降级-1
        # Robots that solve the last level are sent to a random one  # 完成最高难度的机器人随机重新分配级别
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),  # 随机分配新级别
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)  # 最低级别不低于0
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]  # 根据新的地形级别更新环境原点
        self.max_move_distance[env_ids] = 0.0  # 重置最大移动距离计数器
        

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)  # y方向采样点坐标列表(机体坐标系)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)  # x方向采样点坐标列表(机体坐标系)
        grid_x, grid_y = torch.meshgrid(x, y)  # 生成二维采样网格

        self.num_height_points = grid_x.numel()  # 总采样点数 = len(x) * len(y)
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)  # 初始化采样点张量: shape=(num_envs, num_points, 3)
        points[:, :, 0] = grid_x.flatten()  # 设置所有环境的x坐标(相同的采样网格)
        points[:, :, 1] = grid_y.flatten()  # 设置所有环境的y坐标
        return points  # 返回机体坐标系下的采样点(z坐标在_get_heights中由地形计算)

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":  # 如果是平面地形
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)  # 平面地形高度全为0
        elif self.cfg.terrain.mesh_type == "none":  # 如果地形类型为none
            raise NameError("Can't measure height with terrain mesh type 'none'")  # 抛出错误

        if env_ids:  # 如果指定了部分环境
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)  # 将机体坐标系采样点转换到世界坐标系(只考虑偏航旋转)
        else:  # 处理所有环境
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)  # 所有环境的采样点转换到世界坐标系

        points += self.terrain.cfg.border_size  # 加上边界偏移(将负坐标转为正索引)
        points = (points / self.terrain.cfg.horizontal_scale).long()  # 将米坐标转换为高度场整数索引
        px = points[:, :, 0].view(-1)  # x方向索引展开为1D
        py = points[:, :, 1].view(-1)  # y方向索引展开为1D
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)  # 限制x索引不越界(留1格余量)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)  # 限制y索引不越界

        heights1 = self.height_samples[px, py]  # 采样点(px, py)的高度
        heights2 = self.height_samples[px + 1, py]  # 采样点(px+1, py)的高度(相邻格子)
        heights3 = self.height_samples[px, py + 1]  # 采样点(px, py+1)的高度(相邻格子)
        heights = torch.min(heights1, heights2)  # 取最小高度(保守估计, 避免脚踩到悬空位置)
        heights = torch.min(heights, heights3)  # 再取最小高度

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale  # reshape回(num_envs, num_points)并转换为米


    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity  # 惩罚机身z轴线速度(减少上下弹跳运动)
        return torch.square(self.base_lin_vel[:, 2])  # 返回z方向线速度的平方(越大惩罚越重)

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity  # 惩罚机身xy轴角速度(减少翻滚和俯仰)
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)  # 返回roll和pitch角速度平方和

    def _reward_orientation(self):
        # Penalize non flat base orientation  # 惩罚机身倾斜(保持水平姿态)
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)  # 重力在机体xy平面的投影越大说明越倾斜

    # def _reward_base_height(self):
    #     # Penalize base height away from target  # 惩罚机身高度偏离目标值(已替换为更精确版本)
    #     base_height = self.root_states[:, 2]
    #     return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_base_height(self):
        # Penalize base height away from target  # 惩罚机身高度偏离目标值(改进版: 使用接触脚位置估计地面高度)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.  # 检测哪些脚有z方向接触力(接触地面)
        if not hasattr(self, 'last_contacts2'):  # 如果是第一次调用, 初始化上次接触状态
            self.last_contacts2 = torch.zeros_like(contact)  # 初始化上次接触状态为全False
        contact_filt = torch.logical_or(contact, self.last_contacts2)  # (N, 4)  # 使用当前和上次接触的OR(过滤PhysX接触不稳定性)
        self.last_contacts2 = contact  # 保存当前接触状态供下次使用
        feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]  # 获取所有脚的世界坐标位置
        num_feet_contact = torch.sum(contact_filt, dim=1, keepdim=True).clamp(min=1.0)  # (N, 1)  # 统计接触脚数量(最少1个, 避免除零)
        feet_contact_pos = (feet_pos * contact_filt.unsqueeze(-1)).sum(dim=1) / num_feet_contact  # (N, 3)  # 计算接触脚位置的加权平均(估计地面接触点)
        base_pos = self.root_states[:, 0:3]  # 获取机身世界坐标位置
        delta_pos = feet_contact_pos - base_pos  # 脚相对于机身的位移向量
        base_height = (delta_pos * self.projected_gravity).sum(1)  # (N,)  # 沿重力方向投影得到高度差(正值=机身在脚上方)
        rew = torch.square(base_height - self.cfg.rewards.base_height_target) * (contact_filt.sum(1) > 0)  # 高度误差平方, 只在有接触时计算
        return rew  # 返回机身高度误差奖励

    def _reward_torques(self):
        # Penalize torques  # 惩罚关节力矩(减少能量消耗, 提高效率)
        return torch.sum(torch.square(self.torques), dim=1)  # 返回所有关节力矩平方和

    def _reward_dof_vel(self):
        # Penalize dof velocities  # 惩罚关节速度(减少高速运动, 提高稳定性)
        return torch.sum(torch.square(self.dof_vel), dim=1)  # 返回所有关节速度平方和

    def _reward_dof_acc(self):
        # Penalize dof accelerations  # 惩罚关节加速度(减少抖动, 保护电机)
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)  # 返回关节角加速度平方和(有限差分近似)

    def _reward_action_rate(self):
        # Penalize changes in actions  # 惩罚动作变化率(减少控制抖动, 提高平滑性)
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)  # 返回相邻步动作差的平方和

    def _reward_collision(self):
        # Penalize collisions on selected bodies  # 惩罚指定刚体的碰撞(如大腿碰地)
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)  # 统计有接触力(>0.1N)的惩罚刚体数量

    def _reward_termination(self):
        # Terminal reward / penalty  # 终止惩罚(摔倒但非超时时惩罚)
        return self.reset_buf * ~self.time_out_buf  # 只在非超时的终止(摔倒)时给惩罚(True=1, False=0)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit  # 惩罚关节角度接近软限制(保护关节)
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit  # 下限: 超出量(负值裁剪为正)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)  # 上限: 超出量(正值)
        return torch.sum(out_of_limits, dim=1)  # 返回所有关节超出软限制的总量

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit  # 惩罚关节速度接近限制(保护电机)
        # clip to max error = 1 rad/s per joint to avoid huge penalties  # 裁剪到最大1 rad/s误差, 避免极端惩罚
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)  # 超出软速度限制的量(裁剪到[0,1])

    def _reward_torque_limits(self):
        # penalize torques too close to the limit  # 惩罚力矩接近限制(避免电机过载)
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)  # 超出软力矩限制的量
    
    def _get_dynamic_sigma(self, target_vel_abs, v_min, v_max):
        # compute dynamic sigma based on terrain level  # 根据地形难度动态计算追踪奖励的sigma值
        # sigma越大 = 追踪奖励越宽松(允许更大的速度误差), 在困难地形上提高鲁棒性
        default_sigma = self.cfg.rewards.tracking_sigma  # 默认sigma值(平地或低速时使用)
        if not self.cfg.terrain.curriculum or self.cfg.rewards.dynamic_sigma is None or not hasattr(self, 'terrain_ids'):  # 如果不使用课程或动态sigma未配置
            return torch.full_like(target_vel_abs, default_sigma)  # 返回全为默认sigma的张量
        target_sigmas = self.terrain_max_sigmas[self.terrain_ids]  # 根据每个环境的地形类型获取目标最大sigma
        sigma = torch.full_like(target_vel_abs, default_sigma)  # 初始化sigma为默认值
        # based on velocity ranges, compute sigma  # 根据速度范围进行线性插值
        # v_min <= v < v_max (linear interpolation)  # 在v_min到v_max之间线性插值
        mask = (target_vel_abs >= v_min) & (target_vel_abs < v_max)  # 找到速度在[v_min, v_max)范围的环境
        if mask.any():  # 如果有环境在插值区间内
            ratio = (target_vel_abs[mask] - v_min) / (v_max - v_min)  # 计算插值比例[0,1]
            sigma[mask] = default_sigma + ratio * (target_sigmas[mask] - default_sigma)  # 线性插值: default_sigma到target_sigma
        # v >= v_max  # 高速时使用最大sigma
        mask = target_vel_abs >= v_max  # 找到速度≥v_max的环境
        if mask.any():  # 如果有高速环境
            sigma[mask] = target_sigmas[mask]  # 直接使用地形对应的目标sigma
        # based on terrain level, compute sigma  # 再根据地形难度级别调整sigma
        level_scale = torch.clamp(torch.exp((self.terrain_levels.float() + 1.0) / 10.0) - 1.0, max=1.0)  # 地形级别缩放: 指数增长(难度越高sigma越大)
        sigma = default_sigma + level_scale * (sigma - default_sigma)  # 最终sigma = 基础值 + 级别缩放 * (目标值-基础值)
        return sigma  # 返回每个环境的动态sigma值

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)  # 追踪线速度指令奖励(核心运动奖励)
        if self.cfg.rewards.dynamic_sigma is None:  # 如果不使用动态sigma
            sigma_x = sigma_y = self.cfg.rewards.tracking_sigma  # 使用固定sigma值
        else:  # 使用动态sigma(根据指令速度大小调整)
            vmin = self.dynamic_sigma_cfg["min_lin_vel"]  # 线速度插值下界
            vmax = self.dynamic_sigma_cfg["max_lin_vel"]  # 线速度插值上界
            sigma_x = self._get_dynamic_sigma(torch.abs(self.commands[:, 0]), vmin, vmax)  # x方向动态sigma
            sigma_y = self._get_dynamic_sigma(torch.abs(self.commands[:, 1]), vmin, vmax)  # y方向动态sigma
        lin_vel_error_sq = torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2])  # 计算xy方向速度误差的平方
        scaled_error = lin_vel_error_sq[:, 0] / sigma_x + lin_vel_error_sq[:, 1] / sigma_y  # 用sigma归一化误差(sigma越大=越宽松)
        # print(f"{self.base_lin_vel[:, :2]=}, {lin_vel_error_sq=}")  # 调试打印(已注释)
        return torch.exp(-scaled_error)  # exp(-误差/sigma): 误差为0时奖励为1, 误差增大奖励指数衰减

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)  # 追踪偏航角速度指令奖励
        if self.cfg.rewards.dynamic_sigma is None:  # 如果不使用动态sigma
            sigma = self.cfg.rewards.tracking_sigma  # 使用固定sigma值
        else:  # 使用动态sigma
            vmin = self.dynamic_sigma_cfg["min_ang_vel"]  # 角速度插值下界
            vmax = self.dynamic_sigma_cfg["max_ang_vel"]  # 角速度插值上界
            sigma = self._get_dynamic_sigma(torch.abs(self.commands[:, 2]), vmin, vmax)  # 动态sigma
        ang_vel_error_sq = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])  # 偏航角速度误差的平方
        return torch.exp(-ang_vel_error_sq/sigma)  # exp(-误差/sigma): 高斯型追踪奖励

    def _reward_feet_air_time(self):
        # Reward long steps  # 奖励较长的步态周期(鼓励机器人迈大步)
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes  # 需要过滤接触信号, PhysX网格接触检测不稳定
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.  # 检测z方向接触力>1N的脚(当前接触状态)
        contact_filt = torch.logical_or(contact, self.last_contacts)  # 取当前和上次的OR(防止接触信号抖动导致漏检)
        self.last_contacts = contact  # 更新上次接触状态
        first_contact = (self.feet_air_time > 0.) * contact_filt  # 识别首次落地: 之前在空中(air_time>0)且当前接触
        self.feet_air_time += self.dt  # 每步累加离地时间
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground  # 首次落地时给奖励: 离地时间超过0.5s才有正奖励(促进稳定步态)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command  # 零速指令时不给步态奖励(允许站立)
        self.feet_air_time *= ~contact_filt  # 接触地面时重置空中计时器(保留非接触脚的计时)
        return rew_airTime  # 返回步态空中时间奖励

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces  # 惩罚脚撞击垂直面(避免绊脚)
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)  # 水平接触力>5倍垂直接触力时判定为撞击垂直面

    def _reward_stand_still(self):
        # Penalize motion at zero commands  # 惩罚零指令时的运动(促进站立稳定性)
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)  # 零速指令时惩罚偏离默认姿态的关节角度

    def _reward_feet_contact_forces(self):
        # penalize high contact forces  # 惩罚过大的脚部接触力(保护关节和地面冲击)
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)  # 超过最大允许接触力的超出量之和

    def _reward_action_smoothness(self):
        # a_t - 2a_{t-1} + a_{t-2}  # 二阶差分: 计算动作的二阶导数(加速度)
        rew = torch.sum((self.actions - 2 * self.last_actions + self.last_last_actions).pow(2), dim=1)  # 动作二阶差分的平方和(惩罚动作加速度)
        return rew  # 比action_rate更严格: 同时惩罚速度和加速度变化

    def _reward_dof_power(self):
        # Penalize power consumption  # 惩罚功率消耗(能效优化)
        power = self.torques * self.dof_vel  # 功率 = 力矩 * 角速度 (W)
        rew = torch.sum(torch.abs(power), dim=1)  # 所有关节功率绝对值之和(总机械功率)
        return rew  # 负功(制动)也计入总功率(保守估计)

    def _get_base_height(self):
        if not self.cfg.terrain.measure_heights:  # 如果不测量地形高度
            return self.root_states[:, 2]  # 直接返回机身z坐标(平地适用)
        # 根据高度扫描点计算base link到地面估计高度  # 使用机身下方高度扫描点估计地面高度
        masked_heights = self.measured_heights * self.base_height_scan_mask.unsqueeze(0)  # 只保留机身正下方区域的高度点
        sum_heights = masked_heights.sum(dim=1)  # 累加机身下方区域的高度值
        estimated_ground_z = sum_heights / self.num_base_height_scan_points  # 平均高度作为地面高度估计

        base_z = self.root_states[:, 2]  # 机身世界坐标z值
        base_height = base_z - estimated_ground_z  # (N,)  # 机身相对于地面的高度
        return base_height  # 返回估计的机身离地高度(米)

    def _reward_correct_base_height(self):
        base_height = self._get_base_height()  # 获取机身离地高度
        rew = torch.square(base_height - self.cfg.rewards.base_height_target)  # 高度误差的平方(MSE)
        return rew  # 返回高度惩罚(与目标高度的偏差)

    def _reward_feet_regulation(self):
        # CTS抬腿正则奖励, 在脚末端速度增大同时, 要求高度尽可能高  # CTS(并发师生)特有奖励: 鼓励迈步时抬高脚部
        base_height = self._get_base_height()  # 更新刚体空间位置 (开悟比赛无法修改环境, 只能在奖励中完成计算了)  # 获取机身离地高度
        feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]  # 获取所有脚的世界坐标: shape=(N, 4, 3)
        feet_xy_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:9]  # 获取所有脚的xy速度: shape=(N, 4, 2)
        base_pos = self.root_states[:, 0:3].unsqueeze(1)  # 机身位置: shape=(N, 1, 3) 用于广播
        delta_feet = feet_pos - base_pos  # 脚相对于机身的位移向量: shape=(N, 4, 3)
        feet2base_height = (delta_feet * self.projected_gravity.unsqueeze(1)).sum(-1)  # 脚相对于身体的高度 (N, 4)  # 沿重力方向投影得到高度差(负值=脚在机身下方)
        feet_height = torch.clamp(base_height.unsqueeze(1) - feet2base_height, min=0.0)  # 脚相对于地面的高度 (N, 4)  # 脚的绝对离地高度(不能为负)
        rew = (feet_xy_vel.pow(2).sum(-1) * torch.exp(-feet_height / (0.025 * self.cfg.rewards.base_height_target))).sum(-1)  # 脚水平速度的平方 * 高度衰减因子之和(脚抬得越高, 速度惩罚越小)
        return rew  # 鼓励快速迈步时抬高脚部(高脚步态)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose  # 惩罚关节角度偏离默认姿态(促进自然站立姿态)
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)  # 所有关节偏离默认角度的绝对值之和

    def _reward_upright(self):
        return (-1 - self.projected_gravity[:, 2]) / 2  # 机身竖直奖励: gravity_z=-1为完全竖直(奖励=0), gravity_z=0为水平(奖励=-0.5)

    def _reward_legs_distance(self):
        # Penalize legs being too close to each other  # 惩罚腿部间距过小(防止腿部碰撞)
        # feet_names: [FL_foot, FR_foot, RL_foot, RR_foot]  # 脚部名称顺序: 左前、右前、左后、右后
        feet_pos_world = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]  # (N, 4, 3)  # 获取脚部世界坐标
        base_pos = self.root_states[:, 0:3]  # (N, 3)  # 机身世界坐标
        base_quat = self.base_quat  # (N, 4)  # 机身四元数方向
        feet_pos_relative_world = feet_pos_world - base_pos.unsqueeze(1)  # (N, 4, 3)  # 脚相对机身的世界坐标系位移
        local_pos = quat_rotate_inverse(  # 将脚的相对位置转换到机体坐标系
            base_quat.repeat_interleave(4, dim=0),  # 为4条腿扩展四元数
            feet_pos_relative_world.reshape(-1, 3)  # 展开为(N*4, 3)
        ).reshape(self.num_envs, 4, 3)  # (N, 4, 3)  # 机体坐标系下的脚部位置

        dy_front =  local_pos[:, 0, 1] - local_pos[:, 1, 1]  # (N,)  # 前腿左右间距(FL_y - FR_y)
        dy_rear =  local_pos[:, 2, 1] - local_pos[:, 3, 1]   # (N,)  # 后腿左右间距(RL_y - RR_y)
        min_dist = self.cfg.rewards.min_legs_distance  # 腿部最小允许间距(米)

        rew_front = torch.square(torch.clamp(min_dist - dy_front, min=0.0))  # 前腿间距小于最小值时的惩罚(平方)
        rew_rear = torch.square(torch.clamp(min_dist - dy_rear, min=0.0))  # 后腿间距小于最小值时的惩罚(平方)
        return rew_front + rew_rear  # 前后腿间距惩罚之和
