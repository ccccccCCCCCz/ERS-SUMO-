# ers_platoon_env.py
import traci
from sumolib import checkBinary
import numpy as np


class ERSPlatoonEnv:
    """
    改良版 SUMO 环境，适配连续动作 MADDPG（actor 输出范围 [-1,1]）。
    每个 agent 对应一辆车（或代表一辆车），action_dim=2:
      action[0] in [-1,1] -> maps to Δv in [-max_delta_v, +max_delta_v] (m/s)
      action[1] in [-1,1] -> lane change intent (if > lane_thr -> try left, < -lane_thr -> try right)

    Observation (per agent, obs_dim=5):
      [soc_norm, speed_norm, is_on_ers (0/1), dist_to_next_ers_norm, time_remaining_norm]
    """

    def __init__(self,
                 sumo_cfg_path="./scenario/ers.sumocfg",
                 n_agents=3,
                 gui=False,
                 ers_lane_ids=["1389427540_0"],
                 max_distance=1000.0,
                 max_speed=30.0,
                 max_delta_v=3.0,
                 decision_interval=5,
                 soc_low_threshold=0.1,
                 fail_on_any_low_energy=True):
        # SUMO binary
        if gui:
            self.sumoBinary = checkBinary('sumo-gui')
        else:
            self.sumoBinary = checkBinary('sumo')
        self.sumoCmd = [self.sumoBinary, "-c", sumo_cfg_path, "--no-step-log", "true", "-W"]

        # env params
        self.n_agents = n_agents
        self.decision_interval = decision_interval
        self.max_distance = float(max_distance)  # for normalization
        self.max_speed = float(max_speed)
        self.max_delta_v = float(max_delta_v)
        self.soc_low_threshold = float(soc_low_threshold)
        self.fail_on_any_low_energy = bool(fail_on_any_low_energy)

        # action/obs dims
        self.action_dim = 2
        self.obs_dim = 5

        # ERS lane ids list (strings like "1389427540_0"); if None, will try to autodetect by searching chargingStation lanes
        self.ers_lane_ids = ers_lane_ids

        # lane change threshold (in [-1,1])
        self.lane_thr = 0.6

        # internal
        self.time = 0
        self.max_time = 3600  # safety timeout (you can change)
        self.agent_ids = []


    # ----------------- Helper utilities -----------------
    def _ensure_ers_lane_ids(self):
        """如果没有手动传 ers_lane_ids，尝试从 additional chargingStation 上找到 lane id（若可用）"""
        if self.ers_lane_ids is None:
            # 尝试从 sumo 获取所有 lanes 上是否有 chargingStation —— SUMO API 无直接查询 chargingStation->lane，
            # 通servative fallback: user should pass ers_lane_ids for 精确性.
            self.ers_lane_ids = []
            # 用户最好传入 ers_lane_ids；这里仅留空集合不会出错
        return

    def _is_on_ers_lane(self, lane_id):
        if lane_id is None:
            return False
        if self.ers_lane_ids is None:
            return False
        return lane_id in self.ers_lane_ids

    def _distance_to_next_ers(self, vid):
        """
        返回车辆 vid 到下一个 ERS 入口的剩余距离（m），若找不到返回 max_distance。
        简单实现：如果车辆当前 edge 的剩余到 edge 末端距离加上路段间的估计距离，
        此处用 traci.route 或 net shortest path 会复杂——先给一个保守实现：
        - 若当前 lane 就是 ERS，则返回 0
        - 否则返回 min 距离（若知道 ERS lane 的 lane length & pos 可更精确）
        """
        lane_id = traci.vehicle.getLaneID(vid)
        if self._is_on_ers_lane(lane_id):
            return 0.0

        # 简化：返回到下一个目标 ERS lane 的直线距离（基于位置）——使用 position (x,y)
        try:
            x, y = traci.vehicle.getPosition(vid)
            min_d = self.max_distance
            for ers_lane in (self.ers_lane_ids or []):
                # 获取 lane 中心点或车道首末点 (如无则跳过)
                try:
                    # 获取车道形状并用第一个点作为代表
                    shape = traci.lane.getShape(ers_lane)
                    if len(shape) > 0:
                        ex, ey = shape[0]
                        d = ((ex - x)**2 + (ey - y)**2)**0.5
                        if d < min_d:
                            min_d = d
                except Exception:
                    continue
            return float(min_d)
        except Exception:
            # 若 traci 报错，返回一个大值
            return float(self.max_distance)


    # ----------------- Core env API -----------------
    def reset(self):
        """启动 SUMO 并返回初始观测 (n_agents, obs_dim)，等待直到所有车辆就绪。"""
        # 1. 启动 Traci
        traci.start(self.sumoCmd)
        self.time = 0

        # 2. 等待所有 n_agents 车辆进入路网
        wait_steps = 0
        max_wait_steps = 500  # 设置一个最大等待时间，防止无限循环

        # 持续运行仿真直到找到足够的车辆
        while len(traci.vehicle.getIDList()) < self.n_agents:
            if wait_steps >= max_wait_steps:
                traci.close()
                raise RuntimeError(
                    f"SUMO vehicle count < n_agents ({self.n_agents}) after {max_wait_steps} steps. "
                    "Please check your .rou.xml file."
                )

            # 运行一步仿真
            traci.simulationStep()
            self.time += 1
            wait_steps += 1

        # 3. 确定 agent_ids
        # 这里选择前 n_agents 辆车作为智能体
        self.agent_ids = list(traci.vehicle.getIDList()[:self.n_agents])

        # 4. 确保 ERS lanes list
        self._ensure_ers_lane_ids()

        # 5. 返回观测
        # 注意：由于在等待过程中已经运行了 self.time 步，
        # 初始观测已经是 t=wait_steps 时的状态。
        return self._get_obs()

    def _get_obs(self):
        """返回 np.array shape (n_agents, obs_dim)"""
        obs = []
        for vid in self.agent_ids:
            # SOC: 使用 getParameter 的 battery 参数 —— 注意不同 SUMO 版本 key 名称差异，请按你 SUMO 版本调整
            try:
                # 尝试两种常见参数名：actualBatteryCharge / device.battery.chargeLevel 等
                # 这里使用 device.battery.chargeLevel 与 device.battery.capacity 约定
                charge = float(traci.vehicle.getParameter(vid, "device.battery.chargeLevel"))
                cap = float(traci.vehicle.getParameter(vid, "device.battery.capacity"))
                soc = np.clip(charge / max(cap, 1e-6), 0.0, 1.0)
            except Exception:
                # 如果没有 battery device，则用默认 1.0（或你可以抛错）
                soc = 1.0

            # speed normalized
            speed = traci.vehicle.getSpeed(vid)
            speed_norm = np.clip(speed / self.max_speed, 0.0, 1.0)

            # is_on_ers
            lane_id = traci.vehicle.getLaneID(vid)
            is_on_ers = 1.0 if self._is_on_ers_lane(lane_id) else 0.0

            # distance to next ers normalized
            dist_to_ers = self._distance_to_next_ers(vid)
            dist_norm = np.clip(dist_to_ers / float(self.max_distance), 0.0, 1.0)

            # time remaining normalized
            time_remain = np.clip((self.max_time - self.time) / max(1.0, self.max_time), 0.0, 1.0)

            obs.append(np.array([soc, speed_norm, is_on_ers, dist_norm, time_remain], dtype=np.float32))
        return np.array(obs, dtype=np.float32)


    def step(self, actions):
        """
        actions: np.array shape (n_agents, action_dim) or list of vectors
        Each action in [-1,1]^action_dim
        """
        # check shape
        actions = np.asarray(actions, dtype=np.float32)
        assert actions.shape == (self.n_agents, self.action_dim), \
            f"actions.shape {actions.shape} != {(self.n_agents, self.action_dim)}"

        # 1) apply continuous actions for each agent

        """
        actions: np.array shape (n_agents, action_dim) or list of vectors
        Each action in [-1,1]^action_dim
        """
        # ... (速度控制代码不变)

        # 1) apply continuous actions for each agent
        for idx, vid in enumerate(self.agent_ids):
            a = actions[idx]
            delta_v_norm = float(a[0])
            lane_signal = float(a[1])

            # ... (速度设置代码不变)

            # lane change decision
            if lane_signal > self.lane_thr:
                # 尝试左变道 (车道索引变小)
                try:
                    cur_lane_index = traci.vehicle.getLaneIndex(vid)
                    # 目标车道：取当前车道索引和 0 中的较大值 (防止变道到索引 < 0 的车道)
                    target_lane = max(cur_lane_index - 1, 0)
                    traci.vehicle.changeLane(vid, target_lane, self.decision_interval)
                except Exception:
                    pass
            elif lane_signal < -self.lane_thr:
                # 尝试右变道 (车道索引变大)
                try:
                    cur_lane_index = traci.vehicle.getLaneIndex(vid)

                    # 🌟 关键修正 🌟
                    # 1. 获取当前道路的车道总数 (num_lanes)
                    # 从当前车道ID获取路段ID (e.g., "1389427542_2" -> "1389427542")
                    lane_id = traci.vehicle.getLaneID(vid)
                    edge_id = lane_id.split("_")[0] if lane_id else None

                    if edge_id:
                        num_lanes = traci.edge.getLaneNumber(edge_id)
                        # 目标车道：取 cur_lane_index + 1 和 num_lanes - 1 中的较小值
                        # 确保 target_lane 不会大于最大索引 (num_lanes - 1)
                        target_lane = min(cur_lane_index + 1, num_lanes - 1)
                    else:
                        # 车辆可能不在任何路段上，或无法获取 edge_id，跳过此次变道
                        continue

                    # 执行变道操作
                    traci.vehicle.changeLane(vid, target_lane, self.decision_interval)
                except Exception as e:
                    # 捕获异常，例如 traci.edge.getLaneNumber 失败或 changeLane 失败
                    pass
            # else no lane change intent

        # ... (仿真步进代码不变)
            # else no lane change intent

        # 2) step simulation for decision_interval steps
        for _ in range(self.decision_interval):
            traci.simulationStep()
            self.time += 1

        # 3) next obs, reward, done
        next_obs = self._get_obs()
        rewards = self._get_reward(next_obs)
        done = self._get_done()

        info = {}  # can include diagnostics like per-agent energy gained, ers usage etc.
        return next_obs, rewards, done, info


    def _get_reward(self, obs):
        """
        简单 reward 设计（可按需替换）
        组合项：
          - soc_penalty: 电量低惩罚（鼓励保持高 SOC）
          - charge_bonus: 在 ERS 区段且 SOC 增加时奖励（需从前后 charge 差计算）
          - speed_eff: 鼓励接近目标速度（这里假设目标 speed = 0.8*max_speed）
        注意：更精确的 ers_charge 需要在 step 前后读取 charge 并差分，环境这里用近似。
        """
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        target_speed = 0.8 * self.max_speed

        for i, vid in enumerate(self.agent_ids):
            soc = float(obs[i][0])
            is_on_ers = bool(obs[i][2])
            speed_norm = float(obs[i][1])

            # soc penalty
            soc_penalty = -1.0 if soc < self.soc_low_threshold else 0.0

            # charge bonus: if on ers and soc not full -> small positive reward
            charge_bonus = 0.5 if is_on_ers and soc < 0.99 else 0.0

            # speed efficiency reward (closer to target speed is better)
            speed = speed_norm * self.max_speed
            speed_eff = -0.1 * abs(speed - target_speed)

            rewards[i] = soc_penalty + charge_bonus + speed_eff

        return rewards


    def _get_done(self):
        """结束条件（可配置）"""
        # any_low_energy?
        any_low = False
        try:
            for v in self.agent_ids:
                try:
                    charge = float(traci.vehicle.getParameter(v, "device.battery.chargeLevel"))
                    cap = float(traci.vehicle.getParameter(v, "device.battery.capacity"))
                    soc = charge / max(cap, 1e-6)
                except Exception:
                    soc = 1.0
                if soc < self.soc_low_threshold:
                    any_low = True
                    break
        except Exception:
            any_low = False

        # all arrived?
        all_arrived = True
        for v in self.agent_ids:
            try:
                # route index equals last index -> arrived
                if traci.vehicle.getRouteIndex(v) < traci.vehicle.getRoute(v).getLength() - 1:
                    all_arrived = False
                    break
            except Exception:
                all_arrived = False
                break

        timeout = self.time >= self.max_time

        done_flag = False
        if self.fail_on_any_low_energy and any_low:
            done_flag = True
        elif (not self.fail_on_any_low_energy) and all_arrived:
            done_flag = True
        elif timeout:
            done_flag = True

        if done_flag:
            try:
                traci.close()
            except Exception:
                pass
            return True
        return False


    def close(self):
        try:
            traci.close()
        except Exception:
            pass
