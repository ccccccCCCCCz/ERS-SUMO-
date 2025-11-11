import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

# 假设您的环境类名为 ERSPlatoonEnv 且在 env.py 中
from env import ERSPlatoonEnv
from maddpg import MADDPG  # 这里是你上面那份MADDPG类
from utils import get_average_travel_time  # 假设这个函数可以工作

parser = argparse.ArgumentParser()
parser.add_argument("-R", "--render", action="store_true",
                    help="whether to render SUMO GUI while training")
args = parser.parse_args()

if __name__ == "__main__":
    # ===== 检查 SUMO 环境 =====
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")

    # ----------------------------------------------
    # 1. 创建环境并运行 reset 获取初始状态和实际智能体数量
    # ----------------------------------------------
    # 注意：这里使用 ERSPlatoonEnv，并按其 __init__ 签名传入 gui 参数
    try:
        env = ERSPlatoonEnv(gui=args.render)
        state = env.reset()  # shape: (n_agents_actual, state_dim)
    except Exception as e:
        sys.exit(f"Environment initialization or reset failed: {e}")

    # ----------------------------------------------
    # 2. 动态获取参数
    # ----------------------------------------------
    n_agents_actual = state.shape[0]
    state_dim = state.shape[1]  # 从状态形状动态获取 obs_dim (应该是 5)

    # ===== 参数配置（使用动态值） =====
    action_dim = 2  # 连续动作维度
    n_episode = 100
    max_steps = 200

    print(f"--- Training Config ---")
    print(f"Detected N_Agents: {n_agents_actual}")
    print(f"Detected State Dim (obs_dim): {state_dim}")
    print(f"Action Dim: {action_dim}")
    print("-------------------------")

    # ----------------------------------------------
    # 3. 创建智能体
    # ----------------------------------------------
    agent = MADDPG(n_agents_actual, state_dim, action_dim)

    performance_list = []

    # ===== 开始训练 =====
    for episode in range(n_episode):
        # state 已经在循环外获取了一次，但我们通常在循环开始时调用 reset
        if episode > 0:
            state = env.reset()  # 重新 reset 以启动新的回合

        episode_reward = np.zeros(n_agents_actual)
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            actions = []
            # 循环从 0 到 n_agents_actual-1
            for i in range(n_agents_actual):
                actions.append(agent.select_action(state[i], i))

            actions = np.array(actions)
            before_state = state.copy()

            # 由于 ERSPlatoonEnv.step 返回 (next_obs, rewards, done, info)
            next_state, reward, done, info = env.step(actions)
            reward = np.array(reward)

            # !!! 修正 Replay Memory 的 done 处理 !!!
            # 转化为 n_agents_actual 长度的 done 数组
            done_array = np.array([done] * n_agents_actual)

            transition = (before_state, actions, next_state, reward, done_array)
            agent.push(transition)

            # 使用 n_agents_actual 进行训练
            if agent.train_ready():  # 假设您已将 train_start() 改为 train_ready()
                for i in range(n_agents_actual):
                    policy_loss, value_loss = agent.train_model(i)

            episode_reward += reward
            state = next_state
            step_count += 1

        env.close()

        # 假设 get_average_travel_time() 能在 traci.close() 之后工作
        average_traveling_time = get_average_travel_time()
        performance_list.append(average_traveling_time)

        print(f"Episode {episode + 1}/{n_episode} | "
              f"AverageTravelTime: {average_traveling_time:.3f} | "
              f"MeanReward: {episode_reward.mean():.3f}")

    # ===== 保存模型 =====
    os.makedirs("results", exist_ok=True)
    # 假设您的 save 方法是 save(path)
    agent.save_model("results/maddpg_traffic_model.pth")

    # ===== 可视化表现 =====
    plt.style.use('ggplot')
    plt.figure(figsize=(10.8, 7.2), dpi=120)
    plt.plot(performance_list)
    plt.xlabel('Episodes')
    plt.ylabel('Average Travel Time')
    plt.title('Performance of MADDPG for Traffic Control')
    plt.savefig('./results/performance.png')
    plt.show()