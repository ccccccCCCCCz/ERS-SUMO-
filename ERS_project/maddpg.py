# maddpg_continuous.py
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def update_model(source, target, tau):
    """soft update: target = tau*source + (1-tau)*target"""
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


class Actor(nn.Module):
    """Continuous actor: obs_dim -> hidden -> hidden -> action_dim (tanh)"""
    def __init__(self, obs_dim, action_dim, hidden=[64, 64]):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], action_dim),
            nn.Tanh()   # outputs in [-1,1]
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """Centralized critic: concatenated state + concatenated action -> Q"""
    def __init__(self, state_dim, action_dim, hidden=[128, 128]):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 1)
        )

    def forward(self, state, action):
        # state: (batch, state_dim), action: (batch, action_dim)
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        """
        transition is a tuple:
        (obs_all, actions_all, next_obs_all, rewards_all, dones_all)

        - obs_all: np.array shape (n_agents, obs_dim) or flattened shape (n_agents*obs_dim,)
        - actions_all: np.array shape (n_agents, action_dim) or flattened
        - rewards_all: iterable of length n_agents (per-agent reward)
        - dones_all: iterable of length n_agents (per-agent done flag)
        We will store flattened obs/actions for simplicity.
        """
        self.memory.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        # each element in transitions: (obs_all, actions_all, next_obs_all, rewards_all, dones_all)
        obs_batch, act_batch, next_obs_batch, rew_batch, done_batch = zip(*transitions)

        # Ensure shapes are (batch, state_dim) where state_dim = n_agents * obs_dim
        obs_batch = np.vstack([np.array(x).flatten() for x in obs_batch])
        act_batch = np.vstack([np.array(x).flatten() for x in act_batch])
        next_obs_batch = np.vstack([np.array(x).flatten() for x in next_obs_batch])
        # rewards and dones: stack as arrays with shape (batch, n_agents)
        rew_batch = np.vstack([np.array(x).reshape(-1) for x in rew_batch])
        done_batch = np.vstack([np.array(x).reshape(-1) for x in done_batch])

        # convert to tensors
        states = torch.tensor(obs_batch, dtype=torch.float32)
        actions = torch.tensor(act_batch, dtype=torch.float32)
        next_states = torch.tensor(next_obs_batch, dtype=torch.float32)
        rewards = torch.tensor(rew_batch, dtype=torch.float32)
        dones = torch.tensor(done_batch, dtype=torch.float32)

        return states, actions, next_states, rewards, dones


class MADDPG:
    """
    Continuous-action MADDPG (centralized critic, decentralized actors).
    - n_agents: number of agents
    - obs_dim: per-agent observation dim
    - action_dim: per-agent action dim (continuous, -1..1)
    """
    def __init__(self, n_agents, obs_dim, action_dim, lr=1e-3, gamma=0.99, batch_size=256, tau=1e-3,
                 memory_capacity=100000, device=None):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = obs_dim * n_agents
        self.action_all_dim = action_dim * n_agents

        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # actors and critics
        self.actors = nn.ModuleList([Actor(obs_dim, action_dim).to(self.device) for _ in range(n_agents)])
        self.target_actors = nn.ModuleList([Actor(obs_dim, action_dim).to(self.device) for _ in range(n_agents)])
        self.critics = nn.ModuleList([Critic(self.state_dim, self.action_all_dim).to(self.device) for _ in range(n_agents)])
        self.target_critics = nn.ModuleList([Critic(self.state_dim, self.action_all_dim).to(self.device) for _ in range(n_agents)])

        # init target networks
        for i in range(n_agents):
            update_model(self.actors[i], self.target_actors[i], tau=1.0)
            update_model(self.critics[i], self.target_critics[i], tau=1.0)

        # optimizers
        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=self.lr) for i in range(n_agents)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=self.lr) for i in range(n_agents)]

        # replay
        self.memory = ReplayMemory(capacity=memory_capacity)

        # exploration noise params (Gaussian)
        self.noise_scale = 0.2
        self.noise_decay = 0.995
        self.min_noise = 0.02

        self.mse_loss = nn.MSELoss()

    def update_noise(self):
        self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)

    def select_action(self, obs, agent_idx, noise=True):
        """
        obs: np.array shape (obs_dim,) for one agent
        returns: action (np.array shape (action_dim,))
        """
        self.actors[agent_idx].eval()
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, obs_dim)
        with torch.no_grad():
            action = self.actors[agent_idx](x).squeeze(0).cpu().numpy()  # in [-1,1]
        if noise:
            action = action + np.random.normal(0, self.noise_scale, size=action.shape)
        action = np.clip(action, -1.0, 1.0)
        self.actors[agent_idx].train()
        return action

    def push(self, transition):
        """
        transition: (obs_all, actions_all, next_obs_all, rewards_all, dones_all)
        where obs_all: shape (n_agents, obs_dim) or list of per-agent obs
        actions_all: shape (n_agents, action_dim)
        rewards_all: list/array length n_agents
        dones_all: list/array length n_agents
        """
        # Basic checks done in ReplayMemory.push
        self.memory.push(transition)

    def train_ready(self):
        return len(self.memory) >= self.batch_size

    def train_model(self, agent_idx):
        """
        Update critic_i and actor_i (agent_idx).
        """
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        # move to device
        states = states.to(self.device)           # (B, state_dim)
        actions = actions.to(self.device)         # (B, action_all_dim)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)         # (B, n_agents)
        dones = dones.to(self.device)             # (B, n_agents)

        # ---- critic update ----
        # current Q
        current_q = self.critics[agent_idx](states, actions)  # (B,1)

        # build next_actions via target actors: for each agent j, actor_j(next_state_j)
        next_states_view = next_states.view(self.batch_size, self.n_agents, self.obs_dim)
        next_actions_list = []
        for j in range(self.n_agents):
            sj = next_states_view[:, j, :]  # (B, obs_dim)
            aj_next = self.target_actors[j](sj)  # (B, action_dim)
            next_actions_list.append(aj_next)
        next_actions = torch.cat(next_actions_list, dim=1)  # (B, action_all_dim)

        with torch.no_grad():
            next_q = self.target_critics[agent_idx](next_states, next_actions)  # (B,1)

        # use per-agent reward and done (index agent_idx)
        r_i = rewards[:, agent_idx].unsqueeze(1)  # (B,1)
        d_i = dones[:, agent_idx].unsqueeze(1)    # (B,1)

        target_q = r_i + self.gamma * (1.0 - d_i) * next_q

        # critic loss
        value_loss = self.mse_loss(current_q, target_q)
        self.critic_optimizers[agent_idx].zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), 1.0)
        self.critic_optimizers[agent_idx].step()

        # ---- actor update ----
        # build actions where agent_i action is from current actor (to be updated), others detached
        states_view = states.view(self.batch_size, self.n_agents, self.obs_dim)
        actions_list = []
        for j in range(self.n_agents):
            sj = states_view[:, j, :]
            if j == agent_idx:
                aj = self.actors[j](sj)  # (B, action_dim)  - requires grad
            else:
                with torch.no_grad():
                    aj = self.actors[j](sj)
            actions_list.append(aj)
        current_actions_for_all = torch.cat(actions_list, dim=1)  # (B, action_all_dim)

        # policy loss (maximize Q -> minimize -Q)
        policy_loss = -self.critics[agent_idx](states, current_actions_for_all).mean()

        self.actor_optimizers[agent_idx].zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 1.0)
        self.actor_optimizers[agent_idx].step()

        # soft update target networks for this agent
        update_model(self.actors[agent_idx], self.target_actors[agent_idx], self.tau)
        update_model(self.critics[agent_idx], self.target_critics[agent_idx], self.tau)

        # decay exploration noise a bit
        self.update_noise()

        return policy_loss.item(), value_loss.item()

    def save_model(self, path):
        """Saves all actors and critics (state_dicts)"""
        ckpt = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'target_actors': [ta.state_dict() for ta in self.target_actors],
            'target_critics': [tc.state_dict() for tc in self.target_critics],
            'optimizers': [opt.state_dict() for opt in self.actor_optimizers]  # optional
        }
        torch.save(ckpt, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(ckpt['actors'][i])
            self.critics[i].load_state_dict(ckpt['critics'][i])
            self.target_actors[i].load_state_dict(ckpt['target_actors'][i])
            self.target_critics[i].load_state_dict(ckpt['target_critics'][i])
