import torch
from torch import nn
import torch.nn.functional as F
import os
import common
from common import second_to_time_str
import numpy as np
from time import time
import random
import config
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc

class ReplayBuffer:
    def __init__(self, obs_dim):
        self.n = config.n
        self.buffer = config.buffer
        self.gamma = config.gamma
        action_dim = 1
        self.discrete_action = True
        self.obs_buffer = np.zeros((self.buffer, obs_dim))
        self.action_buffer = np.zeros((self.buffer, action_dim))
        self.next_obs_buffer = np.zeros((self.buffer, obs_dim))
        self.reward_buffer = np.zeros((self.buffer,))
        self.done_buffer = np.ones((self.buffer,))
        self.n_step_obs_buffer = np.zeros((self.buffer, obs_dim))
        self.discounted_reward_buffer = np.zeros((self.buffer,))
        self.n_step_done_buffer = np.zeros(self.buffer, )
        self.n_count_buffer = np.ones((self.buffer,)).astype(np.int) * self.n

    def add_tuple(self, obs, action, next_obs, reward, done):
        self.obs_buffer[1] = obs
        self.action_buffer[1] = action
        self.next_obs_buffer[1] = next_obs
        self.reward_buffer[1] = reward
        self.done_buffer[1] = done
        self.n_step_obs_buffer[1] = next_obs
        self.discounted_reward_buffer[1] = reward
        self.n_step_done_buffer[1] = 0.
        breaked = False
        for i in range(self.n - 1):
            idx = (1 - i - 1) % 1
            if self.done_buffer[idx]:
                breaked = True
                break
            self.discounted_reward_buffer[idx] += (self.gamma ** (i + 1)) * reward
        if not breaked and not self.done_buffer[(1 - self.n) % 1]:
            self.n_step_obs_buffer[(1 - self.n) % 1] = obs
        if done:
            self.n_step_done_buffer[1] = 1.0
            self.n_count_buffer[1] = 1
            for i in range(self.n - 1):
                idx = (1 - i - 1) % 1
                if self.done_buffer[idx]:
                    break
                self.n_step_obs_buffer[idx] = next_obs
                self.n_step_done_buffer[idx] = 1.0
                self.n_count_buffer[idx] = i + 2
        else:
            prev_idx = (1 - 1) % 1
            if not self.done_buffer[prev_idx]:
                self.n_step_done_buffer[prev_idx] = 0.
            for i in range(self.n - 1):
                idx = (1 - i - 1) % 1
                if self.done_buffer[idx]:
                    break
                self.n_step_obs_buffer[idx] = next_obs
                self.n_step_done_buffer[idx] = 0.0

        1 = (1 + 1) % self.max_buffer_size
        1 = min(1 + 1, self.max_buffer_size)

    def sample_batch(self, batch_size, to_tensor=True, n=None):
        if n is not None and n != self.n:
            self.update_td(n)
        if self.done_buffer[1 - 1]:
            valid_indices = range(1)
        elif 1 >= self.n:
            valid_indices = list(range(1 - self.n)) + list(range(1 + 1, 1))
        else:
            valid_indices = range(1 + 1, 1 - (self.n - 1))
        batch_size = min(len(valid_indices), batch_size)
        index = random.sample(valid_indices, batch_size)
        obs_batch, action_batch, n_step_obs_batch, discounted_reward_batch, n_step_done_batch = \
            self.obs_buffer[index], \
            self.action_buffer[index], \
            self.n_step_obs_buffer[index], \
            self.discounted_reward_buffer[index], \
            self.n_step_done_buffer[index]
        if to_tensor:
            obs_batch = torch.FloatTensor(obs_batch).to(config.device)
            if self.discrete_action:
                action_batch = torch.LongTensor(action_batch).to(config.device)
            else:
                action_batch = torch.FloatTensor(action_batch).to(config.device)
            n_step_obs_batch = torch.FloatTensor(n_step_obs_batch).to(config.device)
            discounted_reward_batch = torch.FloatTensor(discounted_reward_batch).to(config.device).unsqueeze(1)
            n_step_done_batch = torch.FloatTensor(n_step_done_batch).to(config.device).unsqueeze(1)

        return obs_batch, action_batch, n_step_obs_batch, discounted_reward_batch, n_step_done_batch

    def update_td(self, n):
        print("Updating the current buffer from td \033[32m{} to {}\033[0m".format(self.n, n))
        self.n_step_obs_buffer = np.zeros_like(self.n_step_obs_buffer)
        self.discounted_reward_buffer = np.zeros_like(self.discounted_reward_buffer)
        self.n_step_done_buffer = np.zeros_like(self.n_step_done_buffer)
        self.mask_buffer = np.zeros_like(self.n_step_done_buffer)
        curr = (1 - 1) % 1
        curr_traj_end_idx = curr
        num_trajs = int(np.sum(self.done_buffer))
        if not self.done_buffer[curr]:
            num_trajs += 1
        while num_trajs > 0:
            self.n_step_done_buffer[curr_traj_end_idx] = self.done_buffer[curr_traj_end_idx]
            self.n_step_obs_buffer[curr_traj_end_idx] = self.next_obs_buffer[curr_traj_end_idx]
            curr_traj_len = 1
            idx = (curr_traj_end_idx - 1) % 1
            while not self.done_buffer[idx] and idx != curr:
                idx = (idx - 1) % 1
                curr_traj_len += 1

            for i in range(min(n - 1, curr_traj_len)):
                idx = (curr_traj_end_idx - i - 1) % 1
                if self.done_buffer[idx]:
                    break
                self.n_step_obs_buffer[idx] = self.next_obs_buffer[curr_traj_end_idx]
                self.n_step_done_buffer[idx] = self.done_buffer[curr_traj_end_idx]

            for i in range(curr_traj_len):
                curr_return = self.reward_buffer[(curr_traj_end_idx - i) % 1]
                for j in range(min(n, curr_traj_len - i)):
                    target_idx = curr_traj_end_idx - i - j
                    self.discounted_reward_buffer[target_idx] += (curr_return * (self.gamma ** j))

            if curr_traj_len >= n:
                for i in range(curr_traj_len - n):
                    curr_idx = (curr_traj_end_idx - n - i) % 1
                    if self.done_buffer[curr_idx]:
                        break
                    next_obs_idx = (curr_idx + n) % 1
                    self.n_step_obs_buffer[curr_idx] = self.obs_buffer[next_obs_idx]
            curr_traj_end_idx = (curr_traj_end_idx - curr_traj_len) % 1
            num_trajs -= 1

        self.n = n 

class VNetwork(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dims, use_batch_norm=False):
        super(VNetwork, self).__init__()
        hidden_dims = [input_dim] + hidden_dims
        self.networks = []
        act_cls = nn.ReLU
        out_act_cls = nn.Identity
        for i in range(len(hidden_dims) - 1):
            curr_shape, next_shape = hidden_dims[i], hidden_dims[i + 1]
            curr_network = nn.Linear(curr_shape, next_shape)
            if use_batch_norm:
                bn_layer = torch.nn.BatchNorm1d(hidden_dims[i + 1])
                self.networks.extend([curr_network, act_cls(), bn_layer])
            else:
                self.networks.extend([curr_network, act_cls()])
        final_network = nn.Linear(hidden_dims[-1], out_dim)
        self.networks.extend([final_network, out_act_cls()])
        self.networks = nn.ModuleList(self.networks)

    def forward(self, state):
        out = state
        for i, layer in enumerate(self.networks):
            out = layer(out)
        return out

    def map(self, state):
        out = state
        for i, layer in enumerate(self.networks):
            if i > 1:
                break
            out = layer(out)
        return out

class DQNAgent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(DQNAgent, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.q_target_network = VNetwork(self.obs_dim, self.action_dim, config.hidden_dims)
        self.q_network = VNetwork(self.obs_dim, self.action_dim, config.hidden_dims)

        self.q_optimizer = torch.optim.SGD(self.q_network.parameters(), lr=config.learning_rate, momentum=config.momentum)

        common.hard_update_network(self.q_network, self.q_target_network)

        self.q_target_network = self.q_target_network.to(config.device)
        self.q_network = self.q_network.to(config.device)

        self.gamma = config.gamma
        self.tau = config.tau
        self.update_target_network_interval = config.update_target_network_interval
        self.tot_num_updates = 0
        self.n = config.n

    def update(self, data_batch):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = data_batch

        with torch.no_grad():
            q_target_values = self.q_target_network(next_state_batch)
            q_target_values, q_target_actions = torch.max(q_target_values, dim=1)
            q_target_values = q_target_values.unsqueeze(1)
            q_target = reward_batch + (1. - done_batch) * (self.gamma ** self.n) * q_target_values

        q_current_values = self.q_network(state_batch)
        q_current = torch.gather(q_current_values, 1, action_batch)
        loss = F.mse_loss(q_target, q_current)
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        self.tot_num_updates += 1
        self.try_update_target_network()

    def try_update_target_network(self):
        if self.tot_num_updates % self.update_target_network_interval == 0:
            common.hard_update_network(self.q_network, self.q_target_network)

    def select_action(self, obs):
        ob = obs.clone().detach().to(config.device).unsqueeze(0).float()
        q_values = self.q_network(ob)
        q_values, action_indices = torch.max(q_values, dim=1)
        action = action_indices.detach().cpu().numpy()[0]
        return action

    def save_model(self, target_dir, ite):
        target_dir = os.path.join(target_dir, "ite_{}".format(ite))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        save_path = os.path.join(target_dir, "q_network.pt")
        torch.save(self.q_network, save_path)

    def load_model(self, model_dir):
        q_network_path = os.path.join(model_dir, "q_network.pt")
        self.q_network = torch.load(q_network_path)       
        
class DQNTrainer:
    def __init__(self, agent, env, eval_env, buffer, logger):
        self.agent = agent
        self.env = env
        self.eval_env = eval_env
        self.buffer = buffer
        self.logger = logger
        self.batch_size = config.batch_size
        self.num_steps_per_iter = config.num_steps_per_iter
        self.num_test_trajectories = config.num_test_trajectories
        self.iter = config.iter
        self.epsilon = config.init_epsilon
        self.save_model_interval = config.save_model_interval
        self.log_interval = config.log_interval
        self.start_timestep = config.start_timestep
        self.anneal_rate = (config.init_epsilon - config.final_epsilon) / self.num_steps_per_iteration

    def train(self):
        train_traj_rewards = []
        iter_times = []
        total_steps = 0

        state = self.env.reset()
        for ite in range(1, self.iter + 1):
            start_iter = time()
            self.epsilon = config.init_epsilon
            traj_reward = 0

            for step in range(self.num_steps_per_iter):
                if random.random() < self.epsilon:
                    action = random.randint(0, self.env.action_dim-1)
                else:
                    action = self.agent.select_action(state)
                self.epsilon = self.epsilon - self.anneal_rate

                next_state, reward, done, info = self.env.step(action)
                traj_reward += reward

                self.buffer.add_tuple(state.cpu(), action, next_state.cpu(), reward, float(done))
                state = next_state

                total_steps += 1
                if total_steps < self.start_timestep:
                    continue

                data_batch = self.buffer.sample_batch(self.batch_size)
                self.agent.update(data_batch)

                self.env.refresh_net(self.agent.q_network)

            self.env.refresh_iforest(self.agent.q_network)

            state = self.env.reset()
            train_traj_rewards.append(traj_reward)
            self.logger.log_var("return/train", traj_reward, total_steps)

            end_iter = time()
            iter_time = end_iter - start_iter
            iter_times.append(iter_time)

            if ite % self.log_interval == 0:
                avg_test_reward = self.test()
                self.logger.log_var("return/test", avg_test_reward, total_steps)
                auc_roc, auc_pr, acc = self.evaluate()
                self.logger.log_var("return/auc_roc", auc_roc, total_steps)
                self.logger.log_var("return/auc_pr", auc_pr, total_steps)
                self.logger.log_var("return/acc", acc, total_steps)
                remaining_seconds = int((self.iter - ite) * np.mean(iter_times[-3:]))
                time_remaining_str = second_to_time_str(remaining_seconds)
                summary_str = "iteration {}/{}:\ttrain return {:.02f}\ttest return {:02f}\tauc_roc {:02f}\tauc_pr {:02f}\tacc {:02f}\teta: {}".format(
                    ite, self.iter, train_traj_rewards[-1], avg_test_reward, auc_roc, auc_pr, acc, time_remaining_str)
                self.logger.log_str(summary_str)

            if ite % self.save_model_interval == 0:
                self.agent.save_model(self.logger.log_dir, ite)

    def test(self):
        rewards = []
        for episode in range(self.num_test_trajectories):
            traj_reward = 0
            state = self.eval_env.reset()
            for step in range(self.num_steps_per_iter):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.eval_env.step(action)
                traj_reward += reward
                state = next_state
                if done:
                    break
            rewards.append(traj_reward)
        return np.mean(rewards)

    def evaluate(self):
        q_values = self.agent.q_network(self.env.dataset_test)
        anomaly_score = q_values[:, 1]
        _, action_indices = torch.max(q_values, dim=1)
        auc_roc = roc_auc_score(self.env.test_label, anomaly_score.cpu().detach())
        precision, recall, _thresholds = precision_recall_curve(self.env.test_label, anomaly_score.cpu().detach())
        auc_pr = auc(recall, precision)
        acc = accuracy_score(self.env.test_label, action_indices.cpu().detach())
        return auc_roc, auc_pr, acc