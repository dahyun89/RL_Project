import numpy as np
import pandas as pd
import config
import torch
from sklearn.ensemble import IsolationForest
import random


class Environment():
    def __init__(self, anomaly_data, unlabeld_data, test_data, test_label):
        self.anomaly_data = anomaly_data
        self.unlabeld_data = unlabeld_data
        self.test_data = test_data
        self.test_label = test_label
        self.current_data = unlabeld_data[random.randint(0, len(unlabeld_data) - 1)]
        self.current_class = 1
        self.clf = IsolationForest(contamination=config.contamination_rate)
        self.mapped = torch.tensor([])
        self.target_net = None
        self.net = None
        self.obs_dim = self.current_data.size()[0]
        self.action_dim = 2

    def reset(self):
        self.current_data = self.unlabeld_data[random.randint(0, len(self.unlabeld_data) - 1)]
        self.current_class = 1
        self.clf = IsolationForest(contamination=config.contamination_rate)
        self.mapped = torch.tensor([])
        self.refresh_iforest(self.net)

        return self.current_data

    def refresh_net(self, net):
        self.net = net

    def refresh_iforest(self, net):
        self.target_net = net
        with torch.no_grad():
            self.mapped = net.map(self.unlabeld_data).cpu()
        self.clf.fit(self.mapped)

    def intrinsic_reward(self):
        target = self.target_net.map(self.current_data)
        score = -self.clf.score_samples(target.detach().cpu().numpy().reshape(1, -1))

        return score

    def external_reward(self, action):
        if self.current_class == 0 and action == 1:
            score = 1
        elif self.current_class == 1 and action == 0:
            score = 0
        else:
            score = -1

        return score

    def sample_method_one(self):
        self.current_class = 0
        self.current_data = self.anomaly_data[random.randint(0, config.anomaly_num - 1)]

    def sample_method_two(self, action):
        self.current_class = 1
        candidate = np.random.choice([i for i in range(len(self.unlabeld_data))], size=config.sample_num, replace=False)
        with torch.no_grad():
            mapped_current = self.net.map(self.current_data).cpu()
        if action == 0:
            max_dist = -float('inf')
            for ind in candidate:
                dist = np.linalg.norm(mapped_current - self.net.map(self.unlabeld_data[ind]).detach().cpu())
                if dist > max_dist:
                    max_dist = dist
                    self.current_data = self.unlabeld_data[ind]
        else:
            min_dist = float('inf')
            for ind in candidate:
                dist = np.linalg.norm(mapped_current - self.net.map(self.unlabeld_data[ind]).detach().cpu())
                if dist < min_dist and dist != 0:
                    min_dist = dist
                    self.current_data = self.unlabeld_data[ind]

    def step(self, action):
        r_i = self.intrinsic_reward()[0]
        r_e = self.external_reward(action)
        reward = r_i + r_e

        choice = np.random.choice([0, 1], size=1, p=[config.p, 1 - config.p])
        if choice == 0:
            self.sample_method_one()
        else:
            self.sample_method_two(action)

        return self.current_data, reward, False, " "
