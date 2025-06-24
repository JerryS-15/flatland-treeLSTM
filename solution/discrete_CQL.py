import argparse
import os
import time

from flatland.envs.line_generators import SparseLineGen
from flatland.envs.malfunction_generators import (
    MalfunctionParameters,
    ParamMalfunctionGen,
)
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import SparseRailGen
from flatland_cutils import TreeObsForRailEnv as TreeCutils

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from eval_env import LocalTestEnvWrapper
from cqlActor import Actor
from nn.net_newCQL import CQLNetwork
from replayBuffer import ReplayBuffer
from trainCQL import CQLTrainer
import copy



class MultiAgentDiscreteCQL:
    def __init__(
        self,
        num_actions,
        device,
        alpha=1.0,
        discount=0.99,
        tau=0.005,
        lr=3e-4,
        target_update_freq=1000
    ):
        self.device = device

        self.model = CQLNetwork().to(device)
        self.target_model = copy.deepcopy(self.model).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.num_actions = num_actions
        self.alpha = alpha
        self.discount = discount
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.iterations = 0

    def train(self, batch):
        device = self.device

        # agents_attr = torch.tensor(batch['agent_attr'], dtype=torch.float32).to(device)
        # forest = torch.tensor(batch['forest'], dtype=torch.float32).to(device)
        # adjacency = torch.tensor(batch['adjacency'], dtype=torch.int64).to(device)
        # node_order = torch.tensor(batch['node_order'], dtype=torch.int64).to(device)
        # edge_order = torch.tensor(batch['edge_order'], dtype=torch.int64).to(device)
        # actions = torch.tensor(batch['actions'], dtype=torch.int64).to(device)
        # rewards = torch.tensor(batch['all_rewards'], dtype=torch.float32).to(device)
        # dones = torch.tensor(batch['dones'], dtype=torch.float32).to(device)
        # next_agents_attr = torch.tensor(batch['next_agent_attr'], dtype=torch.float32).to(device)
        # next_forest = torch.tensor(batch['next_forest'], dtype=torch.float32).to(device)
        # next_adjacency = torch.tensor(batch['next_adjacency'], dtype=torch.int64).to(device)
        # next_node_order = torch.tensor(batch['next_node_order'], dtype=torch.int64).to(device)
        # next_edge_order = torch.tensor(batch['next_edge_order'], dtype=torch.int64).to(device)
        agents_attr = batch['agent_attr'].clone().detach().to(device)
        forest = batch['forest'].clone().detach().to(device)
        adjacency = batch['adjacency'].clone().detach().long().to(device)
        node_order = batch['node_order'].clone().detach().long().to(device)
        edge_order = batch['edge_order'].clone().detach().long().to(device)
        actions = batch['actions'].clone().detach().to(device)
        rewards = batch['all_rewards'].clone().detach().to(device)
        dones = batch['dones'].clone().detach().to(device)
        next_agents_attr = batch['next_agent_attr'].clone().detach().to(device)
        next_forest = batch['next_forest'].clone().detach().to(device)
        next_adjacency = batch['next_adjacency'].clone().detach().long().to(device)
        next_node_order = batch['next_node_order'].clone().detach().long().to(device)
        next_edge_order = batch['next_edge_order'].clone().detach().long().to(device)

        # Target Q
        with torch.no_grad():
            next_q_values = self.target_model(next_agents_attr, next_forest, next_adjacency, next_node_order, next_edge_order)
            max_next_q, _ = next_q_values.max(dim=-1, keepdim=True)
            max_next_q = max_next_q.squeeze(-1)
            # print("**********DEBUG**********")
            # print("rewards.shape:", rewards.shape)
            # print("dones.shape:", dones.shape)
            # print("max_next_q.shape:", max_next_q.shape)
            # print("**********DEBUG**********")
            target_q = rewards + (1.0 - dones) * self.discount * max_next_q

        current_q_values = self.model(agents_attr, forest, adjacency, node_order, edge_order)
        chosen_q = current_q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        bellman_loss = F.mse_loss(chosen_q, target_q)

        # CQL Regularization
        logsumexp_q = torch.logsumexp(current_q_values, dim=-1)
        cql_penalty = (logsumexp_q - chosen_q).mean()

        total_loss = bellman_loss + self.alpha * cql_penalty

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if self.iterations % self.target_update_freq == 0:
            self._soft_update()

        self.iterations += 1

        return {
            'total_loss': total_loss.item(),
            'q_loss': bellman_loss.item(),
            'cql_penalty': cql_penalty.item(),
            'q_mean': current_q_values.mean().item(),
            'target_q_mean': target_q.mean().item()
        }

    def _soft_update(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def act(self, states_attr, forest, adjacency, node_order, edge_order, epsilon=0.05):
        with torch.no_grad():
            q_values = self.model(states_attr, forest, adjacency, node_order, edge_order)
            if np.random.rand() < epsilon:
                return torch.randint(0, self.num_actions, q_values.shape[:-1], device=q_values.device)
            return q_values.argmax(dim=-1)
    
    def save(self, filename):
        torch.save(self.model.state_dict(), filename + "_model.pt")
        torch.save(self.optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename + "_model.pt", map_location=self.device))
        self.target_model = copy.deepcopy(self.model)
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer", map_location=self.device))
        