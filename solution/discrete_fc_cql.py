import numpy as np
import torch
import torch.nn.functional as F

from nn.fc_cql import CQLNetwork
import copy

class MultiAgentDiscreteCQL:
    def __init__(
        self,
        num_actions,
        device,
        alpha=10.0,
        discount=0.99,
        tau=0.005,
        lr=1e-4,  # lower from 3e-4, improve stability
        target_update_freq=200, # lower from 1000
        # auto_alpha = False
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

        # self.auto_alpha = auto_alpha
        # if auto_alpha:
        #     self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        #     self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def train(self, batch):
        device = self.device

        agents_attr = batch['agent_attr'].clone().detach().to(device)
        forest = batch['forest'].clone().detach().to(device)
        adjacency = batch['adjacency'].clone().detach().long().to(device)
        # node_order = batch['node_order'].clone().detach().long().to(device)
        # edge_order = batch['edge_order'].clone().detach().long().to(device)
        actions = batch['actions'].clone().detach().to(device)
        rewards = batch['all_rewards'].clone().detach().to(device)
        dones = batch['dones'].clone().detach().to(device)
        next_agents_attr = batch['next_agent_attr'].clone().detach().to(device)
        next_forest = batch['next_forest'].clone().detach().to(device)
        next_adjacency = batch['next_adjacency'].clone().detach().long().to(device)
        # next_node_order = batch['next_node_order'].clone().detach().long().to(device)
        # next_edge_order = batch['next_edge_order'].clone().detach().long().to(device)

        # Target Q
        with torch.no_grad():
            next_q_values = self.target_model(next_agents_attr, next_forest)
            max_next_q, _ = next_q_values.max(dim=-1, keepdim=True)
            max_next_q = max_next_q.squeeze(-1)
            # print("**********DEBUG**********")
            # print("rewards.shape:", rewards.shape)
            # print("dones.shape:", dones.shape)
            # print("max_next_q.shape:", max_next_q.shape)
            # print("**********DEBUG**********")
            target_q = rewards + (1.0 - dones) * self.discount * max_next_q

        current_q_values = self.model(agents_attr, forest)
        chosen_q = current_q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        bellman_loss = F.mse_loss(chosen_q, target_q)

        # CQL Regularization - uniform mu(s, a)
        logsumexp_q = torch.logsumexp(current_q_values, dim=-1)

        # min_q = current_q_values.min(dim=-1).values.detach()

        cql_penalty = (logsumexp_q - chosen_q).mean()

        # --------------- Behavior Cloning loss ---------------
        # actions = actions.long().squeeze(-1)
        # bc_loss = F.cross_entropy(current_q_values, actions)
        # bc_lambda = 1.0  # weight for behavior cloning loss

        # --------------- Warm-up logic ---------------
        # warmup_steps = 5000
        # apply_cql = self.iterations >= warmup_steps

        # alpha = self.log_alpha.exp() if self.auto_alpha else self.alpha

        total_loss = bellman_loss + self.alpha * cql_penalty
        # --------------- Total Loss ---------------
        # total_loss = bellman_loss + bc_lambda * bc_loss
        # total_loss = bellman_loss
        # if apply_cql:
        #     total_loss += alpha * cql_penalty

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # if self.auto_alpha and apply_cql:
        #     alpha_loss = -self.log_alpha * (cql_penalty.detach() - 1.0)  # target gap adjustable
        #     self.alpha_optimizer.zero_grad()
        #     alpha_loss.backward()
        #     self.alpha_optimizer.step()

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
            q_values = self.model(states_attr, forest)
            if np.random.rand() < epsilon:
                return torch.randint(0, self.num_actions, q_values.shape[:-1], device=q_values.device)
            return q_values.argmax(dim=-1)
    
    def save(self, filename):
        torch.save(self.model.state_dict(), filename + "_model.pt")
        torch.save(self.optimizer.state_dict(), filename + "_optimizer")
        # if self.auto_alpha:
        #     torch.save(self.log_alpha, filename + "_log_alpha.pt")

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename + "_model.pt", map_location=self.device))
        self.target_model = copy.deepcopy(self.model)
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer", map_location=self.device))