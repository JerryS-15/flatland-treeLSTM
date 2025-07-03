import numpy as np
import torch
import torch.nn.functional as F

from nn.net_globalCQL import CQLNetwork
import copy

class MultiAgentGlobalDiscreteCQL:
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
        self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=lr)

        self.num_actions = num_actions
        self.alpha = alpha
        self.discount = discount
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.iterations = 0

    def train(self, batch):
        device = self.device

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

        logits = self.model.actor(agents_attr, forest, adjacency, node_order, edge_order)
        probs = F.log_softmax(logits, dim=-1)
        action_log_probs = torch.gather(probs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

        # Critic Loss with CQL
        with torch.no_grad():
            next_logits = self.model.actor(next_agents_attr, next_forest, next_adjacency, next_node_order, next_edge_order)
            next_probs = F.softmax(next_logits, dim=-1)
            next_actions = next_probs.argmax(dim=-1)
            next_q = self.target_model.critic(next_agents_attr, next_forest, next_adjacency, next_node_order, next_edge_order, next_actions)
            target_q = rewards + self.discount * (1 - dones) * next_q

            # next_q_values = self.target_model(next_agents_attr, next_forest, next_adjacency, next_node_order, next_edge_order)
            # max_next_q, _ = next_q_values.max(dim=-1, keepdim=True)
            # max_next_q = max_next_q.squeeze(-1)
            # # print("**********DEBUG**********")
            # # print("rewards.shape:", rewards.shape)
            # # print("dones.shape:", dones.shape)
            # # print("max_next_q.shape:", max_next_q.shape)
            # # print("**********DEBUG**********")
            # target_q = rewards + (1.0 - dones) * self.discount * max_next_q

        current_q = self.model.critic(agents_attr, forest, adjacency, node_order, edge_order, actions)
        bellman_loss = F.mse_loss(current_q, target_q)

        # CQL Penalty
        with torch.no_grad():
            all_logits = self.model.actor(agents_attr, forest, adjacency, node_order, edge_order, actions)
        all_probs = F.log_softmax(all_logits, dim=-1)
        num_actions = all_logits.size(-1)

        cql_qs = []

        for a in range(num_actions):
            a_batch = torch.full_like(actions, a)
            q_val = self.model.critic(agents_attr, forest, adjacency, node_order, edge_order, a_batch)
            cql_qs.append(q_val)
        cql_stack = torch.stack(cql_qs, dim=-1)
        logsumexp = torch.logsumexp(cql_stack, dim=-1)
        cql_penalty = (logsumexp - current_q).mean()

        critic_loss = bellman_loss + self.alpha * cql_penalty

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss: maximize critic(Q)
        logits = self.model.actor(agents_attr, forest, adjacency, node_order, edge_order)
        dist = F.log_softmax(logits, dim=-1)
        actions_pred = dist.exp().multinomial(num_samples=1).squeeze(-1)
        actor_q = self.model.critic(agents_attr, forest, adjacency, node_order, edge_order, actions_pred)
        actor_loss = -actor_q.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step() 

        if self.iterations % self.target_update_freq == 0:
            self._soft_update()

        self.iterations += 1

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'cql_penalty': cql_penalty.item(),
            'q_mean': current_q.mean().item(),
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
        torch.save(self.model.state_dict(), filename + "_global_model.pt")
        torch.save(self.optimizer.state_dict(), filename + "_global_optimizer")

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename + "_global_model.pt", map_location=self.device))
        self.target_model = copy.deepcopy(self.model)
        self.optimizer.load_state_dict(torch.load(filename + "_global_optimizer", map_location=self.device))
        