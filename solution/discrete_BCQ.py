import numpy as np
import torch
import torch.nn.functional as F

from nn.net_bcq import BCQNetwork
import copy

class MultiAgentDiscreteBCQ:
    def __init__(
        self,
        num_actions,
        device,
        discount=0.99,
        tau=0.005,
        bcq_threshold=0.3,
        lr=3e-4,
        polyak_target_update=False,
        target_update_freq=1000
    ):
        self.device = device

        self.num_actions = num_actions
        self.discount = discount
        self.tau = tau
        self.threshold = bcq_threshold
        self.polyak_target_update = polyak_target_update
        self.target_update_freq = target_update_freq

        self.Q = BCQNetwork().to(device)
        self.Q2 = BCQNetwork().to(device)
        self.Q_target = copy.deepcopy(self.Q).to(device)
        self.Q2_target = copy.deepcopy(self.Q2).to(device)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

        self.iterations = 0
        self.last_metrics = {}
    
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

        with torch.no_grad():
            q_next, imt_next_log, _ = self.Q(next_agents_attr, next_forest, next_adjacency, next_node_order, next_edge_order)
            imt = imt_next_log.exp()
            mask = (imt / imt.max(dim=-1, keepdim=True)[0]) > self.threshold
            q_masked = mask.float() * q_next + (~mask).float() * -1e8
            next_action = q_masked.argmax(dim=-1, keepdim=True)

            q_target, _, _ = torch.min(
                self.Q_target(next_agents_attr, next_forest, next_adjacency, next_node_order, next_edge_order),
                self.Q2_target(next_agents_attr, next_forest, next_adjacency, next_node_order, next_edge_order)
            )
            max_next_q = q_target.gather(-1, next_action).squeeze(-1)
            target_q = rewards + (1.0 - dones) * self.discount * max_next_q
        
        q_pred, imt_log, i_logits = self.Q(agents_attr, forest, adjacency, node_order, edge_order)
        current_q = q_pred.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_loss = F.smooth_l1_loss(current_q, target_q)
        i_loss = F.nll_loss(imt_log.view(-1, self.num_actions), actions.view(-1))
        total_loss = q_loss + i_loss + 1e-2 * i_logits.pow(2).mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.iterations += 1
        if self.polyak_target_update:
            for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            if self.iterations % self.target_update_freq == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())
        
        self.last_metrics = {
            "total_loss": total_loss.item(),
            "q_loss": q_loss.item(),
            "i_loss": i_loss.item(),
            "q_values_mean": current_q.mean().item(),
            "target_q_mean": target_q.mean().item(),
            "imt_max": imt_log.exp().max(dim=-1)[0].mean().item(),
            "unlikely_actions_ratio": (imt_log.exp() < self.threshold).float().mean().item()
        }

        return self.last_metrics
    
    def select_action(self, agent_attr, forest, adjacency, node_order, edge_order, eval=False):
        with torch.no_grad():
            q, imt_log, _ = self.Q(agent_attr, forest, adjacency, node_order, edge_order)
            imt = imt_log.exp()
            mask = (imt / imt.max(dim=-1, keepdim=True)[0]) > self.threshold
            masked_q = mask.float() * q + (~mask).float() * -1e8
            return masked_q.argmax(dim=-1)
    
    def save(self, filename):
        torch.save({
            "Q": self.Q.state_dict(),
            "Q_target": self.Q_target.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, filename + "_model.pt")
        # torch.save(self.optimizer.state_dict(), filename + "_optimizer")
    
    def load(self, filename):
        checkpoint = torch.load(filename + "_model.pt", map_location=self.device)
        self.Q.load_state_dict(checkpoint["Q"])
        self.Q_target.load_state_dict(checkpoint["Q_target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
