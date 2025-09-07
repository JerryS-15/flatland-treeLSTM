import numpy as np
import torch
import torch.nn.functional as F

from nn.net_bcq import BCQNetwork

class MultiAgentBehaviorCloning:
    def __init__(
        self,
        num_actions,
        device,
        lr=3e-4
    ):
        self.device = device
        self.num_actions = num_actions

        self.policy = BCQNetwork().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

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

        # Forward pass through policy network
        _, imt_log, i_logits = self.policy(agents_attr, forest, adjacency, node_order, edge_order)

        # Behavior Cloning loss
        i_loss = F.nll_loss(imt_log.view(-1, self.num_actions), actions.view(-1))
        total_loss = i_loss + 1e-2 * i_logits.pow(2).mean()

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.iterations += 1

        self.last_metrics = {
            "total_loss": total_loss.item(),
            "i_loss": i_loss.item(),
            "imt_max": imt_log.exp().max(dim=-1)[0].mean().item(),
        }

        return self.last_metrics
    
    def select_action(self, agent_attr, forest, adjacency, node_order, edge_order, eval=False):
        with torch.no_grad():
            _, imt_log, _ = self.policy(agent_attr, forest, adjacency, node_order, edge_order)
            return imt_log.argmax(dim=-1)  # use imitation policy directly
    
    def save(self, filename):
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, filename + "_model.pt")
    
    def load(self, filename):
        checkpoint = torch.load(filename + "_model.pt", map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])