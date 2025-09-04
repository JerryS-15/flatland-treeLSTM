import torch
import torch.nn.functional as F

from nn.net_dt import DecisionTransformer
import copy


class MultiAgentDecisionTransformer:
    def __init__(
        self,
        num_actions,
        device,
        discount=0.99,
        lr=3e-4,
        max_timestep=1024
    ):
        self.device = device
        self.num_actions = num_actions
        self.discount = discount

        self.model = DecisionTransformer(
            state_dim=128,
            act_dim=num_actions,
            embed_dim=128,
            n_layers=3,
            n_heads=4,
            dropout=0.1,
            max_timestep=max_timestep
        ).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.iterations = 0
        self.last_metrics = {}

    def train(self, batch):
        device = self.device

        forest = batch['forest'].clone().detach().to(device)                 # [B, T, N, nodes, feat]
        agent_attr = batch['agent_attr'].clone().detach().to(device)         # [B, T, N, attr]
        adjacency = batch['adjacency'].clone().detach().long().to(device)
        node_order = batch['node_order'].clone().detach().long().to(device)
        edge_order = batch['edge_order'].clone().detach().long().to(device)

        actions = batch['actions'].clone().detach().to(device)      # [B, T, N]
        rtgs = batch['rtgs'].clone().detach().to(device)            # [B, T, N]
        timesteps = batch['timesteps'].clone().detach().to(device)    # [B, T]

        # 模型预测
        pred_logits = self.model(
            forest, agent_attr, adjacency,
            node_order, edge_order,
            actions, rtgs, timesteps
        )  # [B, N, act_dim]

        # 计算 Loss（行为克隆）
        last_actions = actions[:, -1, :]        # [B, N]
        pred_logits = pred_logits.view(-1, self.num_actions)
        target_actions = last_actions.view(-1)

        loss = F.cross_entropy(pred_logits, target_actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.iterations += 1
        self.last_metrics = {
            "loss": loss.item()
        }

        return self.last_metrics

    def select_action(self, batch, eval=False):
        """
        batch: same format as in training (single-timestep at end)
        returns: [B, N] actions
        """
        with torch.no_grad():
            forest = batch['forest'].clone().detach().to(self.device)
            agent_attr = batch['agent_attr'].clone().detach().to(self.device)
            adjacency = batch['adjacency'].clone().detach().long().to(self.device)
            node_order = batch['node_order'].clone().detach().long().to(self.device)
            edge_order = batch['edge_order'].clone().detach().long().to(self.device)
            actions = batch['actions'].clone().detach().to(self.device)
            rtgs = batch['rtgs'].clone().detach().to(self.device)
            timesteps = batch['timesteps'].clone().detach().to(self.device)

            pred_logits = self.model(
                forest, agent_attr, adjacency,
                node_order, edge_order,
                actions, rtgs, timesteps
            )  # [B, N, act_dim]

            return pred_logits.argmax(dim=-1)  # [B, N]

    def save(self, filename):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, filename + "_dt_model.pt")

    def load(self, filename):
        checkpoint = torch.load(filename + "_dt_model.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
