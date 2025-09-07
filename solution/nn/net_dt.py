import torch
import torch.nn as nn
import torch.nn.functional as F

from impl_config import FeatureParserConfig as fp
from impl_config import NetworkConfig as ns
from .TreeLSTM import TreeLSTM


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, embed_dim=128, n_layers=3, n_heads=4, dropout=0.1, max_timestep=1024):
        super(DecisionTransformer, self).__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.embed_dim = embed_dim

        # ---- Graph + attribute encoders ----
        self.tree_lstm = TreeLSTM(fp.node_sz, ns.tree_embedding_sz)

        self.attr_embedding = nn.Sequential(
            nn.Linear(fp.agent_attr, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, ns.hidden_sz),
            nn.GELU(),
        )

        self.state_projector = nn.Linear(ns.tree_embedding_sz + ns.hidden_sz, state_dim)

        # ---- Embedding layers for sequence ----
        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_action = nn.Embedding(act_dim, embed_dim)
        # self.embed_action = nn.Embedding(act_dim, embed_dim) if fp.action_discrete else nn.Linear(act_dim, embed_dim)
        self.embed_rtg = nn.Linear(1, embed_dim)
        self.embed_timestep = nn.Embedding(max_timestep, embed_dim)

        # ---- Positional transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ---- Output head ----
        self.predict_action = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, act_dim)
        )

    def forward(self, forest, agent_attr, adjacency, node_order, edge_order,
                actions, rtgs, timesteps):
        """
        Inputs:
        - forest:       [B, T, N_agents, N_nodes, node_dim]
        - agent_attr:   [B, T, N_agents, agent_attr]
        - adjacency:    [B, T, N_agents, E, 2]
        - node_order, edge_order: [B, T, N_agents]
        - actions:      [B, T, N_agents]
        - rtgs:         [B, T, N_agents]
        - timesteps:    [B, T]
        """
        B, T, N, num_nodes, node_dim = forest.shape
        device = next(self.parameters()).device

        # ---- flatten batch/time/agent for TreeLSTM ----
        forest_flat = forest.reshape(B * T * N, num_nodes, node_dim)
        adjacency_flat = adjacency.reshape(B * T * N, adjacency.shape[3], 2)
        node_order_flat = node_order.reshape(B * T * N, node_order.shape[2])
        edge_order_flat = edge_order.reshape(B * T * N, edge_order.shape[2])

        adjacency_flat = self.modify_adjacency(adjacency_flat, device)

        # TreeLSTM embedding
        tree_emb = self.tree_lstm(forest_flat, adjacency_flat, node_order_flat, edge_order_flat)
        # 输出 shape: [B*T*N*num_nodes, hidden_dim], 取 root 节点
        tree_emb = tree_emb.view(B, T, N, num_nodes, -1)[:, :, :, 0, :]  # [B, T, N, hidden_dim]

        # agent_attr embedding
        agent_attr_emb = self.attr_embedding(agent_attr)  # [B, T, N, hidden_dim] 或 [B, N, hidden_dim] 根据实际情况
        if agent_attr_emb.dim() == 3:  # [B, N, hidden_dim]
            agent_attr_emb = agent_attr_emb.unsqueeze(1).repeat(1, T, 1, 1)

        # concat tree_emb + agent_attr_emb
        state_emb = torch.cat([tree_emb, agent_attr_emb], dim=-1)  # [B, T, N, hidden*2]
        state_emb = self.state_projector(state_emb)                # [B, T, N, state_dim]

        # embed for transformer
        s = self.embed_state(state_emb)
        a = self.embed_action(actions)
        r = self.embed_rtg(rtgs.unsqueeze(-1))
        t = self.embed_timestep(timesteps).unsqueeze(2).expand(-1, -1, N, -1)

        # interleave R,S,A
        x = torch.stack([r, s, a], dim=3).reshape(B, T * 3, N, self.embed_dim)
        x = x.permute(0, 2, 1, 3).reshape(B * N, T * 3, self.embed_dim)

        # transformer
        h = self.transformer(x)

        # predict actions
        act_preds = self.predict_action(h[:, -1, :])
        return act_preds.view(B, N, -1)
    
    def modify_adjacency(self, adjacency, device):
        batch_size, n_agents, num_edges, _ = adjacency.shape
        num_nodes = num_edges + 1
        mask0_invalid = (adjacency[..., 0] < 0) | (adjacency[..., 0] >= num_nodes)
        mask1_invalid = (adjacency[..., 1] < 0) | (adjacency[..., 1] >= num_nodes)
        adjacency[..., 0][mask0_invalid] = -2
        adjacency[..., 1][mask1_invalid] = -2
        id_tree = torch.arange(0, batch_size * n_agents, device=device)
        id_nodes = id_tree.view(batch_size, n_agents, 1)
        adjacency[adjacency == -2] = (-batch_size * n_agents * num_nodes)
        adjacency[..., 0] += id_nodes * num_nodes
        adjacency[..., 1] += id_nodes * num_nodes
        adjacency[adjacency < 0] = -2
        return adjacency
