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
        self.embed_action = nn.Embedding(act_dim, embed_dim) if fp.action_discrete else nn.Linear(act_dim, embed_dim)
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
        - node_order, edge_order: shape matching TreeLSTM
        - actions:      [B, T, N_agents]
        - rtgs:         [B, T, N_agents]
        - timesteps:    [B, T]
        """
        B, T, N, *_ = forest.shape
        device = forest.device

        # Flatten batch+time+agent for TreeLSTM
        forest = forest.view(B * T * N, fp.max_nodes, -1)
        adjacency = adjacency.view(B * T * N, -1, 2)
        node_order = node_order.view(B * T * N, -1)
        edge_order = edge_order.view(B * T * N, -1)

        tree_emb = self.tree_lstm(forest, adjacency, node_order, edge_order)  # [B*T*N, node_dim]
        tree_emb = tree_emb[:, 0, :]  # assume root is first node
        tree_emb = tree_emb.view(B, T, N, -1)

        attr_emb = self.attr_embedding(agent_attr)  # [B, T, N, dim]
        state_emb = self.state_projector(torch.cat([tree_emb, attr_emb], dim=-1))  # [B, T, N, state_dim]

        # Embed each modality
        s = self.embed_state(state_emb)                         # [B, T, N, D]
        a = self.embed_action(actions)                          # [B, T, N, D]
        r = self.embed_rtg(rtgs.unsqueeze(-1))                  # [B, T, N, D]
        t = self.embed_timestep(timesteps).unsqueeze(2).expand(-1, -1, N, -1)  # [B, T, N, D]

        # Interleave input as: [R_0, S_0, A_0, R_1, S_1, A_1, ...]
        x = torch.stack([r, s, a], dim=3).reshape(B, T * 3, N, self.embed_dim)  # [B, T*3, N, D]
        x = x.permute(0, 2, 1, 3).reshape(B * N, T * 3, self.embed_dim)         # [B*N, L, D]

        # Apply transformer
        h = self.transformer(x)  # [B*N, L, D]

        # Predict action from last state (e.g., token at position -2 or -1)
        # For simplicity, predict next action from last position:
        act_preds = self.predict_action(h[:, -1, :])  # [B*N, act_dim]
        return act_preds.view(B, N, -1)  # [B, N, act_dim]
