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

        # Reshape to [B*T, N, ...] so that each timestep's batch is a separate item
        forest = forest.view(B * T, N, num_nodes, node_dim)
        adjacency = adjacency.view(B * T, N, -1, 2)
        node_order = node_order.view(B * T, N, -1)
        edge_order = edge_order.view(B * T, N, -1)

        # Process adjacency to avoid index collisions
        adjacency = self.modify_adjacency(adjacency, device)

        # Get tree embeddings: [B*T*N*num_nodes, hidden_dim]
        tree_emb = self.tree_lstm(forest, adjacency, node_order, edge_order)

        # Only keep root node embeddings: [B*T, N, hidden_dim]
        tree_emb = tree_emb.unflatten(0, (B * T, N, num_nodes))[:, :, 0, :]
        tree_emb = tree_emb.view(B, T, N, -1)  # [B, T, N, hidden_dim]

        # Process agent attributes (shared across time)
        agent_attr = self.attr_embedding(agent_attr)  # [B, N, hidden_dim]
        agent_attr = agent_attr.unsqueeze(1).repeat(1, T, 1, 1)  # [B, T, N, hidden_dim]

        # Concatenate tree embedding and attr embedding
        x = torch.cat([tree_emb, agent_attr], dim=-1)  # [B, T, N, hidden_dim * 2]

        # Apply transformer
        x = self.transformer(x)  # [B, T, N, hidden_dim]

        # Predict logits per agent
        logits = self.predictor(x)  # [B, T, N, num_actions]

        return logits
    
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
