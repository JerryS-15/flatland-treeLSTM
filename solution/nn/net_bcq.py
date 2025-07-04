import torch
import torch.nn as nn
import torch.nn.functional as F
from impl_config import FeatureParserConfig as fp
from impl_config import NetworkConfig as ns

from .TreeLSTM import TreeLSTM

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Transformer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.att_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )

    def forward(self, input):
        batch_size, n_agents, embedding_size = input.shape
        input = input.permute(1, 0, 2)  # agents, batch, embedding
        input = input.view(n_agents, -1, embedding_size)
        output, _ = self.attention(input, input, input)

        input = input.view(n_agents, -1, embedding_size)
        input = input.permute(1, 0, 2)
        output = output.view(n_agents, -1, embedding_size)
        output = output.permute(1, 0, 2)
        output = self.att_mlp(torch.cat([input, output], dim=-1))
        return output

class BCQNetwork(nn.Module):
    def __init__(self):
        super(BCQNetwork, self).__init__()
        self.tree_lstm = TreeLSTM(fp.node_sz, ns.tree_embedding_sz)
        self.attr_embedding = nn.Sequential(
            nn.Linear(fp.agent_attr, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, ns.hidden_sz),
            nn.GELU(),
        )
        self.transformer = nn.Sequential(
            Transformer(ns.hidden_sz + ns.tree_embedding_sz, 4),
            Transformer(ns.hidden_sz + ns.tree_embedding_sz, 4),
            Transformer(ns.hidden_sz + ns.tree_embedding_sz, 4),
        )
        self.q_net = nn.Sequential(
            nn.Linear(ns.hidden_sz * 2 + ns.tree_embedding_sz * 2, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, ns.hidden_sz),
            nn.GELU(),
            nn.Linear(ns.hidden_sz, fp.action_sz)  # <== Output Q(s, a) for all a
        )
        self.i_net = nn.Sequential(
            nn.Linear(ns.hidden_sz * 2 + ns.tree_embedding_sz * 2, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, ns.hidden_sz),
            nn.GELU(),
            nn.Linear(ns.hidden_sz, fp.action_sz)  # <== Output Q(s, a) for all a
        )
    
    def forward(self, agents_attr, forest, adjacency, node_order, edge_order):
        batch_size, n_agents, num_nodes, _ = forest.shape
        device = next(self.parameters()).device
        adjacency = self.modify_adjacency(adjacency, device)

        tree_embedding = self.tree_lstm(forest, adjacency, node_order, edge_order)
        tree_embedding = tree_embedding.unflatten(0, (batch_size, n_agents, num_nodes))[:, :, 0, :]

        agent_attr_embedding = self.attr_embedding(agents_attr)
        embedding = torch.cat([agent_attr_embedding, tree_embedding], dim=2)
        att_embedding = self.transformer(embedding)

        agent_embedding = torch.cat([embedding, att_embedding], dim=-1)

        i_logits = self.i_net(agent_embedding)
        log_probs = F.log_softmax(i_logits, dim=-1)

        q_values = self.q_net(agent_embedding)

        return q_values, log_probs, i_logits
    
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