import torch
import torch.nn as nn
from impl_config import FeatureParserConfig as fp
from impl_config import NetworkConfig as ns


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Transformer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.att_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        """
        Input:
            x: [batch, n_agents, embed_dim]
        """
        B, N, D = x.shape

        # → MultiheadAttention expects [seq_len, batch, dim]
        inp = x.permute(1, 0, 2)

        out, _ = self.attention(inp, inp, inp)
        out = out.permute(1, 0, 2)

        # concat + MLP
        return self.att_mlp(torch.cat([x, out], dim=-1))

class CQLNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.attr_embedding = MLPEncoder(
            input_dim=fp.agent_attr,
            hidden_dim=2 * ns.hidden_sz,
            output_dim=ns.hidden_sz,
        )
        self.forest_encoder = MLPEncoder(
            input_dim=fp.node_sz * fp.max_nodes,      # flatten forest
            hidden_dim=2 * ns.hidden_sz,
            output_dim=ns.tree_embedding_sz,
        )
        embed_dim = ns.hidden_sz + ns.tree_embedding_sz
        self.transformer = nn.Sequential(
            Transformer(embed_dim, num_heads=4),
            Transformer(embed_dim, num_heads=4),
            Transformer(embed_dim, num_heads=4),
        )
        self.q_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, ns.hidden_sz),
            nn.GELU(),
            nn.Linear(ns.hidden_sz, fp.action_sz),  # Q(s,a)
        )


    def forward(self, agents_attr, forest):

        """
        agents_attr: [batch, n_agents, fp.agent_attr]
        forest:      [batch, n_agents, num_nodes, fp.node_sz]
        """

        B, N, num_nodes, node_dim = forest.shape

        # ---- Encode agent attributes ----
        attr_emb = self.attr_embedding(agents_attr)

        # ---- Encode forest using MLP instead of TreeLSTM ----
        forest_flat = forest.reshape(B * N, num_nodes * node_dim)
        tree_emb = self.forest_encoder(forest_flat)
        tree_emb = tree_emb.reshape(B, N, -1)

        # ---- Combine embeddings ----
        embedding = torch.cat([attr_emb, tree_emb], dim=2)

        # ---- Transformer ----
        att_embedding = self.transformer(embedding)

        # ---- Final Q ----
        combined = torch.cat([embedding, att_embedding], dim=-1)
        q_values = self.q_net(combined)

        return q_values