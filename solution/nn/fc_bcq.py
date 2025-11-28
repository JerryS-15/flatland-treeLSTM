import torch
import torch.nn as nn
import torch.nn.functional as F
from impl_config import FeatureParserConfig as fp
from impl_config import NetworkConfig as ns

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
        output, _ = self.attention(input, input, input)
        input = input.permute(1, 0, 2)
        output = output.permute(1, 0, 2)
        output = self.att_mlp(torch.cat([input, output], dim=-1))
        return output

class BCQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # agent 属性 embedding
        self.attr_embedding = nn.Sequential(
            nn.Linear(fp.agent_attr, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, ns.hidden_sz),
            nn.GELU(),
        )

        # agent 状态 embedding，代替 tree embedding
        self.state_embedding = nn.Sequential(
            nn.Linear(fp.node_sz, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, ns.hidden_sz),
            nn.GELU(),
        )

        # Transformer 层
        self.transformer = nn.Sequential(
            Transformer(ns.hidden_sz * 2, 4),
            Transformer(ns.hidden_sz * 2, 4),
            Transformer(ns.hidden_sz * 2, 4),
        )

        # Q 网络
        self.q_net = nn.Sequential(
            nn.Linear(ns.hidden_sz * 4, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, ns.hidden_sz),
            nn.GELU(),
            nn.Linear(ns.hidden_sz, fp.action_sz),
        )

        # I 网络（行为策略网络）
        self.i_net = nn.Sequential(
            nn.Linear(ns.hidden_sz * 4, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, ns.hidden_sz),
            nn.GELU(),
            nn.Linear(ns.hidden_sz, fp.action_sz),
        )

    def forward(self, agents_attr, agents_state):
        """
        agents_attr: [batch_size, n_agents, agent_attr]
        agents_state: [batch_size, n_agents, node_sz]  # 代替 forest 节点信息
        """
        agent_attr_emb = self.attr_embedding(agents_attr)
        agent_state_emb = self.state_embedding(agents_state)

        embedding = torch.cat([agent_attr_emb, agent_state_emb], dim=-1)
        att_embedding = self.transformer(embedding)

        combined = torch.cat([embedding, att_embedding], dim=-1)

        i_logits = self.i_net(combined)
        log_probs = F.log_softmax(i_logits, dim=-1)

        q_values = self.q_net(combined)

        return q_values, log_probs, i_logits