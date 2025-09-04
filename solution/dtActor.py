import numpy as np
import torch

from nn.net_dt import DecisionTransformer  # 或你定义的路径

class Actor:
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)
        self.net = DecisionTransformer().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def get_actions(self, obs_list, valid_actions, n_agents):
        # Step 1: 构造输入
        agents_attr, forest, adjacency, node_order, edge_order = self.get_feature(obs_list)

        # 创建 dummy timestep、RTG、actions 输入
        # （评估时只看当前时刻，输入长度设为 1）
        B = 1
        T = 1
        dummy_timesteps = torch.zeros((B, T, n_agents), dtype=torch.long, device=self.device)
        dummy_rtgs = torch.ones((B, T+1, n_agents), dtype=torch.float32, device=self.device) * 10.0  # 假设目标为10

        dummy_actions = torch.zeros((B, T, n_agents), dtype=torch.long, device=self.device)

        # Step 2: Forward
        with torch.no_grad():
            logits = self.net(
                agent_attr=agents_attr,
                forest=forest,
                adjacency=adjacency,
                node_order=node_order,
                edge_order=edge_order,
                actions=dummy_actions,
                rtgs=dummy_rtgs,
                timesteps=dummy_timesteps
            )
            logits = logits[:, -1]  # [B, n_agents, num_actions] 只取当前时刻动作分布

        logits = logits[0].cpu().numpy()  # [n_agents, num_actions]
        valid_actions = np.array(valid_actions)

        actions = dict()
        for i in range(n_agents):
            valid_mask = valid_actions[i].astype(bool)
            filtered_logits = logits[i].copy()
            filtered_logits[~valid_mask] = -1e8

            if valid_mask.sum() == 0:
                actions[i] = np.random.randint(len(valid_actions[i]))
            else:
                actions[i] = int(np.argmax(filtered_logits))
        return actions

    def get_feature(self, obs_list):
        obs = obs_list[0]  # eval mode: batch size = 1
        agents_attr = torch.unsqueeze(torch.from_numpy(obs["agent_attr"]), axis=0).float().to(self.device)
        forest = torch.unsqueeze(torch.from_numpy(obs["forest"]), axis=0).float().to(self.device)
        adjacency = torch.unsqueeze(torch.from_numpy(obs["adjacency"]), axis=0).long().to(self.device)
        node_order = torch.unsqueeze(torch.from_numpy(obs["node_order"]), axis=0).long().to(self.device)
        edge_order = torch.unsqueeze(torch.from_numpy(obs["edge_order"]), axis=0).long().to(self.device)

        return agents_attr, forest, adjacency, node_order, edge_order
