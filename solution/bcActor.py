import numpy as np
import torch
from nn.net_bcq import BCQNetwork

class Actor:
    def __init__(self, model_path) -> None:
        self.net = BCQNetwork()
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        if "policy" in state_dict:
            state_dict = state_dict["policy"]
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def get_actions(self, obs_list, valid_actions, n_agents):
        feature = self.get_feature(obs_list)
        with torch.no_grad():
            _, log_imt, _ = self.net(*feature)
            imt_probs = log_imt.exp()[0].cpu().numpy()  # shape: [n_agents, num_actions]

        actions = dict()
        valid_actions = np.array(valid_actions)

        for i in range(n_agents):
            action_probs = imt_probs[i].copy()
            mask = valid_actions[i].astype(bool)
            action_probs[~mask] = 0.0

            if mask.sum() == 0:
                actions[i] = np.random.randint(len(valid_actions[i]))
            else:
                action_probs /= action_probs.sum()
                actions[i] = int(np.argmax(action_probs))  # greedy 选择
                # 如果需要 stochastic 策略，可以改为：
                # actions[i] = int(np.random.choice(len(action_probs), p=action_probs))
        return actions

    def get_feature(self, obs_list):
        agents_attr = obs_list[0]["agent_attr"]
        agents_attr = torch.unsqueeze(torch.from_numpy(agents_attr), axis=0).to(
            dtype=torch.float32
        )

        forest = obs_list[0]["forest"]
        forest = torch.unsqueeze(torch.from_numpy(forest), axis=0).to(
            dtype=torch.float32
        )

        adjacency = obs_list[0]["adjacency"]
        adjacency = torch.unsqueeze(torch.from_numpy(adjacency), axis=0).to(
            dtype=torch.int64
        )

        node_order = obs_list[0]["node_order"]
        node_order = torch.unsqueeze(torch.from_numpy(node_order), axis=0).to(
            dtype=torch.int64
        )

        edge_order = obs_list[0]["edge_order"]
        edge_order = torch.unsqueeze(torch.from_numpy(edge_order), axis=0).to(
            dtype=torch.int64
        )

        return agents_attr, forest, adjacency, node_order, edge_order