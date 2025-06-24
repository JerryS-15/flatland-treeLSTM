import numpy as np
import torch

from nn.net_newCQL import CQLNetwork


class Actor:
    def __init__(self, model_path) -> None:
        self.net = CQLNetwork()
        self.net.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        self.net.eval()

    def get_actions(self, obs_list, valid_actions, n_agents):
        feature = self.get_feature(obs_list)
        with torch.no_grad():
            q_values = self.net(*feature)  # shape: [batch_size=1, n_agents, num_actions]
        q_values = q_values.squeeze(0).cpu().numpy()  # shape: [n_agents, num_actions]

        actions = dict()
        valid_actions = np.array(valid_actions)
        for i in range(n_agents):
            actions[i] = self._choose_action(valid_actions[i, :], q_values[i, :])
        return actions

    def _choose_action(self, valid_actions, q_values, soft_or_hard_max="soft"):
        valid_indices = np.where(valid_actions != 0)[0]
        if len(valid_indices) == 0:
            # Random return an action if no valid exists
            return 0

        valid_q = q_values[valid_indices]

        def _softmax(x):
            if x.size != 0:
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()
            else:
                return None
        
        probs = _softmax(valid_q)

        if soft_or_hard_max == "soft":
            np.random.seed(42)
            chosen_idx = np.random.choice(len(valid_indices), p=probs)
        else:
            chosen_idx = np.argmax(probs)

        return valid_indices[chosen_idx]

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
