import numpy as np
import torch

from nn.fc_bcq import BCQNetwork


class Actor:
    def __init__(self, model_path, threshold=0.3) -> None:
        self.net = BCQNetwork()
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        if 'Q' in state_dict:
            state_dict = state_dict['Q']
        self.net.load_state_dict(state_dict, strict=False)
        # self.net.load_state_dict(
        #     torch.load(model_path, map_location=torch.device("cpu"))
        # )
        self.net.eval()
        self.threshold = threshold

    def get_actions(self, obs_list, valid_actions, n_agents):
        feature = self.get_feature(obs_list)
        with torch.no_grad():
            q_values, log_imt, _ = self.net(*feature)
            q_values = q_values[0].cpu().numpy()
            imt = log_imt.exp()[0].cpu().numpy()
        
        actions = dict()
        valid_actions = np.array(valid_actions)
        for i in range(n_agents):
            mask = (imt[i] / imt[i].max()) > self.threshold
            valid_mask = valid_actions[i].astype(bool)

            filtered_logits = q_values[i].copy()
            filtered_logits[~(mask & valid_mask)] = -1e8

            if valid_mask.sum() == 0:
                actions[i] = np.random.randint(len(valid_actions[i]))
            else:
                actions[i] = int(np.argmax(filtered_logits))
        return actions

    def _choose_action(self, valid_actions, logits, soft_or_hard_max="soft"):
        def _softmax(x):
            if x.size != 0:
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()
            else:
                return None

        _logits = _softmax(logits[valid_actions != 0])
        if soft_or_hard_max == "soft":
            if valid_actions.nonzero()[0].size == 0:
                valid_actions = np.ones((1, 5))
            np.random.seed(42)
            action = np.random.choice(valid_actions.nonzero()[0], p=_logits)
        else:
            action = valid_actions.nonzero()[0][np.argmax(_logits)]
        return action

    def get_feature(self, obs_list):
        agents_attr = obs_list[0]["agent_attr"]
        agents_attr = torch.unsqueeze(torch.from_numpy(agents_attr), axis=0).to(
            dtype=torch.float32
        )

        forest = obs_list[0]["forest"]
        forest = torch.unsqueeze(torch.from_numpy(forest), axis=0).to(
            dtype=torch.float32
        )

        # adjacency = obs_list[0]["adjacency"]
        # adjacency = torch.unsqueeze(torch.from_numpy(adjacency), axis=0).to(
        #     dtype=torch.int64
        # )

        # node_order = obs_list[0]["node_order"]
        # node_order = torch.unsqueeze(torch.from_numpy(node_order), axis=0).to(
        #     dtype=torch.int64
        # )

        # edge_order = obs_list[0]["edge_order"]
        # edge_order = torch.unsqueeze(torch.from_numpy(edge_order), axis=0).to(
        #     dtype=torch.int64
        # )

        return agents_attr, forest
