import numpy as np
import torch

from nn.bcq import QNetwork, BehaviorCloneNetwork


class Actor:
    def __init__(self, model_path) -> None:
        self.net = QNetwork(input_dim=30300, output_dim=50 * 5)
        self.net.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        self.net.eval()

    def get_actions(self, obs_list, valid_actions, n_agents):
        # obs = torch.from_numpy(np.array(obs)).float()
        feature = self.get_feature(obs_list)
        # print("feature: ", type(feature), len(feature))

        logits = self.net(feature)
        logits = logits.detach().numpy().reshape(n_agents, 5)
        actions = dict()
        valid_actions = np.array(valid_actions)
        # print("Dimension of valid_actions: ", valid_actions.shape)

        for i in range(n_agents):
            if n_agents == 1:
                actions[i] = self._choose_action(valid_actions[i, :], logits)
            else:
                actions[i] = self._choose_action(valid_actions[i, :], logits[i, :])
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
        # print("obs_list: ", len(obs_list))
        features = []

        for obs_dict in obs_list:
            agent_attr = np.array(obs_dict["agent_attr"], dtype=np.float32).flatten()
            forest = np.array(obs_dict["forest"], dtype=np.float32).reshape(50, -1).flatten()
            adjacency = np.array(obs_dict["adjacency"], dtype=np.float32).reshape(50, -1).flatten()
            node_order = np.array(obs_dict["node_order"], dtype=np.float32).flatten()
            edge_order = np.array(obs_dict["edge_order"], dtype=np.float32).flatten()
            feature = np.concatenate([agent_attr, forest, adjacency, node_order, edge_order], axis=0)
            features.append(feature)
        # print(f"Flattened features: {flat_features[:10]}...")
        return torch.FloatTensor(np.array(features, dtype=np.float32))
        # # 处理每个特征，确保它们是numpy数组，并且转换为tensor
        # agents_attr = torch.from_numpy(obs_list[0]["agent_attr"]).float()
        # forest = torch.from_numpy(obs_list[0]["forest"]).float()
        # adjacency = torch.from_numpy(obs_list[0]["adjacency"]).long()
        # node_order = torch.from_numpy(obs_list[0]["node_order"]).long()
        # edge_order = torch.from_numpy(obs_list[0]["edge_order"]).long()

        # # 进行拼接，假设我们要拼接这些特征形成一个大state
        # state = torch.cat([agents_attr, forest.flatten(start_dim=1), adjacency.flatten(start_dim=1),
        #         node_order.flatten(start_dim=1), edge_order.flatten(start_dim=1)], dim=1)
    
        # return state