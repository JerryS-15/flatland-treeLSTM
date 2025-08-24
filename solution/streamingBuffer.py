import os
import pickle
from torch.utils.data import Dataset
import torch
import numpy as np

class StreamingReplayDataset(Dataset):
    def __init__(self, data_dir):
        self.data_paths = [
            os.path.join(data_dir, fname)
            for fname in sorted(os.listdir(data_dir))
            if fname.endswith('.pkl')
        ]
        self.index_map = []
        self.episodes = []

        for ep_idx, path in enumerate(self.data_paths):
            with open(path, 'rb') as f:
                episode = pickle.load(f)
                self.episodes.append(episode)
                self.index_map.extend([(ep_idx, s_idx) for s_idx in range(len(episode))])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ep_idx, step_idx = self.index_map[idx]
        return self.episodes[ep_idx][step_idx]


def collate_transitions(batch):
    obs, action, all_rewards, next_obs, done = zip(*batch)

    agent_attr = np.array([o[0]["agent_attr"] for o in obs])
    forest = np.array([o[0]["forest"] for o in obs])
    adjacency = np.array([o[0]["adjacency"] for o in obs])
    node_order = np.array([o[0]["node_order"] for o in obs])
    edge_order = np.array([o[0]["edge_order"] for o in obs])
    actions = np.array([list(a.values()) for a in action])
    all_rewards = np.array([list(r.values()) for r in all_rewards])
    dones = np.array([
        [v for k, v in d.items() if k == "__all__"]
        for d in done
    ])

    next_agent_attr = np.array([o[0]["agent_attr"] for o in next_obs])
    next_forest = np.array([o[0]["forest"] for o in next_obs])
    next_adjacency = np.array([o[0]["adjacency"] for o in next_obs])
    next_node_order = np.array([o[0]["node_order"] for o in next_obs])
    next_edge_order = np.array([o[0]["edge_order"] for o in next_obs])

    return {
        "agent_attr": torch.FloatTensor(agent_attr),
        "forest": torch.FloatTensor(forest),
        "adjacency": torch.FloatTensor(adjacency),
        "node_order": torch.FloatTensor(node_order),
        "edge_order": torch.FloatTensor(edge_order),
        "actions": torch.LongTensor(actions),
        "all_rewards": torch.LongTensor(all_rewards),
        "dones": torch.LongTensor(dones),
        "next_agent_attr": torch.FloatTensor(next_agent_attr),
        "next_forest": torch.FloatTensor(next_forest),
        "next_adjacency": torch.FloatTensor(next_adjacency),
        "next_node_order": torch.FloatTensor(next_node_order),
        "next_edge_order": torch.FloatTensor(next_edge_order),
    }