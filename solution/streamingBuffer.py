import os
import pickle
from torch.utils.data import Dataset
from collections import OrderedDict
import torch
import numpy as np

class StreamingReplayDataset(Dataset):
    def __init__(self, data_sources, cache_size=50):
        """
        data_sources: str, list of folders, or list of file paths
        cache_size: maximum number of recently visited .pkl files
        """
        if isinstance(data_sources, str):
            data_sources = [data_sources]

        # Construct all .pkl dataset paths
        self.data_paths = []
        for src in data_sources:
            if os.path.isdir(src):
                for fname in sorted(os.listdir(src)):
                    if fname.endswith(".pkl"):
                        self.data_paths.append(os.path.join(src, fname))
            elif os.path.isfile(src):
                self.data_paths.append(src)
            else:
                raise ValueError(f"Invalid data source: {src}")

        # Construct map: (file_idx, step_idx)
        self.index_map = []
        self.file_lengths = []
        for file_idx, path in enumerate(self.data_paths):
            with open(path, 'rb') as f:
                episode = pickle.load(f)
                length = len(episode)
                self.file_lengths.append(length)
                self.index_map.extend([(file_idx, i) for i in range(length)])

        # LRU cache，save the data that recently visited
        self.cache = OrderedDict()
        self.cache_size = cache_size

        # self.data_paths = data_paths
        # self.index_map = []
        # self.episodes = []

        # for ep_idx, path in enumerate(self.data_paths):
        #     with open(path, 'rb') as f:
        #         episode = pickle.load(f)
        #         self.episodes.append(episode)
        #         self.index_map.extend([(ep_idx, s_idx) for s_idx in range(len(episode))])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # ep_idx, step_idx = self.index_map[idx]
        # return self.episodes[ep_idx][step_idx]
        file_idx, step_idx = self.index_map[idx]
        path = self.data_paths[file_idx]

        # Read from cache
        if file_idx in self.cache:
            episode = self.cache[file_idx]
            # Move to end, as recently been used
            self.cache.move_to_end(file_idx)
        else:
            # Not in cache, read from file
            with open(path, 'rb') as f:
                episode = pickle.load(f)
            # Add to cache
            self.cache[file_idx] = episode
            if len(self.cache) > self.cache_size:
                # Pop out longtime not used files
                self.cache.popitem(last=False)

        return episode[step_idx]


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