import pickle
import numpy as np
import os
import random
import torch

class ReplayBufferDT:
    def __init__(self, max_len=None):
        self.trajectories = []
        self.max_len = max_len  # for truncating/padding T

    def load_from_folder(self, folder_path):
        # files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
        # print(f"Found {len(files)} episodes.")
        # for file in files:
        #     path = os.path.join(folder_path, file)
        #     with open(path, 'rb') as f:
        #         episode = pickle.load(f)
        #     self.add_episode(episode)
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        print(f"Loading {len(file_paths)} episode files...")
        for path in file_paths:
            with open(path, 'rb') as f:
                episode = pickle.load(f)
            self.add_episode(episode)

    def add_episode(self, episode_steps):
        traj = {
            "agent_attr": [],
            "forest": [],
            "adjacency": [],
            "node_order": [],
            "edge_order": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

        for obs, action, reward, _, done in episode_steps:
            traj["agent_attr"].append(obs[0]["agent_attr"])
            traj["forest"].append(obs[0]["forest"])
            traj["adjacency"].append(obs[0]["adjacency"])
            traj["node_order"].append(obs[0]["node_order"])
            traj["edge_order"].append(obs[0]["edge_order"])
            traj["actions"].append(list(action.values()))
            traj["rewards"].append(list(reward.values()))
            traj["dones"].append([done["__all__"]])

        self.trajectories.append(traj)

    def compute_rtgs(self, rewards, gamma=1.0):
        """
        Compute Return-to-Go for rewards: shape (T, N)
        """
        T, N = rewards.shape
        rtgs = np.zeros_like(rewards, dtype=np.float32)
        for n in range(N):
            running_rtg = 0
            for t in reversed(range(T)):
                running_rtg = rewards[t, n] + gamma * running_rtg
                rtgs[t, n] = running_rtg
        return rtgs

    def pad_sequence(self, seq_list, dtype, pad_shape):
        """
        Pad a list of arrays to shape (B, T, ...)
        """
        B = len(seq_list)
        out = np.zeros((B, self.max_len) + pad_shape, dtype=dtype)
        for i, seq in enumerate(seq_list):
            T = min(self.max_len, len(seq))
            out[i, :T] = seq[:T]
        return out

    def sample_batch(self, batch_size, gamma=1.0):
        batch = random.sample(self.trajectories, batch_size)

        # Infer dimensions
        N = len(batch[0]["actions"][0])
        if self.max_len is None:
            self.max_len = max(len(traj["actions"]) for traj in batch)

        def stack_and_pad(key, dtype=np.float32):
            return self.pad_sequence([np.array(traj[key]) for traj in batch], dtype, np.shape(batch[0][key][0]))

        agent_attr = stack_and_pad("agent_attr")
        forest = stack_and_pad("forest")
        adjacency = stack_and_pad("adjacency", dtype=np.int64)
        node_order = stack_and_pad("node_order", dtype=np.int64)
        edge_order = stack_and_pad("edge_order", dtype=np.int64)
        actions = stack_and_pad("actions", dtype=np.int64)
        rewards = stack_and_pad("rewards")
        dones = stack_and_pad("dones")

        # Compute RTG (Return to Go)
        rtgs = np.zeros_like(rewards, dtype=np.float32)
        for i in range(batch_size):
            T = len(batch[i]["rewards"])
            traj_r = np.array(batch[i]["rewards"])
            rtg = self.compute_rtgs(traj_r, gamma)
            rtgs[i, :T] = rtg[:self.max_len]

        # Create timestep indices
        timesteps = np.broadcast_to(np.arange(self.max_len), (batch_size, self.max_len)).copy()

        return {
            "agent_attr": torch.FloatTensor(agent_attr),
            "forest": torch.FloatTensor(forest),
            "adjacency": torch.LongTensor(adjacency),
            "node_order": torch.LongTensor(node_order),
            "edge_order": torch.LongTensor(edge_order),
            "actions": torch.LongTensor(actions),
            "rewards": torch.FloatTensor(rewards),
            "rtgs": torch.FloatTensor(rtgs),
            "dones": torch.FloatTensor(dones),
            "timesteps": torch.LongTensor(timesteps),
        }