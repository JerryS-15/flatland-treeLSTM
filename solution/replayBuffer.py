import pickle
import numpy as np
import torch
import os
import random
    
class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, transition):
        obs, action, reward, next_obs, done = transition
        # print(f"========== Step ==========")
        # print("\nAction:", type(action))
        # print(action)
        # obs = self.flatten_obs(obs)
        # next_obs = self.flatten_obs(next_obs)
        # print("flattend obs: ", type(obs), len(obs))

        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, all_rewards, next_state, done = zip(*batch)
        obs = state
        next_obs = next_state
        # print("DEBUG----------state.shape: ", len(state))
        # print("DEBUG----------obs[0].shape: ", len(obs[0]))

        agent_attr = np.array([o[0]["agent_attr"] for o in obs])
        forest = np.array([o[0]["forest"] for o in obs])
        adjacency = np.array([o[0]["adjacency"] for o in obs])
        node_order = np.array([o[0]["node_order"] for o in obs])
        edge_order = np.array([o[0]["edge_order"] for o in obs])
        actions = np.array([list(a.values()) for a in action])
        all_rewards = np.array([list(r.values()) for r in all_rewards])
        # dones = np.array([list(d.values()) for d in done])
        dones = np.array([
            [v for k, v in d.items() if k == "__all__"]
            for d in done
        ])

        next_agent_attr = np.array([o[0]["agent_attr"] for o in next_obs])
        next_forest = np.array([o[0]["forest"] for o in next_obs])
        next_adjacency = np.array([o[0]["adjacency"] for o in next_obs])
        next_node_order = np.array([o[0]["node_order"] for o in next_obs])
        next_edge_order = np.array([o[0]["edge_order"] for o in next_obs])

        # print("DEBUG----------agent_attr.shape: ", agent_attr.shape)

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

    def load_from_file(self, file_paths):
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                raw_data = pickle.load(f)
            for transition in raw_data:
                self.add(transition)

        # with open(file_path, 'rb') as f:
        #     raw_data = pickle.load(f)
        # for transition in raw_data:
        #     self.add(transition)

        # for file_path in file_paths:
        #     with open(file_path, 'rb') as f:
        #         raw_data = pickle.load(f)
        #     for transition in raw_data:
        #         self.add(transition)
