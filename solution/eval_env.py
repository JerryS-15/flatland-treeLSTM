from flatland.evaluators.client import FlatlandRemoteClient
from flatland.envs.rail_env import TrainState
from flatland_cutils import TreeObsForRailEnv as TreeCutils

import numpy as np
from impl_config import FeatureParserConfig as fp


class TestEnvWrapper:
    def __init__(self) -> None:
        self.remote_client = FlatlandRemoteClient()
        self.env = None

    def reset(self):
        feature, _ = self.remote_client.env_create(
            TreeCutils(fp.num_tree_obs_nodes, fp.tree_pred_path_depth)
        )
        self.env = self.remote_client.env
        if feature is False:
            return False
        self.env.number_of_agents = len(self.env.agents)
        self.update_obs_properties()
        stdobs = (feature, self.obs_properties)
        obs_list = [self.parse_features(*stdobs)]
        return obs_list

    def action_required(self):
        return {
            i: self.env.action_required(agent)
            for i, agent in enumerate(self.env.agents)
        }

    def parse_actions(self, actions):
        action_required = self.action_required()
        parsed_action = dict()
        for idx, act in actions.items():
            if action_required[idx]:
                parsed_action[idx] = act
        return parsed_action

    def step(self, actions):
        actions = self.parse_actions(actions)
        feature, reward, done, info = self.remote_client.env_step(actions)

        # print("\nreward in TestEnvWrapper():")
        # print(reward)

        self.update_obs_properties()
        stdobs = (feature, self.obs_properties)
        obs_list = [self.parse_features(*stdobs)]
        return obs_list, reward, done

    def get_valid_actions(self):  # actor id in plf
        valid_actions = self.obs_properties["valid_actions"]
        return valid_actions  # numpy.array

    def submit(self):
        self.remote_client.submit()

    def update_obs_properties(self):
        self.obs_properties = {}
        properties = self.env.obs_builder.get_properties()
        env_config, agents_properties, valid_actions = properties
        self.obs_properties.update(env_config)
        self.obs_properties.update(agents_properties)
        self.obs_properties["valid_actions"] = valid_actions

    def parse_features(self, feature, obs_properties):
        def _fill_feature(items: np.ndarray, max):
            shape = items.shape
            new_items = np.zeros((max, *shape[1:]))
            new_items[: shape[0], ...] = items
            return new_items

        feature_list = {}
        feature_list["agent_attr"] = np.array(feature[0])
        feature_list["forest"] = np.array(feature[1][0])
        feature_list["forest"][feature_list["forest"] == np.inf] = -1
        feature_list["adjacency"] = np.array(feature[1][1])
        feature_list["node_order"] = np.array(feature[1][2])
        feature_list["edge_order"] = np.array(feature[1][3])
        feature_list.update(obs_properties)
        return feature_list

    def final_metric(self):
        assert self.env.dones["__all__"]
        env = self.env

        n_arrival = 0
        for a in env.agents:
            if a.position is None and a.state != TrainState.READY_TO_DEPART:
                n_arrival += 1

        arrival_ratio = n_arrival / env.get_num_agents()
        total_reward = sum(list(env.rewards_dict.values()))
        # print("Rewards at final step: ")
        # print(env.rewards_dict)
        norm_reward = 1 + total_reward / env._max_episode_steps / env.get_num_agents()

        return arrival_ratio, total_reward, norm_reward
    
    # def get_step_reward(self):
    #     env = self.env
    #     step_reward = env.rewards_dict
    #     return step_reward


class LocalTestEnvWrapper(TestEnvWrapper):
    def __init__(self, env) -> None:
        self.env = env
        ''' Reward params '''
        self.prev_departed = 0
        self.prev_arrived = 0
        self.prev_deadlocks = 0
        self.prev_distance = []

    def reset(self):
        feature, _ = self.env.reset()
        self.update_obs_properties()
        stdobs = (feature, self.obs_properties)
        obs_list = [self.parse_features(*stdobs)]
        return obs_list

    def step(self, actions):
        actions = self.parse_actions(actions)
        feature, reward, done, info = self.env.step(actions)

        ''' Reward Customize '''
        stdobs = (feature, self.obs_properties)
        obs_list = [self.parse_features(*stdobs)]
        obs = obs_list[0]
        # print("curr_step: ", obs['curr_step'])
        if obs['curr_step'] == 0:
            self.prev_distance = [0] * len(self.env.agents)
        # print("prev_distance: ", self.prev_distance)

        # r(e)_t
        env_reward = sum(self.env.rewards_dict.values())

        # r(d)_t
        num_departed = sum([1 for agent in self.env.agents if agent.state == TrainState.MOVING])
        departure_reward = num_departed - self.prev_departed

        # r(a)_t
        num_arrived = sum([1 for agent in self.env.agents if agent.position is None and agent.state != TrainState.READY_TO_DEPART])
        arrival_reward = num_arrived - self.prev_arrived

        # r(l)_t
        num_deadlocks = sum(obs['deadlocked'])
        deadlock_penalty = num_deadlocks - self.prev_deadlocks

        # agent specific moving reward
        # num_moving = sum([1 for agent in self.env.agents if agent.state == TrainState.MOVING])
        # moving_reward = num_moving / len(self.env.agents)

        # progress_rewards = [
        #     1.0 / (obs['dist_target'][i] + 1) if obs['dist_target'][i] > 0 else 2.0  
        #     for i in range(len(self.env.agents))
        # ]

        k = 0
        forward_rewards = [0] * len(self.env.agents)
        for agent in self.env.agents:
            if agent.state == TrainState.MOVING:
                forward_rewards[k] = 0.1
                k += 1
            else:
                forward_rewards[k] = 0
                k += 1


        # forward_rewards = [
        #     0.1 if obs['dist_target'][i] - self.prev_distance[i] < 0 else 0
        #     for i in range(len(self.env.agents))
        # ]

        progress_rewards = [
            0.0 if obs['dist_target'][i] > 0 else 2.0  
            for i in range(len(self.env.agents))
        ]

        cf, cp = 0.5, 0.7

        ce, ca, cd, cl = 0.0, 5.0, 1.0, 2.5
        step_reward = (ce * env_reward + ca * arrival_reward + cd * departure_reward - cl * deadlock_penalty)

        if obs['curr_step'] == 0:
            step_reward = 0

        custom_rewards = {
            i: step_reward + cf * forward_rewards[i] + cp * progress_rewards[i] for i in range(len(self.env.agents))
        }

        self.prev_departed = num_departed
        self.prev_arrived = num_arrived
        self.prev_deadlocks = num_deadlocks
        self.prev_distance = [
            obs['dist_target'][i]
            for i in range(len(self.env.agents))
        ]

        # self.env.rewards_dict = {i: step_reward for i in range(len(self.env.agents))}

        # reward = self.env.rewards_dict

        # print("R_step: ", step_reward)
        # print("R_forward: ", forward_rewards)
        # print("R_progress: ", progress_rewards)

        # print("\nCustom_reward in step():")
        # # # print(reward)
        # print(custom_rewards)
        # print("Rewards_dict: ")
        # print(self.env.rewards_dict)

        self.update_obs_properties()
        stdobs = (feature, self.obs_properties)
        obs_list = [self.parse_features(*stdobs)]
        return obs_list, reward, done, custom_rewards
