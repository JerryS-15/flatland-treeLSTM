import numpy as np
from time import sleep

from flatland.envs.line_generators import SparseLineGen
from flatland.envs.malfunction_generators import (
    MalfunctionParameters,
    ParamMalfunctionGen,
)
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import SparseRailGen
from flatland_cutils import TreeObsForRailEnv as TreeCutils

from eval_env import LocalTestEnvWrapper
from impl_config import FeatureParserConfig as fp
from utils import debug_show

def map_generator(env_params, seed, eval_episodes=10):
    """
    model: trained BC
    env_params: 
    seed: 
    eval_episodes: number of episodes for evaluation
    """
    eval_env = RailEnv(
        number_of_agents=env_params['number_of_agents'],
        width=env_params['width'],
        height=env_params['height'],
        rail_generator=SparseRailGen(
            max_num_cities=env_params['max_num_cities'],
            grid_mode=False,
            max_rails_between_cities=env_params['max_rails_between_cities'],
            max_rail_pairs_in_city=env_params['max_rail_pairs_in_city'],
        ),
        line_generator=SparseLineGen(
            speed_ratio_map=env_params['speed_ratio_map'],
        ),
        malfunction_generator=ParamMalfunctionGen(
            MalfunctionParameters(
                malfunction_rate=env_params['malfunction_rate'],
                min_duration=env_params['min_duration'],
                max_duration=env_params['max_duration']
            )
        ),
        obs_builder_object=TreeCutils(fp.num_tree_obs_nodes, fp.tree_pred_path_depth),
        random_seed=seed + 100
    )

    env_wrapper = LocalTestEnvWrapper(eval_env)
    n_agents = env_wrapper.env.number_of_agents

    # actor = Actor(model_path)
    # print(f"Evaluating using actor from {model_path}")

    total_rewards = []
    norm_rewards = []
    arrival_ratios = []

    # Keep actions always the same
    actions = {i: 0 for i in range(n_agents)}

    for ep in range(eval_episodes):
        obs = env_wrapper.reset()
        # done = {"__all__": False}
        # ep_reward = 0
        counter = 0

        if env_params['render'] == True:
            debug_show(env_wrapper.env)
            sleep(1 / env_params['fps'])

        print(f"Test Episode {ep+1}:")
        for i, agent in enumerate(eval_env.agents):
            print(f"Agent {i}: initial={agent.initial_position}, target={agent.target}")

        while counter < 11:
            counter += 1
            valid_actions = env_wrapper.get_valid_actions()
            # actions = actor.get_actions(obs, valid_actions, n_agents)
            obs, all_rewards, done, _ = env_wrapper.step(actions)
            # ep_reward += sum(all_rewards.values())
        
        print(f"Test Episode {ep+1} Finished.")
        
        # arrival_ratio, total_reward, norm_reward = env_wrapper.final_metric()
        # total_rewards.append(total_reward)
        # norm_rewards.append(norm_reward)
        # arrival_ratios.append(arrival_ratio)

        # print(f"[Test Episode {ep+1}] Total Reward: {total_reward}, Normalized Reward: {norm_reward:.4f}, Arrival Ratio: {arrival_ratio*100:.2f}%")
    
    # avg_total_reward = np.mean(total_rewards)
    # avg_norm_reward = np.mean(norm_rewards)
    # avg_arrival_ratio = np.mean(arrival_ratios)

    # print("---------------------------------------")
    # print(f"Evaluation over {eval_episodes} episodes:")
    # print(f"  Avg Total Reward: {avg_total_reward:.2f}")
    # print(f"  Avg Normalized Reward: {avg_norm_reward:.4f}")
    # print(f"  Avg Arrival Ratio: {avg_arrival_ratio*100:.2f}%")
    # print("---------------------------------------")

    # return avg_norm_reward

if __name__ == "__main__":

    flatland_parameters = {
		# Flatland Env
        "number_of_agents": 5,
        "width": 30,
        "height": 35,
        "max_num_cities": 3,
        "max_rails_between_cities": 2,
        "max_rail_pairs_in_city": 2,
        "speed_ratio_map": {1.0: 1 / 4, 0.5: 1 / 4, 0.33: 1 / 4, 0.25: 1 / 4},
        # Flatland - malfunction
        "malfunction_rate": 1 / 4500,
        "min_duration": 20,
        "max_duration": 50,
        # Rendering
        "fps": 30,
        "render": True
	}

    seed = 0

    print("------------------------------")
    map_generator(flatland_parameters, seed)
    print("------------------------------")