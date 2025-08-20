import pickle
import argparse
from argparse import ArgumentParser
from time import sleep
import wandb
import time
import os
import copy
import pandas as pd
from tqdm import tqdm

from flatland.envs.rail_env import RailEnv
from flatland.envs.line_generators import Line
from flatland.envs.timetable_utils import Timetable
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland_cutils import TreeObsForRailEnv as TreeCutils

from flatland.envs.step_utils.states import TrainState, StateTransitionSignals
from flatland.envs.agent_utils import SpeedCounter

from eval_env import LocalTestEnvWrapper
from impl_config import FeatureParserConfig as fp
from plfActor import Actor
from utils import VideoWriter, debug_show
from flatland.utils.rendertools import RenderTool

from test_env_rebuild import load_env_data, rebuild_env_from_data

def rebuild_wrapped_env_from_data(data):
    rail = data["rail"]
    stations = data["stations"]
    agents = data["agent_info"]
    env_params = data["env_params"]
    seed = data["seed"]

    def custom_rail_generator(*args, **kwargs):
        return rail, {}

    def custom_line_generator(rail, number_of_agents, hints, num_resets, np_random):
        agent_positions = [agent["initial_position"] for agent in agents]
        agent_directions = [agent["direction"] for agent in agents]
        agent_targets = [agent["target"] for agent in agents]
        agent_speeds = [agent["speed"] for agent in agents]

        return Line(agent_positions, agent_directions, agent_targets, agent_speeds)

    env = RailEnv(
        width=env_params["width"],
        height=env_params["height"],
        number_of_agents=env_params["number_of_agents"],
        rail_generator=custom_rail_generator,
        line_generator=custom_line_generator,
        # malfunction_generator=None,
        malfunction_generator=ParamMalfunctionGen(
            MalfunctionParameters(
                malfunction_rate=env_params['malfunction_rate'],
                min_duration=env_params['min_duration'],
                max_duration=env_params['max_duration']
            )
        ),
        obs_builder_object=TreeCutils(fp.num_tree_obs_nodes, fp.tree_pred_path_depth),
        remove_agents_at_target=True,
        random_seed=seed
    )

    env_wrapper = LocalTestEnvWrapper(env)
        
    return env_wrapper

def get_or_actions(step):
    action_data = load_env_data(action_path)
    return action_data[step]

def inject_agent_states(env, agent_states, action_dict):
    """
    Sync agent states (from reproduction of v2.2.1) to current v3.0.15
    agent_states: List[Dict], with each dict including position, direction, status, etc
    """
    for i, agent in enumerate(env.agents):
        agent.state = TrainState.READY_TO_DEPART
        agent.position = None

    for i, agent_state in enumerate(agent_states):
        agent = env.agents[i]
        agent.direction = agent_state['direction']
        if agent_state['status'] ==  "READY_TO_DEPART":
            agent.state = TrainState.READY_TO_DEPART
            agent.position = None
            action_dict[i] = 0
        elif agent_state['status'] == "ACTIVE":
            agent.position = agent_state['position']
            if action_dict[i] == 4:
                agent.state = TrainState.STOPPED
            else:
                agent.state = TrainState.MOVING
        elif agent_state['status'] == "DONE_REMOVED":
            agent.position = agent_state['position']
            agent.state = TrainState.DONE
        else:
            pass

# def get_states(env, agent_states, action_dict):
#     for i, agent_state in enumerate(agent_states):
#         agent = env.agents[i]
#         if agent_state['status'] == "READY_TO_DEPART":
#             agent.state = TrainState.READY_TO_DEPART
#             agent.position = None
#             action_dict[i] = 0
#             print("########## Agent State Set! ##########")
#         else:
#             continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm-rewards", "-nr", action="store_true", help="if collect norm rewards for agent")
    parser.add_argument("--seed", default=1, type=int, help="Initial seed for data collection.") # seed=0 generate random env in v2.2.1
    parser.add_argument("--eps", default=100, type=int, help="Number of episodes to collect for dataset.")
    parser.add_argument("--n-agents", default=5, type=int, help="Number of agents for data collection.")
    args = parser.parse_args()

    seed_init = args.seed
    n_eps = args.eps
    n_agents = args.n_agents

    if args.norm_rewards:
        print("Use norm reward for data collection.")

    print("---------------------------------------")
    print(f"OR Data Collection Started for {n_agents} agents, {n_eps} episodes.")
    print("---------------------------------------")
    
    if args.norm_rewards:
        save_dir = f"./orData_agent_{n_agents}_normR"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = f"./orData_agent_{n_agents}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    print("Starting wandb, view at https://wandb.ai/")
    wandb.init(
		project='flatland-TreeLSTM', 
		name=f"OR-DATASET_{n_agents}agents_{n_eps}eps_{time.strftime('%m%d%H%M%S')}"
	)

    all_offline_data = []
    dataset_info = []
    all_save_path = f"{save_dir}/or_data_{n_agents}_agents_{n_eps}_episodes.pkl"

    for i in tqdm(range(0, n_eps), desc="Collect OR dataset"):
        seed = seed_init + i

        env_path = f"or_solution_data_agent_{n_agents}/env_data_v2_{seed}.pkl"
        action_path = f"or_solution_data_agent_{n_agents}/action_data_v2_{seed}.pkl"
        step_path = f"or_solution_data_agent_{n_agents}/step_data_v2_{seed}.pkl"

        save_path = f"{save_dir}/or_data_{seed}.pkl"

        env_data = load_env_data(env_path)
        env_wrapper = rebuild_wrapped_env_from_data(env_data)
        step_data = load_env_data(step_path)

        obs = env_wrapper.reset()
    
        for agent in env_wrapper.env.agents:
            agent.earliest_departure = 0

        default_actions = {handle: 0 for handle in range(env_wrapper.env.number_of_agents)}

        step = 0
        episode_data = []
        info_data = []
        while True:
            if step < 0:
                action = default_actions
            else:
                action = get_or_actions(step)
                agent_states = step_data[step]
                inject_agent_states(env_wrapper.env, agent_states, action)
            # print(f"{step}: {action}")
            next_obs, all_rewards, done, step_rewards = env_wrapper.step(action)
            done_dict = done.copy()

            episode_data.append((
                copy.deepcopy(obs),
                action,
                copy.deepcopy(all_rewards),
                copy.deepcopy(next_obs),
                done_dict,
            ))

            step += 1
            obs = next_obs

            if done["__all__"]:
                arrival_ratio, total_reward, norm_reward, agent_norm_reward = env_wrapper.final_metric()

                if args.norm_rewards:
                    last_entry = list(episode_data[-1])
                    last_entry[2] = copy.deepcopy(agent_norm_reward)  # Replace reward
                    episode_data[-1] = tuple(last_entry)
                
                print(f"TOTAL_REW: {total_reward}, NORM_REW: {norm_reward:.4f}, ARR_RATIO: {arrival_ratio*100:.2f}%, with {len(episode_data)} samples.")
                info_data.append((
                    i,
                    total_reward,
                    norm_reward,
                    arrival_ratio*100,
                    len(episode_data)
                ))

                wandb.log({"Total Reward": total_reward, "Episode": i+1})
                wandb.log({"Normalized Reward": norm_reward, "Episode": i+1})
                wandb.log({"Arrival Ratio %": arrival_ratio*100, "Episode": i+1})
                wandb.log({"Number of Samples": len(episode_data), "Episode": i+1})
                break
        
        all_offline_data.extend(episode_data)
        dataset_info.extend(info_data)

        with open(save_path, "wb") as f:
            pickle.dump(episode_data, f)
        
    with open(all_save_path, "wb") as f:
        pickle.dump(all_offline_data, f)
    print("✅ Offline RL data is saved at ", all_save_path)

    dataset_info_path = f"{save_dir}/INFO-or_dataset.csv"
    df_info = pd.DataFrame(dataset_info, columns=[
        "episode", "total_reward", "norm_reward", "arrival_ratio", "num_samples_per_episode"
    ])  
    df_info.to_csv(dataset_info_path, index=False)
    print("Dataset INFO documented at ", dataset_info_path)
