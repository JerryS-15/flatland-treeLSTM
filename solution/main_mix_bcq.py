import argparse
import copy
import importlib
import json
import os
from time import sleep

import numpy as np
import torch

import utils
import wandb
import time
from tqdm import tqdm

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
from bcqActor import Actor
from replayBuffer import ReplayBuffer

import discrete_BCQ


def train_BCQ(replay_buffer, data_file, num_actions, args, parameters):
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu:7")
    print(f"Using device: {device}")

    if args.use_or:
        policy_name = f"bcq-or-ALL-agents-normReward-bs{parameters['batch_size']}"
    elif args.use_mix:
        policy_name = f"bcq-mix-ALL-agents-normReward-bs{parameters['batch_size']}"
    else:
        policy_name = f"bcq-ALL-agents-normReward-bs{parameters['batch_size']}"
    policy_path = f"./policy/{policy_name}"

    policy = discrete_BCQ.MultiAgentDiscreteBCQ(
        num_actions,
        device=device
    )

    replay_buffer.load_from_file(data_file)
    print(f"Loaded {len(replay_buffer.buffer)} transitions.")

    evaluations_5 = []
    evaluations_10 = []
    evaluations_20 = []
    training_iters = 0

    print("BCQ Training started.")

    while training_iters < args.max_timesteps:
        epoch_metrics = []
        for ep in tqdm(range(int(parameters["eval_freq"])), desc="BCQ Training Progress"):
            batch = replay_buffer.sample(parameters["batch_size"])
            metrics = policy.train(batch)
            epoch_metrics.append(metrics)
            if ep % 1000 == 0:
                tqdm.write(f"Epoch {ep}, Loss: {metrics['total_loss']:.4f}, Q_Loss: {metrics['q_loss']:.4f}, i_Loss: {metrics['i_loss']:.4f}")
        
        avg_metrics = {
            k: np.mean([m[k] for m in epoch_metrics])
            for k in epoch_metrics[0].keys()
        }

        policy.save(policy_path)
        model_path = f"{policy_path}_model.pt"
        evaluations_5.append(eval_policy(model_path, 5, parameters, args.seed+2000))
        evaluations_10.append(eval_policy(model_path, 10, parameters, args.seed+1000))
        evaluations_20.append(eval_policy(model_path, 20, parameters, args.seed))
        np.save(f"./results/{policy_name}_5-agent", evaluations_5)
        np.save(f"./results/{policy_name}_10-agent", evaluations_10)
        np.save(f"./results/{policy_name}_20-agent", evaluations_20)

        wandb.log({
            "BCQ/total_loss": avg_metrics["total_loss"],
            "BCQ/q_loss": avg_metrics["q_loss"],
            "BCQ/i_loss": avg_metrics["i_loss"],
            "BCQ/q_values_mean": avg_metrics["q_values_mean"],
            "BCQ/target_q_mean": avg_metrics["target_q_mean"],
            "BCQ/imt_max": avg_metrics["imt_max"],
            "BCQ/unlikely_actions_ratio": avg_metrics["unlikely_actions_ratio"]
        }, step=training_iters)

        print(f"[BCQ] Iteration: {training_iters} | Total Loss: {avg_metrics['total_loss']:.3f}")

        training_iters += int(parameters["eval_freq"])
    
    policy.save(policy_path)


def eval_policy(model_path, num_agents, env_params, seed, eval_episodes=10):
    """
    model_path:
    env_params: 
    seed: 
    eval_episodes: number of episodes for evaluation
    """
    eval_env = RailEnv(
        number_of_agents=num_agents,
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
        # malfunction_generator=None,
        obs_builder_object=TreeCutils(fp.num_tree_obs_nodes, fp.tree_pred_path_depth),
        remove_agents_at_target=True,
        random_seed=seed + 100
    )

    env_wrapper = LocalTestEnvWrapper(eval_env)
    n_agents = env_wrapper.env.number_of_agents

    actor = Actor(model_path)
    print(f"Evaluating using actor from {model_path}")

    total_rewards = []
    norm_rewards = []
    arrival_ratios = []

    for ep in range(eval_episodes):
        eval_seed = seed + 100 + ep*10
        obs = env_wrapper.reset(random_seed=eval_seed)
        done = {"__all__": False}
        ep_reward = 0

        while not done["__all__"]:
            valid_actions = env_wrapper.get_valid_actions()
            actions = actor.get_actions(obs, valid_actions, n_agents)
            obs, all_rewards, done, _ = env_wrapper.step(actions)
            # ep_reward += sum(all_rewards.values())
        
        arrival_ratio, total_reward, norm_reward, _ = env_wrapper.final_metric()
        total_rewards.append(total_reward)
        norm_rewards.append(norm_reward)
        arrival_ratios.append(arrival_ratio)

        print(f"[Eval Episode {ep+1}] [Eval Seed {eval_seed}] Total Reward: {total_reward}, Normalized Reward: {norm_reward:.4f}, Arrival Ratio: {arrival_ratio*100:.2f}%")
        wandb.log({"Total Reward": total_reward, "Evaluation Episodes": ep+1})
        wandb.log({"Normalized Reward": norm_reward, "Evaluation Episodes": ep+1})
        wandb.log({"Arrival Ratio %": arrival_ratio*100, "Evaluation Episodes": ep+1})
    
    avg_total_reward = np.mean(total_rewards)
    avg_norm_reward = np.mean(norm_rewards)
    avg_arrival_ratio = np.mean(arrival_ratios)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes:")
    print(f"  Avg Total Reward: {avg_total_reward:.2f}")
    print(f"  Avg Normalized Reward: {avg_norm_reward:.4f}")
    print(f"  Avg Arrival Ratio: {avg_arrival_ratio*100:.2f}%")
    print("---------------------------------------")
    wandb.log({f"{num_agents}-agents Avg Total Reward": avg_total_reward, "Evaluation Episodes": eval_episodes})
    wandb.log({f"{num_agents}-agents Avg Norm Reward": avg_norm_reward, "Evaluation Episodes": eval_episodes})
    wandb.log({f"{num_agents}-agents Avg Arrival Ratio %": avg_arrival_ratio*100, "Evaluation Episodes": eval_episodes})

    return avg_norm_reward

def collect_pickle_paths(folder_path, max_files):
    paths = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".pkl"):
            paths.append(os.path.join(folder_path, file))
    if max_files is not None:
        paths = paths[:max_files]
    return paths
    # return [
    #     os.path.join(folder, f)
    #     for f in os.listdir(folder)
    #     if f.endswith(".pkl")
    # ]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # 1e6
    parser.add_argument("--seed", default=5000, type=int)
    # parser.add_argument("--data_n_eps", default=1000, type=int, help="Number of episodes that dataset have")
    # parser.add_argument("--normal_reward", action="store_true", help="Use dataset with normed_rewards for agent")
    parser.add_argument("--use_or", "-or", action="store_true", help="Train with or-solution dataset.")
    parser.add_argument("--batch_size", "-bs", default=128, type=int, help="Training batch size.")
    parser.add_argument("--use_mix", "-mix", action="store_true", help="Use mixed dataset for training.")

    args = parser.parse_args()

    batch_size = args.batch_size
    max_num_files_1 = 400
    max_num_files_2 = 300

    flatland_parameters = {
		# Evaluation
		"eval_freq": 1e4,  #5e4
		# "eval_eps": 1e-3,
		# Learning
		# "discount": 0.99,
		# "buffer_size": 1e6,
		"batch_size": batch_size,   # default setting - 128
		# "optimizer": "Adam",
		# "optimizer_parameters": {
		# 	"lr": 1e-4,   # 0.0000625
		# 	"eps": 0.00015
		# },
		# Flatland Env
        "number_of_agents": 5,
        "width": 30,
        "height": 35,
        "max_num_cities": 3,
        "max_rails_between_cities": 2,
        "max_rail_pairs_in_city": 2,
        "speed_ratio_map": {1.0: 1 / 3, 0.5: 1 / 3, 0.25: 1 / 3},
        # Flatland - malfunction
        "malfunction_rate": 1 / 4500,
        "min_duration": 20,
        "max_duration": 50
	}

    parameters = flatland_parameters

    num_actions = 5
    if args.use_or:
        or_folder_5 = f"./orData_agent_5_normR"
        or_folder_10 = f"./orData_agent_10_normR"
        or_folder_20 = f"./orData_agent_20_normR"
        data_file = collect_pickle_paths(or_folder_5, max_num_files_1) + collect_pickle_paths(or_folder_10, max_num_files_1) + collect_pickle_paths(or_folder_20, max_num_files_2)
    elif args.use_mix:
        data_folder_5_a = f"./orData_agent_5_normR"
        data_folder_5_b = f"./offlineData_5"
        data_folder_10_a = f"./orData_agent_10_normR"
        data_folder_10_b = f"./offlineData_10"
        data_folder_20_a = f"./orData_agent_20_normR"
        data_folder_20_b = f"./offlineData_20"
        data_file_5 = collect_pickle_paths(data_folder_5_a, int(max_num_files_1/2)) + collect_pickle_paths(data_folder_5_b, int(max_num_files_1/2))
        data_file_10 = collect_pickle_paths(data_folder_10_a, int(max_num_files_1/2)) + collect_pickle_paths(data_folder_10_b, int(max_num_files_1/2))
        data_file_20 = collect_pickle_paths(data_folder_20_a, int(max_num_files_2/2)) + collect_pickle_paths(data_folder_20_b, int(max_num_files_2/2))
        data_file = data_file_5 + data_file_10 + data_file_20
    else:
        data_folder_5_a = f"./offlineData_5"
        data_folder_5_b = f"./offlineData_5_noisy"
        data_folder_10_a = f"./offlineData_10"
        data_folder_10_b = f"./offlineData_10_noisy"
        data_folder_20_a = f"./offlineData_20"
        data_folder_20_b = f"./offlineData_20_noisy"
        data_file = collect_pickle_paths(data_folder_5_a, max_num_files_1) + collect_pickle_paths(data_folder_10_a, max_num_files_1) + collect_pickle_paths(data_folder_20_a, max_num_files_2)

    print("---------------------------------------")
    if args.use_or:
        print("Start BCQ training for flatland TreeLSTM with OR-Solution Dataset.")
        mode = "BCQ-OR"
    elif args.use_mix:
        print("Start BCQ training for flatland TreeLSTM with OR-RL Mixed Dataset.")
        mode = "BCQ-MIX"
    else:
        print("Start BCQ training for flatland TreeLSTM.")
        mode = "BCQ"
    print("Training Details:")
    print(f"Batch Size: {parameters['batch_size']}")
    print(f"Number of agents: {parameters['number_of_agents']}")
    print(f"Dataset episodes: {args.data_n_eps}")
    if args.use_mix:
        print(f"Dataset ratio: {max_num_files_1} : {max_num_files_1} : {max_num_files_2}")
    elif args.use_or:
        print(f"Dataset ratio: {max_num_files_1} : {max_num_files_1} : {max_num_files_2}")
    else:
        print(f"Dataset ratio: {max_num_files_1} : {max_num_files_1} : {max_num_files_2}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    print("Starting wandb, view at https://wandb.ai/")
    wandb.init(
		project='flatland-TreeLSTM', 
		name=f"{mode}_ALL_seed{args.seed}_{time.strftime('%m%d%H%M%S')}",
		config=parameters
	)

    replay_buffer = ReplayBuffer()

    train_BCQ(replay_buffer, data_file, num_actions, args, parameters)
