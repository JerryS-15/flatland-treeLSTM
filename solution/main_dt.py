import argparse
import copy
import os
import time
import numpy as np
import torch
from tqdm import tqdm
import wandb

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
from replayBufferDT import ReplayBufferDT
from dtActor import Actor  # optional for evaluation

from discrete_DT import MultiAgentDecisionTransformer


# def add_dt_fields(batch, discount=1.0):
#     """
#         Add rtgs (return-to-go) to batch

#         Input:
#         - batch: dict, with ['all_rewards', 'dones'] included
#                 all_rewards: [B, T, N], dones: [B, T, N]

#         Return:
#         - batch with two additional keys:
#             - 'rtgs': same shape as rewards
#             - 'timesteps': shape [B, T]
#     """
#     rewards = batch['all_rewards']  # [B, T, N]
#     dones = batch['dones']          # [B, T, N]
#     B, T, N = rewards.shape
#     rtgs = torch.zeros_like(rewards)
#     timesteps = torch.arange(T).unsqueeze(0).expand(B, T)
#     for b in range(B):
#         for n in range(N):
#             R = 0.0
#             for t in reversed(range(T)):
#                 R = rewards[b, t, n] + (1 - dones[b, t, n]) * discount * R
#                 rtgs[b, t, n] = R
#     batch['rtgs'] = rtgs
#     batch['timesteps'] = timesteps
#     return batch


def train_DT(replay_buffer, data_file, num_actions, args, parameters):
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu:6")
    print(f"Using device: {device}")

    policy_name = f"DT-{parameters['number_of_agents']}agents-bs{parameters['batch_size']}"
    policy_path = f"./policy/{policy_name}"

    policy = MultiAgentDecisionTransformer(
        num_actions=num_actions,
        device=device
    )

    replay_buffer.load_from_folder(data_file)
    print(f"Loaded {len(replay_buffer.buffer)} transitions.")

    evaluations = []
    training_iters = 0

    print("DT Training started.")

    while training_iters < args.max_timesteps:
        epoch_metrics = []
        for ep in tqdm(range(int(parameters["eval_freq"])), desc="DT Training Progress"):
            batch = replay_buffer.sample(parameters["batch_size"])
            # batch = add_dt_fields(batch, discount=0.99)
            metrics = policy.train(batch)
            epoch_metrics.append(metrics)
            if ep % 1000 == 0:
                tqdm.write(f"Epoch {ep}, Loss: {metrics['loss']:.4f}")
        
        avg_metrics = {
            k: np.mean([m[k] for m in epoch_metrics])
            for k in epoch_metrics[0].keys()
        }

        policy.save(policy_path)
        evaluations.append(eval_policy(policy_path, parameters, args.seed))
        np.save(f"./results/{policy_name}", evaluations)

        wandb.log({
            "DT/loss": avg_metrics["loss"]
        }, step=training_iters)

        print(f"[DT] Iteration: {training_iters} | Avg Loss: {avg_metrics['loss']:.3f}")

        training_iters += int(parameters["eval_freq"])

    policy.save(policy_path)


def eval_policy(model_path, env_params, seed, eval_episodes=10):
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
        malfunction_generator=None,
        obs_builder_object=TreeCutils(fp.num_tree_obs_nodes, fp.tree_pred_path_depth),
        remove_agents_at_target=True,
        random_seed=seed + 100
    )

    env_wrapper = LocalTestEnvWrapper(eval_env)
    n_agents = env_wrapper.env.number_of_agents

    actor = Actor(model_path, device="cuda:6")
    print(f"Evaluating using actor from {model_path}")

    total_rewards = []
    norm_rewards = []
    arrival_ratios = []

    for ep in range(eval_episodes):
        eval_seed = seed + 100 + ep*10
        obs = env_wrapper.reset(random_seed=eval_seed)
        done = {"__all__": False}

        while not done["__all__"]:
            valid_actions = env_wrapper.get_valid_actions()
            actions = actor.get_actions(obs, valid_actions, n_agents)
            obs, all_rewards, done, _ = env_wrapper.step(actions)
        
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
    wandb.log({"Avg Total Reward": avg_total_reward, "Evaluation Episodes": eval_episodes})
    wandb.log({"Avg Norm Reward": avg_norm_reward, "Evaluation Episodes": eval_episodes})
    wandb.log({"Avg Arrival Ratio %": avg_arrival_ratio*100, "Evaluation Episodes": eval_episodes})
    return avg_norm_reward


def collect_pickle_paths(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".pkl")
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--seed", default=5000, type=int)
    parser.add_argument("--data_n_eps", default=1000, type=int)
    parser.add_argument("--n_agents", default=5, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--normal_reward", action="store_true", help="Use dataset with normed_rewards for agent")
    parser.add_argument("--use_or", "-or", action="store_true")
    parser.add_argument("--use_mix", "-mix", action="store_true")
    args = parser.parse_args()

    n_agents = args.n_agents
    batch_size = args.batch_size

    flatland_parameters = {
        "eval_freq": 1e4,
        "batch_size": batch_size,
        "number_of_agents": n_agents,
        "width": 30,
        "height": 35,
        "max_num_cities": 3,
        "max_rails_between_cities": 2,
        "max_rail_pairs_in_city": 2,
        "speed_ratio_map": {1.0: 1 / 3, 0.5: 1 / 3, 0.25: 1 / 3},
        "malfunction_rate": 1 / 4500,
        "min_duration": 20,
        "max_duration": 50
    }

    parameters = flatland_parameters

    num_actions = 5
    if args.use_or:
        if args.normal_reward:
            or_folder = f"./orData_agent_{n_agents}_normR"
            data_file = collect_pickle_paths(or_folder)
        else:
            or_folder = f"./orData_agent_{n_agents}"
            data_file = collect_pickle_paths(or_folder)
    elif args.use_mix:
        data_folder1 = f"./orData_agent_{n_agents}_normR"
        data_folder2 = f"./offlineData_{n_agents}"
        data_file = collect_pickle_paths(data_folder1) + collect_pickle_paths(data_folder2)
    else:
        rl_folder = f"./offlineData_{n_agents}"
        data_file = collect_pickle_paths(rl_folder)

    print("---------------------------------------")
    if args.use_or:
        print("Start DT training for flatland TreeLSTM with OR-Solution Dataset.")
        mode = "DT-OR"
    elif args.use_mix:
            print("Start DT training for flatland TreeLSTM with OR-RL Mixed Dataset.")
            mode = "DT-MIX"
    else:
        print("Start DT training for flatland TreeLSTM.")
        mode = "DT"
    print("Training Details:")
    print(f"Batch Size: {parameters['batch_size']}")
    print(f"Number of agents: {parameters['number_of_agents']}")
    print(f"Dataset episodes: {args.data_n_eps}")
    if args.use_mix:
        print(f"Dataset folder: {data_folder1} & {data_folder2}")
    elif args.use_or:
        print(f"Dataset folder: {or_folder}")
    else:
        print(f"Dataset file: {data_file}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    print("Starting wandb, view at https://wandb.ai/")
    wandb.init(
        project='flatland-TreeLSTM', 
        name=f"{mode}_{n_agents}agents_seed{args.seed}_{time.strftime('%m%d%H%M%S')}",
        config=parameters
    )

    replay_buffer = ReplayBufferDT(max_len=50)
    train_DT(replay_buffer, data_file, num_actions, args, parameters)
