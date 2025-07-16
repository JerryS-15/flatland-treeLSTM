import pickle
from argparse import ArgumentParser
from time import sleep
import wandb
import time

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
from plfActor import Actor
from utils import VideoWriter, debug_show

import os
import copy
import pandas as pd

N_AGENTS = 20
WIDTH = 30
HEIGHT = 35
NUM_EPISODES = 1000
MAX_TIMESTEPS = 5000
collect_data_path_name = f"offline_rl_data_treeLSTM_{N_AGENTS}_agents_{NUM_EPISODES}_episodes"

def create_random_env():
    return RailEnv(
        number_of_agents=N_AGENTS,
        width=WIDTH,
        height=HEIGHT,
        rail_generator=SparseRailGen(
            max_num_cities=3,
            grid_mode=False,
            max_rails_between_cities=2,
            max_rail_pairs_in_city=2,
        ),
        line_generator=SparseLineGen(
            speed_ratio_map={1.0: 1 / 4, 0.5: 1 / 4, 0.33: 1 / 4, 0.25: 1 / 4}
        ),
        malfunction_generator=ParamMalfunctionGen(
            MalfunctionParameters(
                malfunction_rate=1 / 4500, min_duration=20, max_duration=50
            )
        ),
        obs_builder_object=TreeCutils(fp.num_tree_obs_nodes, fp.tree_pred_path_depth),
    )


def get_model_path(n_agents):
    if n_agents <= 50:
        model_path = "policy/phase-III-50.pt"
    elif n_agents <= 80:
        model_path = "policy/phase-III-80.pt"
    elif n_agents <= 100:
        model_path = "policy/phase-III-100.pt"
    else:
        model_path = "policy/phase-III-200.pt"
    return model_path


def get_args():
    parser = ArgumentParser(
        description="A multi-agent reinforcement learning solution to flatland3."
    )
    parser.add_argument(
        "--nr",
        "--no-render",
        dest="render",
        action="store_const",
        const=False,
        default=False,
        help="do not display game window",
    )
    parser.add_argument(
        "--fps", type=float, default=30, help="frames per second (default 10)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="the checkpoint file of saved model. If not given, a proper model is chosen according to number of agents.",
    )
    parser.add_argument(
        "--env", default=None, help="path to saved '*.pkl' file of envs"
    )
    parser.add_argument("--save-video", "-s", default=None, help="path to save video")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="number of episodes when collecting data")
    parser.add_argument("--norm-rewards", "-nr", action="store_true", help="if collect norm rewards for agent")
    # parser.add_argument("--n_agents", type=int, default=10, help="number of agents in the environment")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    if args.norm_rewards:
        print("Use norm reward for data collection.")

    print("---------------------------------------")
    print(f"Data Collection Started for {N_AGENTS} agents, {args.episodes} episodes.")
    print("---------------------------------------")

    if not os.path.exists("./offlineData"):
        os.makedirs("./offlineData")

    print("Starting wandb, view at https://wandb.ai/")
    wandb.init(
		project='flatland-TreeLSTM', 
		name=f"DATASET_{N_AGENTS}agents_{NUM_EPISODES}eps_{time.strftime('%m%d%H%M%S')}"
	)

    # create env
    if args.env is None:
        env = create_random_env()
    else:
        env, _ = RailEnvPersister.load_new(args.env)
        env.obs_builder = TreeCutils(fp.num_tree_obs_nodes, fp.tree_pred_path_depth)
    env_wrapper = LocalTestEnvWrapper(env)
    n_agents = env_wrapper.env.number_of_agents

    # load actor
    if args.model is None:
        model_path = get_model_path(n_agents)
    else:
        model_path = args.model
    actor = Actor(model_path)
    print(f"Load actor from {model_path}")

    if args.norm_rewards:
        save_path = f"offlineData/{collect_data_path_name}_normR.pkl"
    else:
        save_path = f"offlineData/{collect_data_path_name}.pkl"

    # create video writer
    if args.save_video is not None:
        video_writer = VideoWriter(args.save_video, args.fps)

    # initialize data storage
    all_offline_data = []
    dataset_info = []

    for episode in range(args.episodes):
        print(f"Episode {episode + 1}/{args.episodes}")
        obs = env_wrapper.reset()
        step_count = 0
        episode_data = []
        info_data = []

        while step_count < MAX_TIMESTEPS:
            step_count += 1
            va = env_wrapper.get_valid_actions()
            action = actor.get_actions(obs, va, n_agents)
            next_obs, all_rewards, done, step_rewards = env_wrapper.step(action)
            done_dict = done.copy()
            # obs_data = obs.deepcopy()
            # all_rewards_data = all_rewards.copy()
            # next_obs_data = next_obs.deepcopy()
            
            # print(f"RAW DATA | Episode {episode+1} | Step {step_count} done:", done) 
            episode_data.append((
                copy.deepcopy(obs),
                action,
                copy.deepcopy(all_rewards),
                copy.deepcopy(next_obs),
                done_dict,
            ))
            obs = next_obs

            if args.render:
                debug_show(env_wrapper.env)
                sleep(1 / args.fps)

            if args.save_video is not None:
                frame = debug_show(env_wrapper.env, mode="rgb_array")
                video_writer.write(frame)

            if done["__all__"]:
                arrival_ratio, total_reward, norm_reward, agent_norm_reward = env_wrapper.final_metric()

                if args.norm_rewards:
                    last_entry = list(episode_data[-1])
                    last_entry[2] = copy.deepcopy(agent_norm_reward)  # Replace reward
                    episode_data[-1] = tuple(last_entry)

                print(f"TOTAL_REW: {total_reward}, NORM_REW: {norm_reward:.4f}, ARR_RATIO: {arrival_ratio*100:.2f}%, with {len(episode_data)} samples.")
                info_data.append((
                    episode,
                    total_reward,
                    norm_reward,
                    arrival_ratio*100,
                    len(episode_data)
                ))

                wandb.log({"Total Reward": total_reward, "Episode": episode+1})
                wandb.log({"Normalized Reward": norm_reward, "Episode": episode+1})
                wandb.log({"Arrival Ratio %": arrival_ratio*100, "Episode": episode+1})
                wandb.log({"Number of Samples": len(episode_data), "Episode": episode+1})

                break
        
        all_offline_data.extend(episode_data)
        dataset_info.extend(info_data)

        # for d in episode_data:
        #     print(f"COLLECTED | done:", d[4]) 

    # print("--------------done-----------------")
    # for d in all_offline_data:
    #     print(d[4])
    # print("-----------------------------------")
    
    with open(save_path, "wb") as f:
        pickle.dump(all_offline_data, f)
        # pickle.dump(episode_data, f)
    print("Offline RL data is saved at ", save_path)

    if args.norm_rewards:
        dataset_info_path = f"offlineData/INFO-{collect_data_path_name}_normR.csv"
    else:
        dataset_info_path = f"offlineData/INFO-{collect_data_path_name}.csv"
    df_info = pd.DataFrame(dataset_info, columns=[
        "episode", "total_reward", "norm_reward", "arrival_ratio", "num_samples_per_episode"
    ])  
    df_info.to_csv(dataset_info_path, index=False)
    print("Dataset INFO documented at ", dataset_info_path)

    # # start step loop
    # obs = env_wrapper.reset()
    # step_count = 0
    
    # while step_count < MAX_TIMESTEPS:
    #     step_count += 1
    #     va = env_wrapper.get_valid_actions() # Valid actions
    #     action = actor.get_actions(obs, va, n_agents)
    #     next_obs, all_rewards, done = env_wrapper.step(action)

    #     offline_data.append((
    #         obs,
    #         action,
    #         all_rewards,
    #         next_obs,
    #         done,
    #     ))
    #     # record action of each agent
    #     # for agent_id in range(n_agents):
    #     #     offline_data.append((
    #     #         obs,          # observation
    #     #         action[agent_id],       # action
    #     #         all_rewards[agent_id],  # rewards
    #     #         next_obs,     # next observation
    #     #         done[agent_id],         # done (if terminate)
    #     #     ))
    #     obs = next_obs

    #     # print(f"[Step {step_count}] Agents: {n_agents}, Obs Shape: {len(obs)}, Valid Actions Shape: {len(va)}")

    #     # rendering
    #     if args.render:
    #         debug_show(env_wrapper.env)
    #         sleep(1 / args.fps)

    #     if args.save_video is not None:
    #         frame = debug_show(env_wrapper.env, mode="rgb_array")
    #         video_writer.write(frame)

    #     if done["__all__"]:
    #         if args.save_video is not None:
    #             video_writer.close()
    #             print(f"Write video to {args.save_video}")

    #         arrival_ratio, total_reward, norm_reward = env_wrapper.final_metric()
    #         print(f"TOTAL_REW: {total_reward}")
    #         print(f"NORM_REW: {norm_reward:.4f}")
    #         print(f"ARR_RATIO: {arrival_ratio*100:.2f}%")

    #         # save collected data
    #         save_path = "offline_rl_data_2003.pkl"

    #         with open(save_path, "wb") as f:
    #             print("Final data: ", type(offline_data), len(offline_data))
    #             pickle.dump(offline_data, f)
    #         print("Offline RL data is saved at ", save_path)
            
    #         break
