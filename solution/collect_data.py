import pickle
from argparse import ArgumentParser
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
from plfActor import Actor
from utils import VideoWriter, debug_show

import os
import math

collect_data_path = ""
MAX_TIMESTEPS = 5000
NUM_EPISODES = 10

def create_random_env():
    return RailEnv(
        number_of_agents=50,
        width=30,
        height=35,
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    print("---------------------------------------")
    print(f"Data Collection Started for {args.episodes} episodes.")
    print("---------------------------------------")

    if not os.path.exists("./offlineData"):
        os.makedirs("./offlineData")

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

    # create video writer
    if args.save_video is not None:
        video_writer = VideoWriter(args.save_video, args.fps)

    # initialize data storage
    all_offline_data = []

    for episode in range(args.episodes):
        print(f"Episode {episode + 1}/{args.episodes}")
        obs = env_wrapper.reset()
        step_count = 0
        episode_data = []

        while step_count < MAX_TIMESTEPS:
            step_count += 1
            va = env_wrapper.get_valid_actions()
            action = actor.get_actions(obs, va, n_agents)
            next_obs, all_rewards, done, step_rewards = env_wrapper.step(action)

            episode_data.append((
                obs,
                action,
                step_rewards,
                next_obs,
                done,
            ))
            obs = next_obs

            if args.render:
                debug_show(env_wrapper.env)
                sleep(1 / args.fps)

            if args.save_video is not None:
                frame = debug_show(env_wrapper.env, mode="rgb_array")
                video_writer.write(frame)

            if done["__all__"]:
                arrival_ratio, total_reward, norm_reward = env_wrapper.final_metric()
                print(f"TOTAL_REW: {total_reward}, NORM_REW: {norm_reward:.4f}, ARR_RATIO: {arrival_ratio*100:.2f}%, with {len(episode_data)} samples.")

                break
        
        all_offline_data.extend(episode_data)
    
    save_path = "offlineData/offline_rl_data_treeLSTM.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(all_offline_data, f)
    print("Offline RL data is saved at ", save_path)

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
