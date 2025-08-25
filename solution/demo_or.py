import argparse
from argparse import ArgumentParser
from time import sleep

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

seed = 4
env_path = f"or_solution_data_agent_5/env_data_v2_{seed}.pkl"
action_path = f"or_solution_data_agent_5/action_data_v2_{seed}.pkl"
step_path = f"or_solution_data_agent_5/step_data_v2_{seed}.pkl"

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
        # agent_start_times = [0 for _ in agents]
        # earliest_departures = [agent.get("earliest_departure", 0) for agent in agents]

        # line = Line(agent_positions, agent_directions, agent_targets, agent_speeds)
        # timetable = Timetable(
        #     earliest_departures=earliest_departures,
        #     latest_arrivals=[10**9] * len(agents),
        #     max_episode_steps=env_params.get("max_episode_steps", 1000)
        # )

        return Line(agent_positions, agent_directions, agent_targets, agent_speeds)
        # return line, timetable

        # print("========== Line is:", Line)
        # line = Line()
        # for agent in agents:
        #     line.add(
        #         start=agent["initial_position"],
        #         target=agent["target"],
        #         direction=agent["direction"],
        #         speed=agent["speed"],
        #         start_time=0
        #     )
        # return line

    env = RailEnv(
        width=env_params["width"],
        height=env_params["height"],
        number_of_agents=env_params["number_of_agents"],
        rail_generator=custom_rail_generator,
        line_generator=custom_line_generator,
        # obs_builder_object=DummyObservationBuilder(),
        malfunction_generator=None,
        # malfunction_generator=ParamMalfunctionGen(
        #     MalfunctionParameters(
        #         malfunction_rate=env_params['malfunction_rate'],
        #         min_duration=env_params['min_duration'],
        #         max_duration=env_params['max_duration']
        #     )
        # ),
        obs_builder_object=TreeCutils(fp.num_tree_obs_nodes, fp.tree_pred_path_depth),
        remove_agents_at_target=True,
        random_seed=seed
    )

    ###
    env._spawn_agents = lambda: None

    env_wrapper = LocalTestEnvWrapper(env)

    # env_wrapper.reset()

    # env.reset()
        
    return env_wrapper

def get_or_actions(step):
    action_data = load_env_data(action_path)
    return action_data[step]

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

def inject_agent_states(env, agent_states, action_dict):
    """
    将外部收集到的 agent 状态(来自 v2.2.1 replay) 同步到当前 v3.0.15 环境。
    agent_states: List[Dict]，每个 dict 包含 position, direction, status 等字段
    """
    for i, agent in enumerate(env.agents):
        # 清除旧状态
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
        # agent.speed_data = agent_state.get("speed_data", agent.speed_data)
        # agent.malfunction_data = agent_state.get("malfunction_data", agent.malfunction_data)

def get_states(env, agent_states, action_dict):
    for i, agent_state in enumerate(agent_states):
        agent = env.agents[i]
        if agent_state['status'] == "READY_TO_DEPART":
            agent.state = TrainState.READY_TO_DEPART
            agent.position = None
            action_dict[i] = 0
            print("########## Agent State Set! ##########")
        else:
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nr",
        "--no-render",
        dest="render",
        action="store_const",
        const=False,
        default=True,
        help="do not display game window",
    )
    parser.add_argument(
        "--fps", type=float, default=30, help="frames per second (default 10)"
    )
    parser.add_argument(
        "--use_model", action="store_true", help="Use trained model in TreeLSTM."
    )
    args = parser.parse_args()

    env_data = load_env_data(env_path)
    env_wrapper = rebuild_wrapped_env_from_data(env_data)
    # env = rebuild_wrapped_env_from_data(env_data)
    # env_renderer = RenderTool(env)
    step_data = load_env_data(step_path)

    obs = env_wrapper.reset()
    
    for agent in env_wrapper.env.agents:
        agent.earliest_departure = 0

    for i, agent in enumerate(env_wrapper.env.agents):
        print(f"Agent {i} earliest_departure: {agent.earliest_departure}")
        print(f"Agent {i} speed: {agent.speed_counter}")

    # if args.use_model:
    n_agents = env_wrapper.env.number_of_agents
    model_path = get_model_path(n_agents)
    actor = Actor(model_path)
    print(f"Load actor from {model_path}")

    default_actions = {handle: 0 for handle in range(env_wrapper.env.number_of_agents)}

    step = 0
    while True:
        if args.use_model:
            va = env_wrapper.get_valid_actions()
            action = actor.get_actions(obs, va, n_agents)
            obs, all_rewards, done, step_rewards = env_wrapper.step(action)
            print(f"[Step {step}]")
            print(f"RL actions: {action}")
        else:
            if step < 0:
                action = default_actions
            else:
                action = get_or_actions(step)
                agent_states = step_data[step]
                inject_agent_states(env_wrapper.env, agent_states, action)
        
            print(f"[Step {step}]")
            print(f"OR actions: {action}")
        
            # Check invalid actions:
            valid_actions = env_wrapper.get_valid_actions()
            model_actions = actor.get_actions(obs, valid_actions, env_wrapper.env.number_of_agents)
            for agent_id in range(env_wrapper.env.number_of_agents):
                if action[agent_id] not in valid_actions[agent_id]:
                    print(f" !!! Agent {agent_id} with invalid action {action[agent_id]}, take model action {model_actions[agent_id]} instead.")
                    action[agent_id] = model_actions[agent_id]

            print(f"Take actions: {action}")
            obs, all_rewards, done, step_rewards = env_wrapper.step(action)

        step = step + 1

        print(f"dones: {done}")
        for idx, agent in enumerate(env_wrapper.env.agents):
            print(f"Agent {idx} position: {agent.position}, direction: {agent.direction}, target={agent.target}, init={agent.initial_position}")
            # print(f"Agent {idx} info: {agent.state}")

        if args.render:
            debug_show(env_wrapper.env)
            sleep(1 / args.fps)

        if done["__all__"]:
            arrival_ratio, total_reward, norm_reward, _ = env_wrapper.final_metric()
            print(f"TOTAL_REW: {total_reward}")
            print(f"NORM_REW: {norm_reward:.4f}")
            print(f"ARR_RATIO: {arrival_ratio*100:.2f}%")
            break

    """
    Original loop
    """
    # step = -1
    # while True:
    #     if args.use_model:
    #         va = env_wrapper.get_valid_actions()
    #         action = actor.get_actions(obs, va, n_agents)
    #     else:
    #         if step < 0:
    #             action = default_actions
    #         else:
    #             agent_states = step_data[step]
    #             action = get_or_actions(step)
    #             get_states(env_wrapper.env, agent_states, action)
    #     print(f"{step}: {action}")
    #     obs, all_rewards, done, step_rewards = env_wrapper.step(action)
    #     # obs, all_rewards, done, _ = env.step(action)
    #     print(f"dones: {done}")
    #     for idx, agent in enumerate(env_wrapper.env.agents):
    #         print(f"Agent {idx} position: {agent.position}, direction: {agent.direction}, target={agent.target}, init={agent.initial_position}")
    #         print(f"Agent {idx} info: {agent.state}")

    #     step = step + 1

    #     # if step >= len(load_env_data(action_path)):
    #     #     print("No more actions in the dataset, stopping.")
    #     #     arrival_ratio, total_reward, norm_reward, _ = env_wrapper.final_metric()
    #     #     print(f"TOTAL_REW: {total_reward}")
    #     #     print(f"NORM_REW: {norm_reward:.4f}")
    #     #     print(f"ARR_RATIO: {arrival_ratio*100:.2f}%")
    #     #     break

    #     if args.render:
    #         debug_show(env_wrapper.env)
    #         # env_renderer.render_env(show=True, frames=True, show_observations=False)
    #         sleep(1 / args.fps)
        
    #     if done["__all__"]:
    #         arrival_ratio, total_reward, norm_reward, _ = env_wrapper.final_metric()
    #         print(f"TOTAL_REW: {total_reward}")
    #         print(f"NORM_REW: {norm_reward:.4f}")
    #         print(f"ARR_RATIO: {arrival_ratio*100:.2f}%")
    #         # print("Agent reached target!")
    #         break
    """
    End of original loop
    """
    # if args.render:
    #     env_renderer.close_window()

