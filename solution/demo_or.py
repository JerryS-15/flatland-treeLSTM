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

seed = 1
env_path = f"or_solution_data/env_data_v2_{seed}.pkl"
action_path = f"or_solution_data/action_data_v2_{seed}.pkl"

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

# === 加入 spawn_from_expert 和 replay_expert 函数 ===
def spawn_from_expert(expert_actions: dict) -> dict:
    """
    Returns a dict: {step: [agent_ids]} indicating at which step each agent should spawn.
    """
    spawn_schedule = {}

    for agent_id, actions in expert_actions.items():
        for step, action in enumerate(actions):
            if action != 4:  # DO_NOTHING
                if step not in spawn_schedule:
                    spawn_schedule[step] = []
                spawn_schedule[step].append(agent_id)
                break  # Only first movement matters

    return spawn_schedule

def replay_expert(env_wrapper, expert_actions: dict, record_data=False, render=False, fps=30):
    spawn_schedule = spawn_from_expert(expert_actions)
    n_agents = len(env_wrapper.env.agents)
    step = 0
    done = {"__all__": False}
    dataset = []

    obs = env_wrapper.reset()

    while not done["__all__"]:
        # 1. Spawn agents
        for agent_id in spawn_schedule.get(step, []):
            agent = env_wrapper.env.agents[agent_id]
            if agent.position is None:
                agent.position = agent.initial_position
                agent.state = TrainState.MOVING

        # 2. Expert actions
        action = {}
        for agent_id in range(n_agents):
            if step < len(expert_actions[agent_id]):
                action[agent_id] = expert_actions[agent_id][step]
            else:
                action[agent_id] = 4  # DO_NOTHING

        # 3. Step
        next_obs, rewards, done, _ = env_wrapper.step(action)

        if record_data:
            dataset.append({
                "step": step,
                "obs": obs,
                "action": action,
                "reward": rewards,
                "next_obs": next_obs,
                "done": done
            })
        
        # 4. Render
        if render:
            debug_show(env_wrapper.env)
            sleep(1 / fps)
        
        if done["__all__"]:
            arrival_ratio, total_reward, norm_reward, _ = env_wrapper.final_metric()
            print(f"TOTAL_REW: {total_reward}")
            print(f"NORM_REW: {norm_reward:.4f}")
            print(f"ARR_RATIO: {arrival_ratio*100:.2f}%")
            # print("Agent reached target!")
            break

        obs = next_obs
        step += 1

    return dataset if record_data else None

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

    obs = env_wrapper.reset()
    
    for agent in env_wrapper.env.agents:
        agent.earliest_departure = 0
    
    ###
    # spawn_blocked_until = {}
    
    # def custom_spawn(env, current_step, spawn_delay=2):
    #     """
    #     自定义 spawn 函数：
    #     - earliest_departure=0 的 agent 在 step=0 就可以出发
    #     - 如果多个 agent 起点相同，第二个及之后的 agent 会排队，
    #       直到前面的 agent 离开起点后才能 spawn
    #     """
    #     # 每个起点维护一个 spawn 队列
    #     if not hasattr(env, "_spawn_queue"):
    #         env._spawn_queue = {}
    #     if not hasattr(env, "_spawn_clock"):
    #         env._spawn_clock = {}  # 每个起点的内部 clock
        
    #     print(f"### {current_step}, {env._spawn_clock} ###")

    #     for handle, agent in enumerate(env.agents):
    #         start_pos = agent.initial_position
    #         if start_pos is None:
    #             continue

    #         # 第一次见到这个起点 → 初始化队列
    #         if start_pos not in env._spawn_queue:
    #             env._spawn_queue[start_pos] = []
    #         if start_pos not in env._spawn_clock:
    #             env._spawn_clock[start_pos] = 0

    #         # agent 未进入轨道且到达 earliest_departure 时 → 加入队列
    #         if agent.position is None and current_step >= agent.earliest_departure:
    #             if handle not in env._spawn_queue[start_pos]:
    #                 env._spawn_queue[start_pos].append(handle)

    #     # 遍历每个起点，检查是否可以 spawn 队列的第一个 agent
    #     for start_pos, queue in env._spawn_queue.items():
    #         if not queue:
    #             continue

            # 检查起点是否空
            # occupied_positions = [pos for pos in env.agent_positions if pos is not None]
            # if tuple(start_pos) in map(tuple, occupied_positions):
            #     continue  # 起点被占，不能 spawn

            # 初始化 clock
            # if start_pos not in env._spawn_clock:
            #     env._spawn_clock[start_pos] = 0
            
            # # 检查队首 agent 是否可以 spawn
            # handle = queue[0]
            # agent = env.agents[handle]

            # # 当 clock 达到 spawn_delay 时才能 spawn
            # if env._spawn_clock[start_pos] >= spawn_delay or env._spawn_clock[start_pos] == 0:
            #     print(f"###### NEW SPAWN! ######")
            #     # spawn agent
            #     agent.position = start_pos
            #     agent.state = TrainState.MOVING
            #     agent.speed_counter = SpeedCounter(agent.speed_counter.speed)
            
            #     # 弹出队列
            #     queue.pop(0)

            #     # 重置 clock（下一个 agent 会从 1 开始计数）
            #     env._spawn_clock[start_pos] = 0
            # else:
            #     # clock 自增
            #     env._spawn_clock[start_pos] += 1

    for i, agent in enumerate(env_wrapper.env.agents):
        print(f"Agent {i} earliest_departure: {agent.earliest_departure}")
        print(f"Agent {i} speed: {agent.speed_counter}")

    if args.use_model:
        n_agents = env_wrapper.env.number_of_agents
        model_path = get_model_path(n_agents)
        actor = Actor(model_path)
        print(f"Load actor from {model_path}")

    default_actions = {handle: 0 for handle in range(env_wrapper.env.number_of_agents)}

    step = -1
    # clock = 0
    # for agent in env_wrapper.env.agents:
    #     if agent.position is None:
    #         agent.earliest_departure = 0
    #         agent.state = agent.state.MOVING
    #         agent.position = agent.initial_position
    while True:
        ###
        # custom_spawn(env_wrapper.env, step)

        if args.use_model:
            va = env_wrapper.get_valid_actions()
            action = actor.get_actions(obs, va, n_agents)
        else:
            if step < 0:
                action = default_actions
            else:
                action = get_or_actions(step)
        print(f"{step}: {action}")
        obs, all_rewards, done, step_rewards = env_wrapper.step(action)
        # obs, all_rewards, done, _ = env.step(action)
        print(f"dones: {done}")
        for idx, agent in enumerate(env_wrapper.env.agents):
            print(f"Agent {idx} position: {agent.position}, direction: {agent.direction}, target={agent.target}, init={agent.initial_position}")

        step = step + 1

        # if step >= len(load_env_data(action_path)):
        #     print("No more actions in the dataset, stopping.")
        #     arrival_ratio, total_reward, norm_reward, _ = env_wrapper.final_metric()
        #     print(f"TOTAL_REW: {total_reward}")
        #     print(f"NORM_REW: {norm_reward:.4f}")
        #     print(f"ARR_RATIO: {arrival_ratio*100:.2f}%")
        #     break

        if args.render:
            debug_show(env_wrapper.env)
            # env_renderer.render_env(show=True, frames=True, show_observations=False)
            sleep(1 / args.fps)
        
        if done["__all__"]:
            arrival_ratio, total_reward, norm_reward, _ = env_wrapper.final_metric()
            print(f"TOTAL_REW: {total_reward}")
            print(f"NORM_REW: {norm_reward:.4f}")
            print(f"ARR_RATIO: {arrival_ratio*100:.2f}%")
            # print("Agent reached target!")
            break
    
    # if args.render:
    #     env_renderer.close_window()

