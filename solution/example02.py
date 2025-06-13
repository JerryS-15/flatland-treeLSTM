import getopt
import random
import sys
import time
from typing import List

import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.line_generators import SparseLineGen
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import SparseRailGen
from flatland.envs.observations import TreeObsForRailEnv
from flatland.utils.misc import str2bool
from flatland.utils.rendertools import RenderTool

import heapq

random.seed(100)
np.random.seed(100)


class SingleAgentNavigationObs(ObservationBuilder):
    """
    We build a representation vector with 3 binary components, indicating which of the 3 available directions
    for each agent (Left, Forward, Right) lead to the shortest path to its target.
    E.g., if taking the Left branch (if available) is the shortest route to the agent's target, the observation vector
    will be [1, 0, 0].
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get(self, handle: int = 0) -> List[int]:
        agent = self.env.agents[handle]

        if agent.position:
            possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        else:
            possible_transitions = self.env.rail.get_transitions(*agent.initial_position, agent.direction)

        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        if num_transitions == 1:
            observation = [0, 1, 0]
        else:
            min_distances = []
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    new_position = get_new_position(agent.position, direction)
                    min_distances.append(
                        self.env.distance_map.get()[handle, new_position[0], new_position[1], direction])
                else:
                    min_distances.append(np.inf)

            observation = [0, 0, 0]
            observation[np.argmin(min_distances)] = 1

        return observation


def create_env():
    nAgents = 1
    n_cities = 2
    max_rails_between_cities = 2
    max_rails_in_city = 4
    seed = 0
    env = RailEnv(
        width=50,
        height=50,
        rail_generator=SparseRailGen(
            max_num_cities=n_cities,
            # seed=seed,
            grid_mode=True,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rails_in_city
        ),
        line_generator=SparseLineGen(),
        number_of_agents=nAgents,
        obs_builder_object=TreeObsForRailEnv(max_depth=2)
    )
    return env


def custom_observation_example_02_SingleAgentNavigationObs(sleep_for_animation, do_rendering):
    env = create_env()
    obs, info = env.reset()

    env_renderer = None
    if do_rendering:
        env_renderer = RenderTool(env)
        env_renderer.render_env(show=True, frames=True, show_observations=False)

    for step in range(100):
        action = np.argmax(obs[0]) + 1
        obs, all_rewards, done, _ = env.step({0: action})
        print("Rewards: ", all_rewards, "  [done=", done, "]")

        if env_renderer is not None:
            env_renderer.render_env(show=True, frames=True, show_observations=True)
        if sleep_for_animation:
            time.sleep(0.1)
        if done["__all__"]:
            break
    if env_renderer is not None:
        env_renderer.close_window()

def a_star(env, start, goal):
    """
    A* pathfinding from start to goal in Flatland grid.
    """
    open_set = []
    heapq.heappush(open_set, (0, start, []))  # (cost, position, path)

    visited = set()

    while open_set:
        cost, current_pos, path = heapq.heappop(open_set)

        if current_pos == goal:
            return path + [current_pos]

        if current_pos in visited:
            continue
        visited.add(current_pos)

        for direction in range(4):
            transitions = env.rail.get_transitions(*current_pos, direction) 
            for dir_offset, is_possible in enumerate(transitions):
                if not is_possible:
                    continue
                new_dir = dir_offset
                next_pos = get_new_position(current_pos, new_dir)
                if next_pos in visited:
                    continue
                heapq.heappush(open_set, (
                    cost + 1 + manhattan_distance(next_pos, goal),
                    next_pos,
                    path + [current_pos]
                ))

    return []  # no path found

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def direction_to_action(current_direction, desired_direction):
    delta = (desired_direction - current_direction) % 4
    return {0: 2, 1: 3, 3: 1}.get(delta, 0)

def get_direction(from_pos, to_pos):
    for d in range(4):
        if get_new_position(from_pos, d) == to_pos:
            return d
    return None

def run_astar_agent(sleep_for_animation=True, do_rendering=True):
    env = create_env()
    obs, info = env.reset()

    env_renderer = RenderTool(env)
    if do_rendering:
        env_renderer.render_env(show=True, frames=True, show_observations=False)

    agent = env.agents[0]
    start = agent.initial_position
    goal = agent.target

    print(f"Start: {start} | Goal: {goal}")

    path = a_star(env, start, goal)
    if not path:
        print("No path found!")
        return

    print("A* Path:", path)

    current_direction = agent.direction
    pos_index = 1

    while pos_index < len(path):
        curr_pos = path[pos_index - 1]
        next_pos = path[pos_index]

        desired_direction = get_direction(curr_pos, next_pos)
        if desired_direction is None:
            print("Error: no valid direction found")
            break

        action = direction_to_action(current_direction, desired_direction)
        current_direction = desired_direction

        obs, all_rewards, done, _ = env.step({0: action})
        print(f"Step {pos_index}: Pos {next_pos} | Action {action} | Reward {all_rewards[0]}")

        if do_rendering:
            env_renderer.render_env(show=True, frames=True, show_observations=False)

        if sleep_for_animation:
            time.sleep(0.2)

        if done["__all__"]:
            print("Agent reached target!")
            break

        pos_index += 1

    if do_rendering:
        env_renderer.close_window()

def main():
    # try:
    #     opts, args = getopt.getopt(args, "", ["sleep-for-animation=", "do_rendering=", ""])
    # except getopt.GetoptError as err:
    #     print(str(err))  # will print something like "option -a not recognized"
    #     sys.exit(2)
    sleep_for_animation = True
    do_rendering = True
    # for o, a in opts:
    #     if o in ("--sleep-for-animation"):
    #         sleep_for_animation = str2bool(a)
    #     elif o in ("--do_rendering"):
    #         do_rendering = str2bool(a)
    #     else:
    #         assert False, "unhandled option"

    # execute example
    # custom_observation_example_02_SingleAgentNavigationObs(sleep_for_animation, do_rendering)
    run_astar_agent(sleep_for_animation, do_rendering)


if __name__ == '__main__':
    # if 'argv' in globals():
    #     main(argv)
    # else:
    #     main(sys.argv[1:])
    main()
    # run_astar_agent()