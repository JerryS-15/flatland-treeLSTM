import pickle
import pytest
import os
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_grid_transition_map, SparseRailGen
from flatland.envs.line_generators import Line
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.agent_utils import EnvAgent
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland_cutils import TreeObsForRailEnv as TreeCutils

from eval_env import LocalTestEnvWrapper
from impl_config import FeatureParserConfig as fp

def load_env_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def rebuild_env_from_data(data):
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
        # line = Line()
        # for agent in agents:
        #     line.append({
        #         "start": agent["initial_position"],
        #         "target": agent["target"],
        #         "speed": agent["speed"],
        #         "start_time": 0
        #     })
        # return line

    env = RailEnv(
        width=env_params["width"],
        height=env_params["height"],
        number_of_agents=env_params["number_of_agents"],
        rail_generator=custom_rail_generator,
        line_generator=custom_line_generator,
        # obs_builder_object=DummyObservationBuilder(),
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
    env_wrapper.reset()

    # env.reset()
        
    return env_wrapper.env

@pytest.mark.parametrize("seed", list(range(42, 52)))
def test_rebuild_env(seed):
    data_path = f"test_env_data_v2/env_v2_{seed}.pkl"
    assert os.path.exists(data_path), f"❌ Missing pickle file: {data_path}"

    original = load_env_data(data_path)
    rebuilt = rebuild_env_from_data(original)

    # Rail grid comparison
    orig_grid = original["rail"].grid
    rebuilt_grid = rebuilt.rail.grid
    # assert rebuilt.width == original["rail"].shape[1], f"❌ Rail grid width mismatch at seed {seed}"
    # assert rebuilt.height == original["rail"].shape[0], f"❌ Rail grid height mismatch at seed {seed}"
    # assert len(rebuilt.agents) == len(original["agent_info"]), f"❌ Agent len mismatch at seed {seed}"
    assert np.array_equal(orig_grid, rebuilt_grid), f"❌ Rail grid mismatch at seed {seed}"

    # Agent comparison
    for i, (a1, a2) in enumerate(zip(original["agent_info"], rebuilt.agents)):
        assert a1["initial_position"] == a2.initial_position, f"❌ Agent {i} initial_position mismatch"
        assert a1["direction"] == a2.direction, f"❌ Agent {i} direction mismatch"
        assert a1["target"] == a2.target, f"❌ Agent {i} target mismatch"
        assert np.isclose(a1["speed"], a2.speed_counter.speed), f"❌ Agent {i} speed mismatch"