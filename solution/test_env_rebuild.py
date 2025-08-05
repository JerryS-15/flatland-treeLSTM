import pickle
import pytest
import os
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.agent_utils import EnvAgentStatic, SpeedData

def load_env_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def rebuild_env_from_data(data):
    env = RailEnv(
        width=data["rail"].width,
        height=data["rail"].height,
        number_of_agents=len(data["agents"]),
        rail_generator=lambda *args: (data["rail"], None),
        schedule_generator=None,
        obs_builder_object=DummyObservationBuilder(),
        malfunction_generator=None,
        remove_agents_at_target=True,
        random_seed=data["seed"]
    )
    env.reset()
    env.agents.clear()

    for agent in data["agents"]:
        env.agents.append(EnvAgentStatic(
            initial_position=agent["initial_position"],
            direction=agent["direction"],
            target=agent["target"],
            speed_data=SpeedData(agent["speed"])
        ))

    env.reset(False, False)
    return env

@pytest.mark.parametrize("seed", list(range(42, 52)))
def test_rebuild_env(seed):
    data_path = f"test_env_data/env_v2_{seed}.pkl"
    assert os.path.exists(data_path), f"❌ Missing pickle file: {data_path}"

    original = load_env_data(data_path)
    rebuilt = rebuild_env_from_data(original)

    # Rail grid comparison
    orig_grid = original["rail"].grid
    rebuilt_grid = rebuilt.rail.grid
    assert np.array_equal(orig_grid, rebuilt_grid), f"❌ Rail grid mismatch at seed {seed}"

    # Agent comparison
    for i, (a1, a2) in enumerate(zip(original["agents"], rebuilt.agents)):
        assert a1["initial_position"] == a2.initial_position, f"❌ Agent {i} initial_position mismatch"
        assert a1["direction"] == a2.direction, f"❌ Agent {i} direction mismatch"
        assert a1["target"] == a2.target, f"❌ Agent {i} target mismatch"
        assert np.isclose(a1["speed"], a2.speed_data["speed"]), f"❌ Agent {i} speed mismatch"