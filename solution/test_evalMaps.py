import pytest
import numpy as np
import hashlib
from copy import deepcopy

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

def rail_hash(env) -> str:
    """Compute a hash of the rail grid structure."""
    rail_grid = env.rail.grid
    rail_array = np.array(rail_grid, dtype=np.uint8)
    return hashlib.md5(rail_array.tobytes()).hexdigest()

@pytest.mark.parametrize("base_seed", [0, 42])
@pytest.mark.parametrize("test_steps", [21, 71])
def test_flatland_deterministic_map_generation(base_seed, test_steps):
    """Test a series of reset seeds generate 10 unique maps, which are reproducible."""
    env_params = {
		# Flatland Env
        "number_of_agents": 5,
        "width": 30,
        "height": 35,
        "max_num_cities": 3,
        "max_rails_between_cities": 2,
        "max_rail_pairs_in_city": 2,
        "speed_ratio_map": {1.0: 1 / 4, 0.5: 1 / 4, 0.33: 1 / 4, 0.25: 1 / 4},
        # Flatland - malfunction
        "malfunction_rate": 1 / 4500,
        "min_duration": 20,
        "max_duration": 50,
        # Rendering
        "fps": 30,
        "render": True
	}

    # Create env
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
        malfunction_generator=ParamMalfunctionGen(
            MalfunctionParameters(
                malfunction_rate=env_params['malfunction_rate'],
                min_duration=env_params['min_duration'],
                max_duration=env_params['max_duration']
            )
        ),
        obs_builder_object=TreeCutils(fp.num_tree_obs_nodes, fp.tree_pred_path_depth),
        random_seed=base_seed + 100
    )

    env_wrapper = LocalTestEnvWrapper(eval_env)
    n_agents = env_wrapper.env.number_of_agents

    actions = {i: 0 for i in range(n_agents)}

    # Finish 10 episodes and record the hash
    episode_hashes = []
    agent_positions = []
    for ep in range(10):
        seed_reset = base_seed + 100 + ep*10
        print(f"Round {ep} : seed_reset to {seed_reset}")
        obs = env_wrapper.reset(random_seed=seed_reset)
        h = rail_hash(env_wrapper.env)
        episode_hashes.append(h)
        counter = 0

        for i, agent in enumerate(eval_env.agents):
            agent = [i, agent.initial_position, agent.target]
            agent_positions.append(agent)
        
        while counter < test_steps:
            counter += 1
            obs, all_rewards, done, _ = env_wrapper.step(actions)
    
    # Finish same rounds of reset, test if hashes consistent
    for ep in range(10):
        seed_reset = base_seed + 100 + ep*10
        print(f"Round {ep} : seed_reset to {seed_reset}")
        obs = env_wrapper.reset(random_seed=seed_reset)
        h = rail_hash(env_wrapper.env)
        assert h == episode_hashes[ep], f"Map hash mismatch on episode {ep+1} !"
        counter = 0

        for i, agent in enumerate(eval_env.agents):
            agent = [i, agent.initial_position, agent.target]
            assert agent == agent_positions[i + ep * n_agents], f"Agent initialized positions mismatch on episode {ep+1}, agent {i} !"
        
        while counter < test_steps:
            counter += 1
            obs, all_rewards, done, _ = env_wrapper.step(actions)
    
    assert len(set(episode_hashes)) == 10, "Expected 10 unique maps with different seeds!"
