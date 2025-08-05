import argparse
import hashlib
from tqdm import tqdm
import numpy as np
import pickle

import flatland
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

def create_env(env_params, seed):
    return RailEnv(
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
        remove_agents_at_target=True,
        random_seed=seed
    )

def hash_array(arr):
    return hashlib.md5(arr.tobytes()).hexdigest()

if __name__ == "__main__":

    init_seed = 42
    save_path = "env_v3.pkl"

    flatland_parameters = {
        # Flatland Env
        "number_of_agents": 10,
        "width": 30,
        "height": 35,
        "max_num_cities": 3,
        "max_rails_between_cities": 2,
        "max_rail_pairs_in_city": 2,
        "speed_ratio_map": {1.0: 1 / 4, 0.5: 1 / 4, 0.33: 1 / 4, 0.25: 1 / 4},
        # Flatland - malfunction
        "malfunction_rate": 1 / 4500,
        "min_duration": 20,
        "max_duration": 50
    }

    all_episodes = []

    for i in tqdm(range(0, 10), desc="Generate v3.0.15 Env"):
        seed = init_seed + i

        env = create_env(flatland_parameters, seed)
        env_wrapper = LocalTestEnvWrapper(env)
        env_wrapper.reset(random_seed=seed)

        rail_grid = env_wrapper.env.rail.grid.astype(np.uint8)
        rail_hash = hash_array(rail_grid)

        agent_info = [(a.initial_position, a.target) for a in env_wrapper.env.agents]
        agent_hash = hashlib.md5(pickle.dumps(agent_info)).hexdigest()

        all_episodes.append({
            "episode_id": i + 1,
            "seed": seed,
            "rail_hash": rail_hash,
            "agent_hash": agent_hash,
            "grid_shape": rail_grid.shape,
            "agent_info": agent_info,
            # Optional storage of original data（for debug）
            "rail_grid": rail_grid
        })

        tqdm.write(f"Episode {i+1}, Seed {seed} Stored.")

    with open(save_path, "wb") as f:
        pickle.dump(all_episodes, f)
    print(f"✅ Flatland v{flatland.__version__} test env data is saved at {save_path}.")