import pytest
import numpy as np
import hashlib
import pickle
import os

# helper functions
def load_env_data(version, seed):
    path = f"test_env_data_{version}/env_{version}_{seed}.pkl"
    assert os.path.exists(path), f"❌ Missing file: {path}"
    with open(path, "rb") as f:
        return pickle.load(f)

def rail_grids_equal(rail1, rail2):
    return np.array_equal(rail1.grid, rail2.grid)

def agents_equal(agent_v2, agent_v3, atol=1e-5):
    if len(agent_v2) != len(agent_v3):
        return False
    for a1, a2 in zip(agent_v2, agent_v3):
        if a1["initial_position"] != a2["initial_position"]:
            return False
        if a1["target"] != a2["target"]:
            return False
        if a1["direction"] != a2["direction"]:
            return False
        if not np.isclose(a1["speed"]["speed"], a2["speed"]["speed"], atol=atol):
            return False
    return True

def stations_equal(station_data_1, station_data_2):
    return sorted(station_data_1["stations"]) == sorted(station_data_2["stations"])


# parameterize seeds
@pytest.mark.parametrize("seed", list(range(42, 52)))
def test_envs_match(seed):
    v2 = load_env_data("v2", seed)
    v3 = load_env_data("v3", seed)

    # 1. Compare rail grids
    assert rail_grids_equal(v2["rail"], v3["rail"]), f"❌ Rail grid mismatch at seed={seed}"

    # 2. Compare agent info
    assert agents_equal(v2["agent_info"], v3["agent_info"]), f"❌ Agent info mismatch at seed={seed}"

    # 3. Compare stations
    assert stations_equal(v2.get("stations", {}), v3.get("stations", {})), f"❌ Stations mismatch at seed={seed}"

# @pytest.mark.parametrize("v1_env", ["test/env_v2.pkl"])
# @pytest.mark.parametrize("v2_env", ["test/env_v3.pkl"])
# def test_map_version(v1_env, v2_env):
#     with open(v1_env, "rb") as f1:
#         v1_env_data = pickle.load(f1)
    
#     with open(v2_env, "rb") as f2:
#         v2_env_data = pickle.load(f2)

#     assert len(v1_env_data) == len(v2_env_data), "❌ Episode count mismatch"

#     for idx, (e1, e2) in enumerate(zip(v1_env_data, v2_env_data), start=1):
#         seed1 = e1.get("seed")
#         seed2 = e2.get("seed")
#         assert seed1 == seed2, f"❌ [Episode {idx}] seed mismatch: {seed1} vs {seed2}"

#         grid_shape1 = e1.get("grid_shape")
#         grid_shape2 = e2.get("grid_shape")
#         assert grid_shape1 == grid_shape2, f"❌ [Episode {idx} seed {seed2}] rail grid shape mismatch"

#         rail_hash1 = e1.get("rail_hash")
#         rail_hash2 = e2.get("rail_hash")
#         assert rail_hash1 == rail_hash2, f"❌ [Episode {idx} seed {seed2}] rail grid mismatch"

#         agent_hash1 = e1.get("agent_hash")
#         agent_hash2 = e2.get("agent_hash")
#         if agent_hash1 != agent_hash2:
#             # Details of info
#             ainfo1 = e1.get("agent_info")
#             ainfo2 = e2.get("agent_info")
#             pytest.fail(f"❌ [Episode {idx} seed {seed2}] agent info mismatch:\n"
#                         f"  hash v2: {agent_hash1}\n"
#                         f"  hash v3: {agent_hash2}\n"
#                         f"  v2 agents: {ainfo1}\n"
#                         f"  v3 agents: {ainfo2}")
