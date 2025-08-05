import pytest
import numpy as np
import hashlib
import pickle

@pytest.mark.parametrize("v1_env", ["test/env_v2.pkl"])
@pytest.mark.parametrize("v2_env", ["test/env_v3.pkl"])
def test_map_version(v1_env, v2_env):
    with open(v1_env, "rb") as f1:
        v1_env_data = pickle.load(f1)
    
    with open(v2_env, "rb") as f2:
        v2_env_data = pickle.load(f2)

    assert len(v1_env_data) == len(v2_env_data), "❌ Episode count mismatch"

    for idx, (e1, e2) in enumerate(zip(v1_env_data, v2_env_data), start=1):
        seed1 = e1.get("seed")
        seed2 = e2.get("seed")
        assert seed1 == seed2, f"❌ [Episode {idx}] seed mismatch: {seed1} vs {seed2}"

        grid_shape1 = e1.get("grid_shape")
        grid_shape2 = e2.get("grid_shape")
        assert grid_shape1 == grid_shape2, f"❌ [Episode {idx} seed {seed2}] rail grid shape mismatch"

        rail_hash1 = e1.get("rail_hash")
        rail_hash2 = e2.get("rail_hash")
        assert rail_hash1 == rail_hash2, f"❌ [Episode {idx} seed {seed2}] rail grid mismatch"

        agent_hash1 = e1.get("agent_hash")
        agent_hash2 = e2.get("agent_hash")
        if agent_hash1 != agent_hash2:
            # Details of info
            ainfo1 = e1.get("agent_info")
            ainfo2 = e2.get("agent_info")
            pytest.fail(f"❌ [Episode {idx} seed {seed2}] agent info mismatch:\n"
                        f"  hash v2: {agent_hash1}\n"
                        f"  hash v3: {agent_hash2}\n"
                        f"  v2 agents: {ainfo1}\n"
                        f"  v3 agents: {ainfo2}")
