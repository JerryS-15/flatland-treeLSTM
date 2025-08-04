import pytest
import numpy as np

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
from utils import debug_show

@pytest.mark.unit
@pytest.mark.parametrize(
    "n_agents, regen_schedule, regen_rail, seed, schedule_match, rail_match",
    [
        (5, True, False, 42, False, True),
        (5, True, True, 42, False, True),
        (5, True, True, None, False, False),
        (5, False, True, None, False, False),
        (5, False, False, 42, True, True),
    ],
)

def test_consistent_schedule_on_reset(
    environment_factory,
    agent_factory,
    n_agents,
    regen_schedule,
    regen_rail,
    seed,
    schedule_match,
    rail_match,
    ) -> None:
    """Check if all_rewards_at_end put the cumulative reward at end."""
    norm_factor = 30
    config_update = {
        "n_agents": n_agents,
        "regenerate_schedule": regen_schedule,
        "regenerate_rail": regen_rail,
        "obs_config": {
            "type": "graph_observation_converter",
            "required_common_info": [OwnPositionCommonInfo()],
            "converters_to_include": [
                OwnPositionObservationConverter(norm_factor),
                OwnHeadNodeTypeObservationConverter(norm_factor),
            ],
            "preprocessor_type": "feature_selection_preprocessor",
        },
        "reward_config": {"converters_to_include": [DestinationReachedRewardConverter()]},
        "seed": seed,
    }
    env = environment_factory(
        DEFAULT_CONTEXT,
        config_update,
    )
    
def get_environment_schedule_features(rail_schedule):
    """Returns the environment schedule features as a tuple."""
    all_keys = [mem for mem in dir(env.env.agents[0]) if not mem.startswith("_")]
    schedule_features = defaultdict(lambda: [])
    for agent_h in env.env.agents:
        agent_handle = agent_h.handle
        for m_key in all_keys:
            try:
                m_value = getattr(env.env.agents[agent_handle], m_key)
                if callable(m_value):
                    continue
                if not isinstance(m_value, Real) and isinstance(m_value, tuple):
                    continue
                schedule_features[agent_handle].append(
                    getattr(env.env.agents[agent_handle], m_key)
                )
            except Exception:
                continue
    return schedule_features

obs = env.reset()
agent = agent_factory("random_agent")
schedule_before = get_environment_schedule_features(env.env.agents)
rail_before = deepcopy(env.env.rail.grid)
for _step in range(5):
    _action = agent(obs)
    obs, all_rewards, done, info = env.step(_action)
    if done["__all__"]:
        break
    env.reset()
    rail_after = env.env.rail.grid
    assert np.equal(rail_before, rail_after).all() == rail_match
    schedule_after = get_environment_schedule_features(env.env.agents)
    assert (schedule_before == schedule_after) == schedule_match