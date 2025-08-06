import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow
import argparse

from test_env_rebuild import rebuild_env_from_data
    
def visualize_single_env(ax, grid, agent_info=None, title="", color="blue"):
    ax.imshow(grid, cmap="Greys", interpolation="none")

    if agent_info is not None:
        for agent in agent_info:
            if agent["initial_position"] is None or agent["target"] is None:
                continue

            x, y = agent["initial_position"][1], agent["initial_position"][0]
            tx, ty = agent["target"][1], agent["target"][0]
            direction = agent["direction"]

            # Initial position
            ax.plot(x, y, "o", color="green", markersize=4)
            # Target position
            ax.plot(tx, ty, "x", color="red", markersize=4)

            # Direction arrow
            dx, dy = {
                0: (0, -0.5),
                1: (0.5, 0),
                2: (0, 0.5),
                3: (-0.5, 0),
            }[direction]
            arrow = FancyArrow(x, y, dx, dy, width=0.05, color=color)
            ax.add_patch(arrow)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def compare_envs(pkl_path_v2, output_path=None):
    with open(pkl_path_v2, "rb") as f:
        data_v2 = pickle.load(f)
    
    grid_v2 = data_v2["rail"].grid
    agent_info_v2 = data_v2["agent_info"]
    seed = data_v2["seed"]

    env_v3 = rebuild_env_from_data(data_v2)
    grid_v3 = env_v3.rail.grid
    agent_info_v3 = [
        {
            "initial_position": agent.initial_position,
            "target": agent.target,
            "direction": agent.direction,
            "speed": getattr(agent, "speed", 1.0)  # default 1.0 to avoid error
        }
        for agent in env_v3.agents
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    visualize_single_env(axes[0], grid_v2, agent_info=agent_info_v2, title=f"v2 from pkl (seed={seed})", color="blue")
    visualize_single_env(axes[1], grid_v3, agent_info=agent_info_v3, title=f"v3 rebuilt (seed={seed})", color="orange")

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"✅ Saved comparison to: {output_path}")
    else:
        plt.show()

def compare_env_with_same_seed(pkl_path_v2, pkl_path_v3, output_path=None):
    with open(pkl_path_v2, "rb") as f:
        data_v2 = pickle.load(f)

    with open(pkl_path_v3, "rb") as f:
        data_v3 = pickle.load(f)
    
    grid_v2 = data_v2["rail"].grid
    agent_info_v2 = data_v2["agent_info"]
    seed_v2 = data_v2["seed"]

    grid_v3 = data_v3["rail"].grid
    agent_info_v3 = data_v3["agent_info"]
    seed_v3 = data_v3["seed"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    visualize_single_env(axes[0], grid_v2, agent_info=agent_info_v2, title=f"v2 env (seed={seed_v2})", color="blue")
    visualize_single_env(axes[1], grid_v3, agent_info=agent_info_v3, title=f"v3 env (seed={seed_v3})", color="orange")

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"✅ Saved comparison with same seed to: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store_true", help="Compare the environment generated with the same seed.")
    args = parser.parse_args()

    if args.seed:
        output_dir = "compare_envs_with_seed"
        os.makedirs(output_dir, exist_ok=True)

        for seed in range(42, 52):
            output_path = os.path.join(output_dir, f"env_compare_seed{seed}.png")

            compare_env_with_same_seed(
                pkl_path_v2=f"test_env_data_v2/env_v2_{seed}.pkl",
                pkl_path_v3=f"test_env_data_v3/env_v3_{seed}.pkl",
                output_path=output_path
            )
    else:
        output_dir = "compare_envs"
        os.makedirs(output_dir, exist_ok=True)

        for seed in range(42, 52):
            output_path = os.path.join(output_dir, f"env_compare_seed{seed}.png")

            compare_envs(
                pkl_path_v2=f"test_env_data_v2/env_v2_{seed}.pkl",
                output_path=output_path
            )