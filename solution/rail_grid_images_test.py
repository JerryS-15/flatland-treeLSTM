import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

def save_rail_grids(pkl_path, output_dir, version_tag):
    os.makedirs(output_dir, exist_ok=True)

    with open(pkl_path, "rb") as f:
        env_data = pickle.load(f)
    
    for idx, episode in enumerate(env_data):
        rail_grid = episode["rail_grid"]
        seed = episode["seed"]
        plt.figure(figsize=(6, 6))
        plt.imshow(rail_grid, cmap="viridis", interpolation="nearest")
        plt.title(f"Seed {seed} - {version_tag}")
        plt.axis("off")
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{version_tag}_ep{idx+1}_seed{seed}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    save_rail_grids("test/env_v2.pkl", "grids_v2", "v2.2.1")
    save_rail_grids("test/env_v3.pkl", "grids_v3", "v3.0.15")