# import pickle
import dill as pickle
import os

n_agents = 10
n_eps = 1000

NUM_EPISODES = 1000
collect_data_path_name = f"offline_rl_data_treeLSTM_{n_agents}_agents_{NUM_EPISODES}_episodes"

FILE_A = f"orData_agent_{n_agents}_normR/or_data_{n_agents}_agents_{n_eps}_episodes.pkl"
FILE_B = f"offlineData/{collect_data_path_name}_normR.pkl"

merged_file = f'mixData/merged_{n_agents}_agents.pkl'

if not os.path.exists("./mixData"):
        os.makedirs("./mixData")

with open(FILE_A, 'rb') as f1, open(FILE_B, 'rb') as f2:
    data1 = pickle.load(f1)
    data2 = pickle.load(f2)

print(f"File 1 has {len(data1)} steps, type: {type(data1)}")
print(f"File 2 has {len(data2)} steps, type: {type(data2)}")

merged_data = data1 + data2

with open(merged_file, 'wb') as f:
    pickle.dump(merged_data, f)

print(f"✅ Merge Completed. Saved in {merged_file}, with total {len(merged_data)} steps")