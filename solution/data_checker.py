import pickle

file_path = 'offlineData/offline_rl_data_treeLSTM_5_agents_50_episodes_normR.pkl'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Successfully loading data from '{file_path}' .")
    print("Data type:", type(data))
    print("Data len: ", len(data))

    for i in range (0, 1000):
        print("printout test:", i)
        print(data[i][2])

    """
    Data type: <class 'list'>
    Data len:  41497
    """

    print("Each data term (data[i]) type:", type(data[0]), " ", len(data[0]))
    print("For each data term (data[i][j]): ", type(data[0][0]), " ", len(data[0][0]))
    """
    Each data term (data[i]) type: <class 'tuple'>   5
    For each data term (data[i][j]):  <class 'list'>   1
    """

    print("obs type:", type(data[0][0][0]))
    print("obs len:", len(data[0][0][0]))
    print("obs keys:", data[0][0][0].keys())
    """
    obs type: <class 'dict'>
    obs len: 17
    obs keys: dict_keys(['agent_attr', 'forest', 'adjacency', 'node_order', 'edge_order', 
    'curr_step', 'height', 'max_timesteps', 'n_agents', 'width', 'deadlocked', 'dist_target', 
    'earliest_departure', 'latest_arrival', 'ready_not_depart', 'speed', 'valid_actions'])
    """

    print("obs[node_order] type: ", type(data[0][0][0]['node_order']))
    print("obs[node_order] len: ", len(data[0][0][0]['node_order']))
    print("obs[node_order]: ", data[0][0][0]['node_order'].shape)

    print("actions type:", type(data[0][1]))
    print("actions len:", len(data[0][1]))
    print("actions keys:", data[0][1].keys())
    """
    actions type: <class 'dict'>
    actions len: 50
    actions keys: dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
    """
    # print(data[0][1])
    # print(data[0][0][0]['agent_attr'])

    # count = 0
    # for i in range(0, len(data)):
        # print(data[i][4]['__all__'])
    # print("Number of count:",  count)

    print("done type:", type(data[0][4]))
    # for d in data:
    #     print("done: ", d[4])

except FileNotFoundError:
    print(f"ERROR: File '{file_path}' Not Found! Please check the file path!")
except Exception as e:
    print(f"Error exists when loading file: {e}")