import torch
import torch.nn as nn
import torch.nn.functional as F

# # ========== Q - Network with imitation  ==========
# class FC_Q(nn.Module):
#     def __init__(self, state_dim, num_actions):
#         super(FC_Q, self).__init__()
#         self.q1 = nn.Linear(state_dim, 256)
#         self.q2 = nn.Linear(256, 256)
#         self.q3 = nn.Linear(256, num_actions)

#         self.i1 = nn.Linear(state_dim, 256)
#         self.i2 = nn.Linear(256, 256)
#         self.i3 = nn.Linear(256, num_actions)		

#     def forward(self, state):
#         q = F.relu(self.q1(state))
#         q = F.relu(self.q2(q))
#         q = self.q3(q)

#         i = F.relu(self.i1(state))
#         i = F.relu(self.i2(i))
#         i = self.i3(i)

#         print(f"i shape: {i.shape}")

#         return q, F.log_softmax(i, dim=1), i

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)  # output_dim = 50 * 5
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).view(-1, 50, 5)  # 50 agents
    
class BehaviorCloneNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BehaviorCloneNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)  # output_dim = 50 * 5

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x).view(-1, 50, 5)  # 50 agents