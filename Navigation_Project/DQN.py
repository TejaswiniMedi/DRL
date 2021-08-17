import torch
import torch.nn as nn
import torch.nn.functional as F
class DQN(nn.Module):
    def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_1_size)
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.fc3 = nn.Linear(hidden_2_size, output_size)

    def forward(self, input):
        hidden = F.leaky_relu(self.fc1(input))
        hidden = F.leaky_relu(self.fc2(hidden))
        output = F.leaky_relu(self.fc3(hidden))
        return output        
    
#----------------------  Exploration and Exploitation strategy ---------------------#  
class Epsilon:
    def __init__(self,epsilon_start,epsilon_decay_rate,epsilon_end):
        self.epsilon_start = epsilon_start
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_end = epsilon_end
    def eps_cal(self,timestep):
        return max(self.epsilon_start*(self.epsilon_decay_rate**timestep),self.epsilon_end)