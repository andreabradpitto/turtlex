#!/usr/bin/env python3

# Import necessary libraries
import time
import logging
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Normal
import math
import numpy as np
import pynever.strategies.conversion as conv
import pynever.strategies.verification as ver


def weights_init_(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class PolicyNetwork(torch.nn.Module):  # PyTorch needs this definition in order to know how load the .pth file correctly
    def __init__(self, state_dim, action_dim, actor_hidden_dim, log_std_min=-20, log_std_max=2):

        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = torch.nn.Linear(state_dim, actor_hidden_dim)
        self.linear2 = torch.nn.Linear(actor_hidden_dim, actor_hidden_dim)

        self.mean_linear = torch.nn.Linear(actor_hidden_dim, action_dim)
        self.log_std_linear = torch.nn.Linear(actor_hidden_dim, action_dim)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = (2 * torch.sigmoid(2 * x_t)) - 1
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std

# def init_const_weights(m, w_val: float, b_val: float):
#     if isinstance(m, torch.nn.Linear):
#         torch.nn.init.constant_(m.weight, w_val)
#         torch.nn.init.constant_(m.bias, b_val)  # m.bias.data.fill_(b_val)



class CompActor(torch.nn.Module):  # COMPatible ACTOR (with pyNeVer)

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):

        super(CompActor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)  # "* 2" layer
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)  # Ex "mean" layer
        self.fc5 = torch.nn.Linear(action_dim, action_dim)  # "* 2; - 1" layer

        torch.nn.init.constant_(self.fc3.weight, 2)
        torch.nn.init.constant_(self.fc3.bias, 0)
        torch.nn.init.constant_(self.fc5.weight, 2)
        torch.nn.init.constant_(self.fc5.bias, -1)

    def forward(self, x):

        z = torch.relu(self.fc1(x))
        z = torch.relu(self.fc2(z))
        z = self.fc3(z)
        z = torch.sigmoid(self.fc4(z))
        z = self.fc5(z)
        return z


if __name__ == "__main__":

    pol_net_id = 'prev_2120_policy_net'

    netspath = "nav_nets/"  # Specify networks directory path

    state_dim = 14
    hidden_dim = 30
    action_dim = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CUDA, if available

    policy_net = torch.load(netspath + pol_net_id + ".pth", map_location=device)
    policy_net.eval()
    
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(1, 5),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(5,1),
    #     torch.nn.LogSigmoid(),
    #     ).to(device)

    # model.apply(init_const_weights(2, 0))


    new_policy_net = CompActor(state_dim, hidden_dim, action_dim).to(device)

    # Display all model layer weights
    for name, para in policy_net.named_parameters():
        print('{}: {}'.format(name, para.shape))
        #print(para.size())
        #print(para)
        print("")

    #weights = dict()
    #for name, para in policy_net.named_parameters():
    #    weights[name] = para
    #print(weights)

    with torch.no_grad():
        new_policy_net.fc1.weight.copy_(policy_net.linear1.weight)
        new_policy_net.fc1.bias.copy_(policy_net.linear1.bias)
        new_policy_net.fc2.weight.copy_(policy_net.linear2.weight)
        new_policy_net.fc2.bias.copy_(policy_net.linear2.bias)
        new_policy_net.fc4.weight.copy_(policy_net.mean_linear.weight)
        new_policy_net.fc4.bias.copy_(policy_net.mean_linear.bias)
    new_policy_net.eval()

    # Test
    inputs = torch.randn([1, 10, state_dim])

    outputs_old, _ = policy_net.forward(inputs)
    outputs_new = new_policy_net.forward(inputs)

    print(
    f"""

    outputs_old:
    {outputs_old}

    """)
    print(
    f"""

    outputs_new:
    {outputs_new}

    """)

    print(outputs_new==outputs_old)

    
    #pytorch_net = conv.PyTorchNetwork(net_id, torch.load(netspath + net_id + ".pth", map_location=device))
    #net = conv.PyTorchConverter().to_neural_network(pytorch_net)
