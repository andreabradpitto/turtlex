#!/usr/bin/env python3

# Import necessary libraries
#import time
#import logging
#import math
#import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
#from torch.autograd import Variable
import onnx
import pynever.strategies.conversion as conv
import pynever.strategies.verification as ver
import pynever.networks as networks
import pynever.nodes as nodes

# Deterministic/Stochastic process (could also have been placed under if __name__ == "__main__")
# seed = 0
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

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


class CompActor(torch.nn.Module):  # COMPatible ACTOR (actor net compatible with pyNeVer)

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):

        super(CompActor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)  # "* 2" layer
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)  # Ex "mean" layer
        self.fc5 = torch.nn.Linear(action_dim, action_dim)  # "* 2; - 1" layer

        torch.nn.init.constant_(self.fc3.weight, 0)
        torch.nn.init.constant_(self.fc3.bias, 0)
        torch.nn.init.constant_(self.fc5.weight, 0)
        torch.nn.init.constant_(self.fc5.bias, -1)

        with torch.no_grad():
            for idx, elem in enumerate(self.fc3.weight):
                elem[idx] = 2
            for idx, elem in enumerate(self.fc5.weight):
                elem[idx] = 2

    def forward(self, x):

        z = torch.relu(self.fc1(x))
        z = torch.relu(self.fc2(z))
        z = self.fc3(z)
        z = torch.sigmoid(self.fc4(z))
        z = self.fc5(z)
        return z

# def init_const_weights(m):
#     if isinstance(m, torch.nn.Linear):
#         torch.nn.init.constant_(m.weight, 2/5)
#         torch.nn.init.constant_(m.bias, 0)  # m.bias.data.fill_(0)

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
    #     torch.nn.Linear(state_dim, hidden_dim),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(hidden_dim, action_dim),
    #     torch.nn.LogSigmoid(),
    #     ).to(device)
    # model.apply(init_const_weights)
    # model.eval()


    new_policy_net = CompActor(state_dim, hidden_dim, action_dim).to(device)

    # # Display all model layer weights
    # for name, para in policy_net.named_parameters():
    #     print('{}: {}'.format(name, para.shape))
    #     #print(para.size())
    #     #print(para)
    #     print("")

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

    torch.save(new_policy_net, netspath + "compactor_" + pol_net_id + ".pth")


    ############# Test

    # Define inputs - loaded (old) net
    state = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.]
    state = np.asarray(state)
    state = np.float32(state)
    state = torch.FloatTensor(state).to(device).unsqueeze(0)
    print(f"state:\n{state}\n")

    # Define inputs - pynever-compatible (new) net
    #inputs = torch.randn([1, 10, state_dim])
    inputs = torch.ones([14])
    inputs[12] = 0.
    inputs[13] = 0.
    # inputs = torch.ones([1, 14])
    print(f"inputs:\n{inputs}\n")

    # Acquire outputs for the loaded (old) net
    _, _, action, _ = policy_net.sample(state)  # == outputs_old, _ = policy_net.forward(inputs)
    action = (2 * torch.sigmoid(2 * action)) - 1
    outputs_old = action.detach().cpu().numpy()[0]

    # Acquire outputs for the pynever-compatible (new) net
    outputs_new = new_policy_net.forward(inputs)

    # outputs_model = model.forward(inputs)

    print(f"outputs_old:\n{outputs_old}\n")
    print(f"outputs_new:\n{outputs_new}\n")
    # print(f"outputs_model:\n{outputs_model}\n")
    print(f"Output equivalence = {outputs_new.data==torch.tensor(outputs_old).data}\n")


    # pytorch_net = conv.PyTorchNetwork("pytorch_net", torch.load(netspath + "compactor_" + pol_net_id + ".pth", map_location=device))
    # net = conv.PyTorchConverter().to_neural_network(pytorch_net)
    # print("success")



    pol_new_pnv = networks.SequentialNetwork('NET_0', "X")
    fc1 = nodes.FullyConnectedNode("FC1", (state_dim,), hidden_dim)
    pol_new_pnv.add_node(fc1)
    rl2 = nodes.ReLUNode("RL2", fc1.out_dim)
    pol_new_pnv.add_node(rl2)
    fc3 = nodes.FullyConnectedNode("FC3", rl2.out_dim, hidden_dim)
    pol_new_pnv.add_node(fc3)
    rl4 = nodes.ReLUNode("RL4", fc3.out_dim)
    pol_new_pnv.add_node(rl4)
    fc5 = nodes.FullyConnectedNode("FC5", rl4.out_dim, hidden_dim)  # "* 2" layer
    pol_new_pnv.add_node(fc5)
    fc6 = nodes.FullyConnectedNode("FC6", fc5.out_dim, action_dim)  # Ex "mean" layer
    pol_new_pnv.add_node(fc6)
    sm7 = nodes.SigmoidNode("SM7", fc6.out_dim)
    pol_new_pnv.add_node(sm7)
    fc8 = nodes.FullyConnectedNode("FC8", sm7.out_dim, action_dim)  # "* 2; - 1" layer
    pol_new_pnv.add_node(fc8)

    pol_new_pnv_pt = conv.PyTorchConverter().from_neural_network(pol_new_pnv)
    # If I do not specify ".pytorch_network" at the end, I do not grab the actual/real pytorch network, but a PyTorchNetwork()

    torch.nn.init.constant_(pol_new_pnv_pt.pytorch_network._modules['4'].weight, 0)
    torch.nn.init.constant_(pol_new_pnv_pt.pytorch_network._modules['4'].bias, 0)
    torch.nn.init.constant_(pol_new_pnv_pt.pytorch_network._modules['7'].weight, 0)
    torch.nn.init.constant_(pol_new_pnv_pt.pytorch_network._modules['7'].bias, -1)

    with torch.no_grad():
        for idx, elem in enumerate(pol_new_pnv_pt.pytorch_network._modules['4'].weight):
            elem[idx] = 2
        for idx, elem in enumerate(pol_new_pnv_pt.pytorch_network._modules['7'].weight):
            elem[idx] = 2

    with torch.no_grad():
        pol_new_pnv_pt.pytorch_network._modules['0'].weight.copy_(policy_net.linear1.weight)
        pol_new_pnv_pt.pytorch_network._modules['0'].bias.copy_(policy_net.linear1.bias)
        pol_new_pnv_pt.pytorch_network._modules['2'].weight.copy_(policy_net.linear2.weight)
        pol_new_pnv_pt.pytorch_network._modules['2'].bias.copy_(policy_net.linear2.bias)
        pol_new_pnv_pt.pytorch_network._modules['5'].weight.copy_(policy_net.mean_linear.weight)
        pol_new_pnv_pt.pytorch_network._modules['5'].bias.copy_(policy_net.mean_linear.bias)
    pol_new_pnv_pt.pytorch_network.eval()  # Not strictly necessary here

    outputs_pnv_pt = pol_new_pnv_pt.pytorch_network.forward(inputs.double())
    print(f"outputs_pnv_pt:\n{outputs_pnv_pt}\n")
    print(f"Output equivalence = {outputs_new==outputs_pnv_pt.float()}\n")

    torch.save(pol_new_pnv_pt.pytorch_network, netspath + "pol_new_pnv" + ".pth")
    # TODO ma sembra create lo stessa rete anche con torch.save(pol_new_pnv_pt.pytorch_network, netspath + "pol_new_pnv" + ".pth")

    pol_new_pnv = conv.PyTorchConverter().to_neural_network(pol_new_pnv_pt)

    pol_new_pnv_onnx = conv.ONNXConverter().from_neural_network(pol_new_pnv).onnx_network
    onnx.save(pol_new_pnv_onnx, netspath + "pol_new_pnv" + ".onnx")
