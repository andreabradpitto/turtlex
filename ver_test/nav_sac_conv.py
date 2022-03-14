#!/usr/bin/env python3

# Import necessary libraries
import torch
import onnx
import pynever.strategies.conversion as conv
import pynever.networks as networks
import pynever.nodes as nodes
import torch.nn.functional as F
from torch.distributions import Normal


class PolicyNetwork(torch.nn.Module):  # PyTorch needs this definition in order to know how load the .pth file correctly
    def __init__(self, state_dim, action_dim, actor_hidden_dim, log_std_min=-20, log_std_max=2):

        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = torch.nn.Linear(state_dim, actor_hidden_dim)
        self.linear2 = torch.nn.Linear(actor_hidden_dim, actor_hidden_dim)

        self.mean_linear = torch.nn.Linear(actor_hidden_dim, action_dim)
        self.log_std_linear = torch.nn.Linear(actor_hidden_dim, action_dim)

        self.apply(self.weights_init)

    def weights_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

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


def pnv_converter(pol_net_id: str, netspath: str, state_dim: int, hidden_dim: int, action_dim: int, device: str):
    """
    This function searches for the trained and saved network, and returns 3 PyNeVer-compatible
    versions of the same new network: PyNeVer internal format, PyTorch format, and ONNX format
    """

    pol_net_id = 'prev_2120_policy_net'  # Specify the network to convert

    netspath = "nav_sac_nets/"  # Specify networks directory path

    state_dim = 14
    hidden_dim = 30
    action_dim = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CUDA, if available

    policy_net = torch.load(netspath + pol_net_id + ".pth", map_location=device)  # This line needs the PolicyNetwork definition above
    policy_net.eval()

    pol_new_pnv = networks.SequentialNetwork('NET_0', "X")
    fc1 = nodes.FullyConnectedNode("FC1", (state_dim,), hidden_dim)
    pol_new_pnv.add_node(fc1)
    rl2 = nodes.ReLUNode("RL2", fc1.out_dim)
    pol_new_pnv.add_node(rl2)
    fc3 = nodes.FullyConnectedNode("FC3", rl2.out_dim, hidden_dim)
    pol_new_pnv.add_node(fc3)
    rl4 = nodes.ReLUNode("RL4", fc3.out_dim)
    pol_new_pnv.add_node(rl4)
    fc5 = nodes.FullyConnectedNode("FC5", rl4.out_dim, action_dim)  # Ex "mean" layer
    pol_new_pnv.add_node(fc5)
    fc6 = nodes.FullyConnectedNode("FC6", fc5.out_dim, action_dim)  # "* 2" layer
    pol_new_pnv.add_node(fc6)
    sm7 = nodes.SigmoidNode("SM7", fc6.out_dim)
    pol_new_pnv.add_node(sm7)
    fc8 = nodes.FullyConnectedNode("FC8", sm7.out_dim, action_dim)  # "* 2; - 1" layer
    pol_new_pnv.add_node(fc8)

    pol_new_pnv_pt = conv.PyTorchConverter().from_neural_network(pol_new_pnv)

    torch.nn.init.constant_(pol_new_pnv_pt.pytorch_network._modules['5'].weight, 0)
    torch.nn.init.constant_(pol_new_pnv_pt.pytorch_network._modules['5'].bias, 0)
    torch.nn.init.constant_(pol_new_pnv_pt.pytorch_network._modules['7'].weight, 0)
    torch.nn.init.constant_(pol_new_pnv_pt.pytorch_network._modules['7'].bias, -1)

    with torch.no_grad():
        for idx, elem in enumerate(pol_new_pnv_pt.pytorch_network._modules['5'].weight):
            elem[idx] = 2
        for idx, elem in enumerate(pol_new_pnv_pt.pytorch_network._modules['7'].weight):
            elem[idx] = 2

    with torch.no_grad():
        pol_new_pnv_pt.pytorch_network._modules['0'].weight.copy_(policy_net.linear1.weight)
        pol_new_pnv_pt.pytorch_network._modules['0'].bias.copy_(policy_net.linear1.bias)
        pol_new_pnv_pt.pytorch_network._modules['2'].weight.copy_(policy_net.linear2.weight)
        pol_new_pnv_pt.pytorch_network._modules['2'].bias.copy_(policy_net.linear2.bias)
        pol_new_pnv_pt.pytorch_network._modules['4'].weight.copy_(policy_net.mean_linear.weight)
        pol_new_pnv_pt.pytorch_network._modules['4'].bias.copy_(policy_net.mean_linear.bias)
    pol_new_pnv_pt.pytorch_network.eval()  # Not strictly necessary here

    torch.save(pol_new_pnv_pt.pytorch_network, netspath + pol_net_id + "_pnv" + ".pth")

    pol_new_pnv = conv.PyTorchConverter().to_neural_network(pol_new_pnv_pt)
    pol_new_pnv_onnx = conv.ONNXConverter().from_neural_network(pol_new_pnv).onnx_network
    onnx.save(pol_new_pnv_onnx, netspath + pol_net_id + "_pnv" + ".onnx")

    return pol_new_pnv, pol_new_pnv_pt, pol_new_pnv_onnx
