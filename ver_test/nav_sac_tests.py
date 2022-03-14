#!/usr/bin/env python3

# 3 networks here: PolicyNetwork is the one which training has been performed on. CompActor is a PyTorch replica
# with a more standard definition (e.g. no sample(), but just a forward()), but it currently does not work with
# pynever. Lastly, the 3rd network is created starting from pynever and works fine.
# There is a little difference in the outputs of the 1st and 3rd nets due to the double() state conversion
# needed by the latter, and to the subsequent Tensor .numpy() conversion. This could be solved by re-performing
# training without applying the numpy() operation

# Import necessary libraries
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import onnx
import pynever.strategies.conversion as conv
import pynever.networks as networks
import pynever.nodes as nodes


class PolicyNetwork(torch.nn.Module):  # [1st network] PyTorch needs this definition in order to know how load the .pth file correctly
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


class CompActor(torch.nn.Module):  # [2nd network] COMPatible ACTOR (actor net compatible with pyNeVer)

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):

        super(CompActor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)  # Ex "mean" layer
        self.fc4 = torch.nn.Linear(action_dim, action_dim)  # "* 2" layer
        self.fc5 = torch.nn.Linear(action_dim, action_dim)  # "* 2; - 1" layer

        torch.nn.init.constant_(self.fc4.weight, 0)
        torch.nn.init.constant_(self.fc4.bias, 0)
        torch.nn.init.constant_(self.fc5.weight, 0)
        torch.nn.init.constant_(self.fc5.bias, -1)

        with torch.no_grad():
            for idx, elem in enumerate(self.fc4.weight):
                elem[idx] = 2
            for idx, elem in enumerate(self.fc4.weight):
                elem[idx] = 2

    def forward(self, x):

        z = torch.relu(self.fc1(x))
        z = torch.relu(self.fc2(z))
        z = self.fc3(z)
        z = torch.sigmoid(self.fc4(z))
        z = self.fc5(z)
        return z


if __name__ == "__main__":

    pol_net_id = 'prev_2120_policy_net'  # Specify the network to convert

    netspath = "nav_sac_nets/"  # Specify networks directory path

    state_dim = 14
    hidden_dim = 30
    action_dim = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CUDA, if available


    ############# Inputs definition

    print("\nInputs\n")

    # Define inputs - original input definition
    state = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.]
    state = np.asarray(state)
    state = np.float32(state)
    state = torch.FloatTensor(state).to(device).unsqueeze(0)
    print(f"state:\n{state}\n")

    # Define inputs - new way to define the inputs (currently discarded)
    #inputs = torch.randn([2, 3, 4, state_dim])  # Tip: Tensors can be passed in each shape as long as the last dimension is state_dim
    inputs = torch.ones([14])
    inputs[12] = 0.
    inputs[13] = 0.
    print(f"inputs:\n{inputs}\n")


    ############# [1st network] Original trained network

    print("\n1st net\n")

    # Load the trained and saved network
    policy_net = torch.load(netspath + pol_net_id + ".pth", map_location=device)  # This line needs the PolicyNetwork definition above
    policy_net.eval()

    # Acquire outputs for testing
    _, _, action, _ = policy_net.sample(state)  # == outputs_old, _ = policy_net.forward(inputs)
    action = (2 * torch.sigmoid(2 * action)) - 1
    print(f"outputs_old (before detach()/numpy()):\n{action}\n")
    outputs_old = action.detach().cpu().numpy()[0]

    print(f"outputs_old:\n{outputs_old}\n")


    ############# [2nd network] CompActor (currently not compatible with PyNeVer)

    print("\n2nd net\n")

    new_policy_net = CompActor(state_dim, hidden_dim, action_dim).to(device)

    with torch.no_grad():
        new_policy_net.fc1.weight.copy_(policy_net.linear1.weight)
        new_policy_net.fc1.bias.copy_(policy_net.linear1.bias)
        new_policy_net.fc2.weight.copy_(policy_net.linear2.weight)
        new_policy_net.fc2.bias.copy_(policy_net.linear2.bias)
        new_policy_net.fc3.weight.copy_(policy_net.mean_linear.weight)
        new_policy_net.fc3.bias.copy_(policy_net.mean_linear.bias)
    new_policy_net.eval()

    torch.save(new_policy_net, netspath + pol_net_id + "_compactor" + ".pth")

    # Acquire outputs for testing
    outputs_new = new_policy_net.forward(inputs)
    actions_new_same_scheme = new_policy_net.forward(state)
    print(f"actions_new_same_scheme (before detach()/numpy()):\n{actions_new_same_scheme}\n")
    outputs_new_same_scheme = actions_new_same_scheme.detach().cpu().numpy()[0]

    print(f"outputs_new:\n{outputs_new}\n")
    print(f"outputs_new_same_scheme:\n{outputs_new_same_scheme}\n")

    # The conversion (1st ONNX instruction) does NOT work due to PyNeVer, presumably:
    # cannot currently convert this network to PyNeVer internal representation

    # PyTorch saving
    # pytorch_net = conv.PyTorchNetwork("pytorch_net", torch.load(netspath + "compactor_" + pol_net_id + ".pth", map_location=device))

    # ONNX conversion
    # net = conv.PyTorchConverter().to_neural_network(pytorch_net)
    # net_onnx = conv.ONNXConverter().from_neural_network(net).onnx_network
    # onnx.save(net_onnx, netspath + pol_net_id + "_pnv" + ".onnx")


    ############# [3rd network] PyNeVer network (working with PyNeVer)

    print("\n3rd net\n")

    # [3rd network] PyNeVer network creation
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

    pol_new_pnv_pt = conv.PyTorchConverter().from_neural_network(pol_new_pnv)  # Converting the network to PyTorch format
    # Tip: If I do not specify ".pytorch_network" at the end, I do not grab the actual/real pytorch network, but a PyTorchNetwork()

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
    pol_new_pnv_pt.pytorch_network.eval()

    # PyTorch saving
    torch.save(pol_new_pnv_pt.pytorch_network, netspath + pol_net_id + "_pnv" + ".pth")

    # ONNX conversion + saving
    pol_new_pnv = conv.PyTorchConverter().to_neural_network(pol_new_pnv_pt)
    pol_new_pnv_onnx = conv.ONNXConverter().from_neural_network(pol_new_pnv).onnx_network
    onnx.save(pol_new_pnv_onnx, netspath + pol_net_id + "_pnv" + ".onnx")

    # Acquire outputs for testing
    outputs_pnv_pt = pol_new_pnv_pt.pytorch_network.forward(inputs.double())
    outputs_pnv_pt_same_scheme = pol_new_pnv_pt.pytorch_network.forward(state.double())
    print(f"outputs_pnv_pt_same_scheme (before detach()/numpy()):\n{outputs_pnv_pt_same_scheme}\n")
    outputs_pnv_pt_same_scheme = outputs_pnv_pt_same_scheme.detach().cpu().numpy()[0]

    print(f"outputs_pnv_pt:\n{outputs_pnv_pt}\n")
    print(f"outputs_pnv_pt_same_scheme:\n{outputs_pnv_pt_same_scheme}\n")


    ############# (Some) Equivalence prints relative to the networks' outputs

    #print("\nEquivalences\n")

    #print(f"Output equivalence = {outputs_new.data==torch.tensor(outputs_old).data}\n")
    #print(f"Output equivalence = {outputs_new==outputs_pnv_pt.float()}\n")
    #print(f"Output equivalence = {outputs_new==outputs_pnv_pt}\n")
    #print(f"Output equivalence = {outputs_new_same_scheme==outputs_pnv_pt_same_scheme}\n")
