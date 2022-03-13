#!/usr/bin/env python3

# Import necessary libraries
import torch
import onnx
import pynever.strategies.conversion as conv
import pynever.networks as networks
import pynever.nodes as nodes



if __name__ == "__main__":

    pol_net_id = 'prev_2120_policy_net'  # Specify the network to convert

    netspath = "nav_nets/"  # Specify networks directory path

    state_dim = 14
    hidden_dim = 30
    action_dim = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CUDA, if available

    policy_net = torch.load(netspath + pol_net_id + ".pth", map_location=device)
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
    fc5 = nodes.FullyConnectedNode("FC5", rl4.out_dim, hidden_dim)  # "* 2" layer
    pol_new_pnv.add_node(fc5)
    fc6 = nodes.FullyConnectedNode("FC6", fc5.out_dim, action_dim)  # Ex "mean" layer
    pol_new_pnv.add_node(fc6)
    sm7 = nodes.SigmoidNode("SM7", fc6.out_dim)
    pol_new_pnv.add_node(sm7)
    fc8 = nodes.FullyConnectedNode("FC8", sm7.out_dim, action_dim)  # "* 2; - 1" layer
    pol_new_pnv.add_node(fc8)

    pol_new_pnv_pt = conv.PyTorchConverter().from_neural_network(pol_new_pnv)

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

    torch.save(pol_new_pnv_pt.pytorch_network, netspath + pol_net_id + "_pnv" + ".pth")

    pol_new_pnv = conv.PyTorchConverter().to_neural_network(pol_new_pnv_pt)
    pol_new_pnv_onnx = conv.ONNXConverter().from_neural_network(pol_new_pnv).onnx_network
    onnx.save(pol_new_pnv_onnx, netspath + pol_net_id + "_pnv" + ".onnx")
