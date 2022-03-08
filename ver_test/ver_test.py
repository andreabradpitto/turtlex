#!/usr/bin/env python3

# Import necessary libraries
import torch
import pynever.strategies.conversion as conv


net_id = ['orig1180_policy_net','5020_policy_net', '1600_policy_net']  # Names of the network to verify

netspath = "nav_nets/"  # Specify networks directory path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CUDA, if available

pytorch_net = conv.PyTorchNetwork(net_id[0], torch.load(netspath + net_id[0] + ".pth", map_location=device))
net = conv.PyTorchConverter().to_neural_network(pytorch_net)

print("DONE")
