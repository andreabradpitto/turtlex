#!/usr/bin/env python3

# Import necessary libraries
import time
import logging
import math
import os
import numpy as np
import torch
import pynever.strategies.verification as ver
from datetime import datetime
from nav_sac_conv import PolicyNetwork, pnv_converter


# Toggle whether to perform complete verification
COMPLETE_VER = False  # default: True

net_id = ['prev_2120_policy_net',]  # Names of the network to verify

property_ids = ["GlobalReach", "Local1", "GlobalPartial", "SpeedThreshold"]  # Definition of properties' names

# Properties explanation
"""
------------------------------------------------------------------------------------------------------------------------------------------------------
When adding a new property, 8 elements must be taken into account: property_ids, eps, delta, local_input, input_lb, input_ub, unsafe_mats, unsafe_vecs
------------------------------------------------------------------------------------------------------------------------------------------------------

-GlobalReach: verify that the given the input bounds to the network, it will always produce outputs inside the output bounds
    Computation of GlobalReach property outputs' constraints:
        1) safe: lin_vel_bounds[0] <= y1 <= lin_vel_bounds[1] ===> unsafe: y1 < lin_vel_bounds[0] OR -y1 < -lin_vel_bounds[1]
        2) safe: ang_vel_bounds[0] <= y2 <= ang_vel_bounds[1] ===> unsafe: y2 < ang_vel_bounds[0] OR -y2 < -ang_vel_bounds[1]

        unsafe_mats       unsafe_vecs
        ----------------------------------
        [ 1, 0]     [ lin_vel_bounds[0]]
        [-1, 0]     [-lin_vel_bounds[1]]
        [ 0, 1]     [ ang_vel_bounds[0]]
        [ 0,-1]     [-ang_vel_bounds[1]]

-Local1: verify that for a local area around local_input (+/- eps), the output is always in an area around its local output (+/- delta)

-GlobalPartial: verify that the network outputs a high angular velocity along with a low linear velocity, when input conditions are met.
               Those input conditions are: at least one of the front laser readings (e.g. the 5th out of 10) provides a low reading
               (i.e. the robot is close to an obstacle)
    Computation of GlobalPartial's output unsafe matrixes/biases (starting from safe area, i.e., low output lin. vel. + high out ang. vel.):
        safe: (lb1 < y1 < lb1 + delta1) AND ((lb2 < y2 < lb2 + delta2) OR (ub2 - delta2 < y2 < ub2))
        ==   (lin_vel_bounds[0] < y1 < lin_vel_bounds[0] + delta[2][0]) AND
                ((ang_vel_bounds[0] < y2 < ang_vel_bounds[0] + delta[2][1]) OR
                (ang_vel_bounds[1] - delta[2][1] < y2 < ang_vel_bounds[1]))

        unsafe:(lin_vel_bounds[0] + delta[2][0] < y1 < lin_vel_bounds[1]) AND
                (ang_vel_bounds[0] + delta[2][1] < y2 < ang_vel_bounds[1] - delta[2][1])
        ==   (y1 < lin_vel_bounds[1] OR -y1 < -lin_vel_bounds[0] - delta[2][0]) AND
                (y2 < ang_vel_bounds[1] - delta[2][1] OR -y2 < -ang_vel_bounds[0] - delta[2][1])
        ==> the unsafe output matrices and vectors are:
            matrixes: [1, 0], [-1, 0], [0, 1], [0, -1]
            vectors: [lin_vel_bounds[1]], [-lin_vel_bounds[0] - delta[2][0]], [ang_vel_bounds[1] - delta[2][1]], [-ang_vel_bounds[0] - delta[2][1]]

-SpeedThreshold: verify that the network decreases linear velocity and increases angular velocity, when input conditions are met.
                 Those input conditions are: at least one of the front laser readings (e.g. the 5th out of 10) provides a low reading
                 (i.e. the robot is close to an obstacle), the previous input linear velocity is over a certain threshold, the previous
                 angular input velocity is under a certain threshold.
    Computation of SpeedThreshold's output unsafe matrixes and biases:
        safe: lb1 < y1 < ub1 - delta1 [delta1 is equal to eps1] AND (lb2 < y2 < delta2 OR delta2 < y2 < ub2) [delta2 is equal to eps2]

        unsafe: ub1 - delta1 < y1 < ub1 AND -delta2 < y2 < delta2
        == (y1 < ub1 - delta1 OR -y1 < -lb1) AND (y2 < delta2 OR -y2 < delta2)
        == (y1 < lin_vel_bounds[1] - delta[3][0] OR -y1 < -lin_vel_bounds[0]) AND (y2 < delta[3][1] OR -y2 < delta[3][1])
        ==> the unsafe output matrices and vectors are:
            matrices: [1, 0], [-1, 0], [0, 1], [0, -1]
            vectors: [lin_vel_bounds[1] - delta[3][0]], [lin_vel_bounds[0]], [delta[3][1]], [delta[3][1]]
"""

# Verification parameters: parameter set ID, heuristic, params, refinement level
ver_params = [
              ["Over-Approx",   ["best_n_neurons", [0]],    None],
              ["Mixed",         ["best_n_neurons", [1]],    None],
              ["Mixed2",        ["best_n_neurons", [2]],    None],
              ["Complete",      ["best_n_neurons", [100]],  None]
             ]


# Specify logs directory path
logpath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'results/ver_logs/office_nav_sac'))

# Loggers and handlers definition and setup
#logger_empty = logging.getLogger("pynever.strategies.abstraction.empty_times")
#logger_lp = logging.getLogger("pynever.strategies.abstraction.lp_times")
#logger_lb = logging.getLogger("pynever.strategies.abstraction.lb_times")
#logger_ub = logging.getLogger("pynever.strategies.abstraction.ub_times")
#logger_train = logging.getLogger("pynever.strategies.training")
logger_nav_file = logging.getLogger("nav_ver_log_file")
logger_nav_stream = logging.getLogger("pynever.strategies.verification")

#logger_empty.setLevel(logging.DEBUG)
#logger_lp.setLevel(logging.DEBUG)
#logger_lb.setLevel(logging.DEBUG)
#logger_ub.setLevel(logging.DEBUG)
#logger_train.setLevel(logging.DEBUG)
logger_nav_file.setLevel(logging.INFO)
logger_nav_stream.setLevel(logging.INFO)

#empty_handler = logging.FileHandler(logpath + "/empty_times.txt")
#lp_handler = logging.FileHandler(logpath + "/lp_times.txt")
#lb_handler = logging.FileHandler(logpath + "/lb_times.txt")
#ub_handler = logging.FileHandler(logpath + "/ub_times.txt")
#train_handler = logging.FileHandler(logpath + "/navTrainLog.txt")
nav_file_handler = logging.FileHandler(logpath + "/nav_sac_ver_log.txt")
nav_stream_handler = logging.StreamHandler()

#empty_handler.setLevel(logging.DEBUG)
#lp_handler.setLevel(logging.DEBUG)
#lb_handler.setLevel(logging.DEBUG)
#ub_handler.setLevel(logging.DEBUG)
#train_handler.setLevel(logging.DEBUG)
nav_file_handler.setLevel(logging.INFO)
nav_stream_handler.setLevel(logging.INFO)

#logger_empty.addHandler(empty_handler)
#logger_lp.addHandler(lp_handler)
#logger_lb.addHandler(lb_handler)
#logger_ub.addHandler(ub_handler)
#logger_train.addHandler(train_handler)
logger_nav_file.addHandler(nav_file_handler)
logger_nav_stream.addHandler(nav_stream_handler)

#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#nav_stream_handler.setFormatter(formatter)

logger_nav_file.info(f"Net_ID,Property,Param_set,Safe,Time_elapsed\n")  # Write legend for the log file as its first row

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CUDA, if available

# Store and format the start time of the whole process
start_time = datetime.now()
f_start_time = start_time.strftime("%H:%M:%S")

if not COMPLETE_VER:  # If False, skip "Complete" verification parameter set
    ver_params = ver_params[:-1]

# Input and output bounds declaration
laser_bounds = [0.10000000149011612, 30.0]
heading_bounds = [-math.pi, math.pi]
distance_bounds = [0.0, 18.02775637732]
lin_vel_bounds = [0.0, 0.22]
ang_vel_bounds = [-2.0, 2.0]

# Network dimensions definition
state_dim = 14  # input layer dimension
hidden_dim = 30  # hidden layers dimension
action_dim = 2  # output layer dimension

# Local input tolerance: laser scans (x10), goal heading, goal distance, linear velocity, angular velocity
# These tolerances define the input area to verify; the higher the value, the more general the property is
eps = [
       [],  # GlobalReach: a global property does not feature input tolerances
       [0.1, 0.2, 0.1, 0.05, 0.2],  # Local1: input tolerance for lasers (x10), goal heading, goal distance, linear velocity, angular velocity
       [0.05],  # GlobalPartial: input tolerance just for a single laser scan
       [0.05, 0.02, 0.2]  # SpeedThreshold: input tolerance for a single laser scan, linear and angular input velocities
      ]
# Local output tolerance: linear velocity, angular velocity.
# These tolerances define the output area to verify; the higher the value, the more general the property is
delta = [
         [],  # GlobalReach: a global property does not feature output tolerances
         [0.05, 0.2],  # Local1: output tolerance for linear velocity, angular velocity
         [0.11, 1],  # GlobalPartial: output tolerance for linear velocity, angular velocity
         [0.02, 0.2]  # SpeedThreshold: output tolerance for linear velocity, angular velocity
        ]

# Define local inputs
local_input = [
               [],  # GlobalReach: a global property does not feature local inputs (and outputs)
               [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0, 0.1, 0.21, 0.0],  # Local1: local value for all the inputs
               [],  # GlobalPartial: this property does not feature local inputs (and outputs)
               []  # SpeedThreshold: this property does not feature local inputs (and outputs)
              ]

# Definition of inputs' lower and upper bounds for each property
input_lb = [
            [laser_bounds[0], laser_bounds[0], laser_bounds[0], laser_bounds[0], laser_bounds[0], laser_bounds[0], laser_bounds[0],\
                laser_bounds[0], laser_bounds[0], laser_bounds[0], heading_bounds[0], distance_bounds[0],\
                    lin_vel_bounds[0], ang_vel_bounds[0]],  # GlobalReach: all the sensors' lower bounds

            [local_input[1][0] - eps[1][0], local_input[1][1] - eps[1][0], local_input[1][2] - eps[1][0], local_input[1][3] - eps[1][0],\
                local_input[1][4] - eps[1][0], local_input[1][5] - eps[1][0], local_input[1][6] - eps[1][0],\
                local_input[1][7] - eps[1][0], local_input[1][8] - eps[1][0], local_input[1][9] - eps[1][0],\
                local_input[1][10] - eps[1][1], local_input[1][11] - eps[1][2], local_input[1][12] - eps[1][3],\
                local_input[1][13] - eps[1][4]],  # Local1: all the sensors' local input values minus the local tolerance

            [laser_bounds[0], laser_bounds[0], laser_bounds[0], laser_bounds[0], laser_bounds[0], laser_bounds[0], laser_bounds[0],\
                laser_bounds[0], laser_bounds[0], laser_bounds[0], heading_bounds[0], distance_bounds[0],\
                    lin_vel_bounds[0], ang_vel_bounds[0]],  # GlobalPartial: all the sensors' lower bounds

            [laser_bounds[0], laser_bounds[0], laser_bounds[0], laser_bounds[0], laser_bounds[0], laser_bounds[0], laser_bounds[0],\
                laser_bounds[0], laser_bounds[0], laser_bounds[0], heading_bounds[0], distance_bounds[0],\
                    lin_vel_bounds[1] - eps[3][1], -eps[3][2]]  # SpeedThreshold: all the sensors' lower bounds, except
                                                                # for the last 2: linear and angular velocities
           ]
input_ub = [
            [laser_bounds[1], laser_bounds[1], laser_bounds[1], laser_bounds[1], laser_bounds[1], laser_bounds[1], laser_bounds[1],\
                laser_bounds[1], laser_bounds[1], laser_bounds[1], heading_bounds[1], distance_bounds[1],\
                    lin_vel_bounds[1], ang_vel_bounds[1]],  # GlobalReach: all the sensors' upper bounds

            [local_input[1][0] + eps[1][0], local_input[1][1] + eps[1][0], local_input[1][2] + eps[1][0], local_input[1][3] + eps[1][0],\
                local_input[1][4] + eps[1][0], local_input[1][5] + eps[1][0], local_input[1][6] + eps[1][0],\
                local_input[1][7] + eps[1][0], local_input[1][8] + eps[1][0], local_input[1][9] + eps[1][0],\
                local_input[1][10] + eps[1][1], local_input[1][11] + eps[1][2], local_input[1][12] + eps[1][3],\
                local_input[1][13] + eps[1][4]],  # Local1: all the sensors' local input values plus the local tolerance

            [laser_bounds[1], laser_bounds[1], laser_bounds[1], laser_bounds[1], laser_bounds[0] + eps[2][0], laser_bounds[1], laser_bounds[1],\
                laser_bounds[1], laser_bounds[1], laser_bounds[1], heading_bounds[1], distance_bounds[1],\
                    lin_vel_bounds[1], ang_vel_bounds[1]],  # GlobalPartial: all the sensors' upper bounds, except for the 5th laser scan

            [laser_bounds[1], laser_bounds[1], laser_bounds[1], laser_bounds[1], laser_bounds[0] + eps[2][0], laser_bounds[1], laser_bounds[1],\
                laser_bounds[1], laser_bounds[1], laser_bounds[1], heading_bounds[1], distance_bounds[1],\
                    lin_vel_bounds[1], eps[3][2]]  # SpeedThreshold: all the sensors' upper bounds, except
                                                   # for the last 1: angular velocity
           ]


for net in range(len(net_id)): # Loop for each neural network

    # Computing local output values (for the local properties only)
    local_output = []

    agent, agent_pt, agent_onnx = pnv_converter(net_id[net], state_dim, hidden_dim, action_dim, device)  # Get PyNeVer-compatible nets
    agent_pt.pytorch_network.eval()  # Set the network in testing mode (its parameters are freezed)
    agent_pt.pytorch_network.to(device)

    for property in range(len(eps)):

        if len(local_input[property]) != state_dim:  # Skip global reachability verification properties

            local_output.append([])

        else:  # Obtain the output local values for local verification properties

            state = local_input[property]

            state = np.asarray(state)  # SAC algorithm action pre-processing operation (1/3)
            state = np.float32(state)  # SAC algorithm action pre-processing operation (2/3)
            state = torch.FloatTensor(state).to(device).unsqueeze(0)  # SAC algorithm action pre-processing operation (3/3)

            action = agent_pt.pytorch_network.forward(state.double())

            action = action.detach().cpu().numpy()[0]  # SAC algorithm action post-processing operation

            # Normalization was performed between [-1, 1] by the Sigmoid layer
            unnorm_action = np.array([lin_vel_bounds[0] + (action[0] + 1.0) * 0.5 * (lin_vel_bounds[1] - lin_vel_bounds[0]),
                                      ang_vel_bounds[0] + (action[1] + 1.0) * 0.5 * (ang_vel_bounds[1] - ang_vel_bounds[0])])

            local_output.append(unnorm_action)

    # Creation of output unsafe matrixes and vectors
    unsafe_mats = [
                    [[1, 0], [-1, 0], [0, 1], [0, -1]],  # GlobalReach: 2 conditions for each output
                    [[1, 0], [-1, 0], [0, 1], [0, -1]],  # Local1: 2 conditions for each output
                    [[1, 0], [-1, 0], [0, 1], [0, -1]],  # GlobalPartial: 2 conditions for each output
                    [[1, 0], [-1, 0], [0, 1], [0, -1]]  # SpeedThreshold: 2 conditions for each output
                ]
    unsafe_vecs = [
                    [[lin_vel_bounds[0]], [-lin_vel_bounds[1]], [ang_vel_bounds[0]], [-ang_vel_bounds[1]]],  # GlobalReach: outputs' lower and upper bounds
                    [[local_output[1][0] - delta[1][0]], [-local_output[1][0] - delta[1][0]],\
                        [local_output[1][1] - delta[1][1]], [-local_output[1][1] - delta[1][1]]],  #Local1: all the outputs' local lower and upper bounds
                    [[lin_vel_bounds[1]], [-lin_vel_bounds[0] - delta[2][0]],\
                        [ang_vel_bounds[1] - delta[2][1]], [-ang_vel_bounds[0] - delta[2][1]]],  #GlobalPartial: linear velocity is higher than chosen threshold,
                                                                                                # (absolute) angular velocity is lower than chosen threshold
                    [[lin_vel_bounds[1] - delta[3][0]], [lin_vel_bounds[0]], [delta[3][1]], [delta[3][1]]]  # SpeedThreshold: linear velocity is higher than chosen threshold,
                                                                                                            # (absolute) angular velocity is lower than chosen threshold
                ]

    # Verification loop
    for prop_idx in range(len(property_ids)):  # Loop for each property

        # Matrixes creation
        in_pred_mat = []
        in_pred_bias = []
        for k in range(len(input_lb[prop_idx])):
            lb_constraint = np.zeros(len(input_lb[prop_idx]))
            ub_constraint = np.zeros(len(input_ub[prop_idx]))
            lb_constraint[k] = -1
            ub_constraint[k] = 1
            in_pred_mat.append(lb_constraint)
            in_pred_mat.append(ub_constraint)
            in_pred_bias.append([-input_lb[prop_idx][k]])
            in_pred_bias.append([input_ub[prop_idx][k]])

        # Convert matrixes and vectors into NumPy arrays
        in_pred_mat = np.array(in_pred_mat)
        in_pred_bias = np.array(in_pred_bias)

        # Creation of the matrixes defining the negation of the wanted property (i.e., unsafe region)
        # (i.e., out_pred_mat * y <= out_pred_bias)
        out_pred_mat = []
        out_pred_bias = []
        out_pred_mat.append(np.array(unsafe_mats[prop_idx], dtype=float))
        out_pred_bias.append(np.array(unsafe_vecs[prop_idx], dtype=float))

        # Code block useful for debugging purposes
        #print("\nin_pred_mat:\n" + str(in_pred_mat) + "\n")
        #print("\nin_pred_bias:\n" + str(in_pred_bias) + "\n")
        #print("\nout_pred_mat:\n" + str(out_pred_mat) + "\n")
        #print("\nout_pred_bias:\n" + str(out_pred_bias) + "\n")

        prop = ver.NeVerProperty(in_pred_mat, in_pred_bias, out_pred_mat, out_pred_bias)  # Create the property to be verified

        for verParam_idx in range(len(ver_params)):  # Loop for each verification parameter set

            # Stream and log the names of the currently active network, property, and parameter set
            logger_nav_stream.info(
                f"Verifying nav_sac_net={net_id[net]}_PROP={property_ids[prop_idx]}_PARAMS={ver_params[verParam_idx][0]}\n")
            #logger_empty.debug(f"\nnav_sac_net={net_id[net]}_PROP={property_ids[prop_idx]}_PARAMS={ver_params[verParam_idx][0]}\n")
            #logger_lp.debug(f"\nnav_sac_net={net_id[net]}_PROP={property_ids[prop_idx]}_PARAMS={ver_params[verParam_idx][0]}\n")
            #logger_lb.debug(f"\nnav_sac_net={net_id[net]}_PROP={property_ids[prop_idx]}_PARAMS={ver_params[verParam_idx][0]}\n")
            #logger_ub.debug(f"\nnav_sac_net={net_id[net]}_PROP={property_ids[prop_idx]}_PARAMS={ver_params[verParam_idx][0]}\n")

            # Create the verifier
            verifier = ver.NeverVerification(ver_params[verParam_idx][1][0], ver_params[verParam_idx][1][1], ver_params[verParam_idx][2])

            # Check if the property is verified and measure time elapsed
            verPar_time_start = time.perf_counter()
            safe = verifier.verify(agent, prop)
            verPar_time_end = time.perf_counter()

            # Log verification results in the verification log file
            logger_nav_file.info(f"nav,{net_id[net]},{property_ids[prop_idx]},{ver_params[verParam_idx][0]}"
                                 f",{safe},{verPar_time_end - verPar_time_start}")

# Store and format the end time of the whole process
end_time = datetime.now()
f_end_time = end_time.strftime("%H:%M:%S")
# Print verification script's start and end time
logger_nav_stream.info(f"Verification script start time: {f_start_time}")
logger_nav_stream.info(f"Verification script end time: {f_end_time}\n")
