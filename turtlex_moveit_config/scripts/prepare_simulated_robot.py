#!/usr/bin/env python3

# Copyright (c) 2015, Fetch Robotics Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Fetch Robotics Inc. nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL FETCH ROBOTICS INC. BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Author: Michael Ferguson

import sys

import rospy
import actionlib
from control_msgs.msg import (FollowJointTrajectoryAction,
                              FollowJointTrajectoryGoal,
                              GripperCommandAction,
                              GripperCommandGoal)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

arm_joint_names = ["joint_1, joint_2, joint_3, joint_4, joint_5"]
arm_intermediate_positions  = [0.0, 0.9575, -0.2510, 0.8364, 0.0] # picking_arm
arm_joint_positions  = [0.0, -1.0, 1.0, 1.2, 0.0] # rest_arm

if __name__ == "__main__":

    rospy.init_node("prepare_simulated_robot")

    rospy.loginfo("Waiting for arm_controller...")
    arm_client = actionlib.SimpleActionClient("arm_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
    arm_client.wait_for_server()
    rospy.loginfo("...connected.")

    rospy.loginfo("Waiting for gripper_controller...")
    gripper_client = actionlib.SimpleActionClient("gripper_controller/gripper_action", GripperCommandAction)
    gripper_client.wait_for_server()
    rospy.loginfo("...connected.")

    trajectory = JointTrajectory()
    trajectory.joint_names = arm_joint_names
    trajectory.points.append(JointTrajectoryPoint())
    trajectory.points[0].positions = [0.0] * len(arm_joint_positions)
    trajectory.points[0].velocities =  [0.0] * len(arm_joint_positions)
    trajectory.points[0].accelerations = [0.0] * len(arm_joint_positions)
    trajectory.points[0].time_from_start = rospy.Duration(1.0)
    trajectory.points.append(JointTrajectoryPoint())
    trajectory.points[1].positions = arm_intermediate_positions
    trajectory.points[1].velocities =  [0.0] * len(arm_joint_positions)
    trajectory.points[1].accelerations = [0.0] * len(arm_joint_positions)
    trajectory.points[1].time_from_start = rospy.Duration(4.0)
    trajectory.points.append(JointTrajectoryPoint())
    trajectory.points[2].positions = arm_joint_positions
    trajectory.points[2].velocities =  [0.0] * len(arm_joint_positions)
    trajectory.points[2].accelerations = [0.0] * len(arm_joint_positions)
    trajectory.points[2].time_from_start = rospy.Duration(7.5)

    arm_goal = FollowJointTrajectoryGoal()
    arm_goal.trajectory = trajectory
    arm_goal.goal_time_tolerance = rospy.Duration(0.0)

    gripper_goal = GripperCommandGoal()
    gripper_goal.command.max_effort = 10.0
    gripper_goal.command.position = 0.108

    rospy.loginfo("Setting positions...")
    arm_client.send_goal(arm_goal)
    gripper_client.send_goal(gripper_goal)
    gripper_client.wait_for_result(rospy.Duration(5.0))
    arm_client.wait_for_result(rospy.Duration(6.0))
    rospy.loginfo("...done")