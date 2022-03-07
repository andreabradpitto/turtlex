#!/usr/bin/env python3

import rospy
import numpy as np

import sys
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

from utils import tcolors

# TODO la prima classe è la stessa di quella in fetch_env. La seconda è inclusa al 100% nell'env di fetch_env.
# Sotto ci sono 4 test da far provare a turno e il main per farli girare.

class MoveFetch(object):
    
    def __init__(self):
        rospy.logdebug("In Move Fetch Class init...")
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.logdebug("moveit_commander initialised...")
        
        rospy.logdebug("Starting Robot Commander...")
        self.robot = moveit_commander.RobotCommander()
        rospy.logdebug("Starting Robot Commander...DONE")
        
        self.scene = moveit_commander.PlanningSceneInterface()  
        rospy.logdebug("PlanningSceneInterface initialised...DONE")
        self.group = moveit_commander.MoveGroupCommander("arm")
        rospy.logdebug("MoveGroupCommander for arm initialised...DONE")

        
    def ee_traj(self, pose):
        
        self.group.set_pose_target(pose)
        
        result = self.execute_trajectory()
        
        return result
        
    def joint_traj(self, positions_array):
        
        self.group_variable_values = self.group.get_current_joint_values()
        rospy.logdebug("Group Vars:")
        rospy.logdebug(self.group_variable_values)
        rospy.logdebug("Point:")
        rospy.logdebug(positions_array)
        self.group_variable_values[0] = positions_array[0]
        self.group_variable_values[1] = positions_array[1]
        self.group_variable_values[2] = positions_array[2]
        self.group_variable_values[3] = positions_array[3]
        self.group_variable_values[4] = positions_array[4]
        self.group_variable_values[5] = positions_array[5]
        self.group_variable_values[6] = positions_array[6]
        self.group.set_joint_value_target(self.group_variable_values)
        result =  self.execute_trajectory()
        
        return result
        
    def execute_trajectory(self):
        
        self.plan = self.group.plan()
        result = self.group.go(wait=True)
        
        return result

    def ee_pose(self):
        
        gripper_pose = self.group.get_current_pose()
        rospy.loginfo(tcolors.CYAN + "EE POSE ==> " + str(gripper_pose) + tcolors.ENDC)

        return gripper_pose
        
    def ee_rpy(self, request):
        
        gripper_rpy = self.group.get_current_rpy()

        return gripper_rpy


class FetchMoveitClass(object):
    
    def __init__(self):
        self.move_fetch_object = MoveFetch()
        
        self.joint_names = ["joint0",
                            "joint1",
                            "joint2",
                            "joint3",
                            "joint4",
                            "joint5",
                            "joint6"]
    def get_joint_names(self):
        return self.joint_names
        
    def set_trajectory_ee(self, action):
        """
        Helper function.
        Wraps an action vector of joint angles into a JointTrajectory message.
        The velocities, accelerations, and effort do not control the arm motion
        """
        # Set up a trajectory message to publish.
        ee_target = geometry_msgs.msg.Pose()
        ee_target.orientation.w = 1.0
        ee_target.position.x = action[0]
        ee_target.position.y = action[1]
        ee_target.position.z = action[2]
        
        result = self.move_fetch_object.ee_traj(ee_target)
        
        return result
        
    def set_trajectory_joints(self, initial_qpos):

        positions_array = [None] * 7
        positions_array[0] = initial_qpos["joint0"]
        positions_array[1] = initial_qpos["joint1"]
        positions_array[2] = initial_qpos["joint2"]
        positions_array[3] = initial_qpos["joint3"]
        positions_array[4] = initial_qpos["joint4"]
        positions_array[5] = initial_qpos["joint5"]
        positions_array[6] = initial_qpos["joint6"]
 
        self.move_fetch_object.joint_traj(positions_array)
        
        return True
        
    def create_action(self,position,orientation):
        """
        position = [x,y,z]
        orientation= [x,y,z,w]
        """
        
        gripper_target = np.array(position)
        gripper_rotation = np.array(orientation)
        action = np.concatenate([gripper_target, gripper_rotation])
        
        return action
        
    def create_joints_dict(self,joints_positions):
        """
        Based on the Order of the positions, they will be assigned to its joint name
        names_in_order:
          joint0: 0.0
          joint1: 0.0
          joint2: 0.0
          joint3: -1.5
          joint4: 0.0
          joint5: 1.5
          joint6: 0.0
        """
        
        assert len(joints_positions) == len(self.joint_names), "Wrong number of joints, there should be " + str(len(self.joint_names))
        joints_dict = dict(zip(self.joint_names,joints_positions))
        
        return joints_dict
        
    def get_ee_pose(self):
            
        return self.move_fetch_object.ee_pose()
        

def MoveArmTest1():
    
    fetchmove = FetchMoveitClass()
    
    position1 = [0.498, 0.005, 0.6]
    orientation1 = [1., 0., 1., 0.]
    action1 = fetchmove.create_action(position1, orientation1)
    
    setup_ee_pos = rospy.get_param('/fetch/setup_ee_pos')
    rospy.logdebug(setup_ee_pos)
    rospy.logdebug(type(setup_ee_pos))
    position2 = [setup_ee_pos["x"], setup_ee_pos["y"], setup_ee_pos["z"]]
    #position2 = [0.598, 0.005, 0.9]
    orientation2 = [1., 0., 1., 0.]
    action2 = fetchmove.create_action(position2, orientation2)
    
    position3 = [0.398, 0.005, 0.6]
    orientation3 = [1., 0., 1., 0.]
    action3 = fetchmove.create_action(position3, orientation3)
    
    position1_bad = [0.498, 0.005, 0.9]
    orientation1_bad = [1., 0., 1., 0.]
    action1_bad = fetchmove.create_action(position1_bad, orientation1_bad)
    
    result_list = []
    
    rospy.loginfo(tcolors.CYAN + "Executing Action1" + " ({})".format(position1) + tcolors.ENDC)
    result_list.append(fetchmove.set_trajectory_ee(action1))
    rospy.loginfo(tcolors.CYAN + str(type(fetchmove.get_ee_pose())) + tcolors.ENDC)
    rospy.loginfo(tcolors.CYAN + str(position1) + " <==VS==> " + str(fetchmove.get_ee_pose()) + tcolors.ENDC)
    
    
    rospy.loginfo(tcolors.CYAN + "Executing Action 2 (SETUP EE _POS)" + " ({})".format(position2) + tcolors.ENDC)
    result_list.append(fetchmove.set_trajectory_ee(action2))

    rospy.loginfo(tcolors.CYAN + "Executing Action 3" + " ({})".format(position3) + tcolors.ENDC)
    result_list.append(fetchmove.set_trajectory_ee(action3))

    rospy.loginfo(tcolors.CYAN + "Executing Action 1 BAD" + " ({})".format(position1_bad) + tcolors.ENDC)
    result_list.append(fetchmove.set_trajectory_ee(action1_bad))
    
    rospy.loginfo(tcolors.MAGENTA + "End MoveArmTest1, results==" + str(result_list) + tcolors.ENDC)
    
    
def MoveArmTest2():
    
    fetchmove = FetchMoveitClass()
    
    result_list = []
    
    number_of_joints = len(fetchmove.get_joint_names())
    
    init_pos = rospy.get_param('/fetch/init_pos')
    joints_positions = [init_pos['joint0'], init_pos['joint1'], init_pos['joint2'], init_pos['joint3'],
                        init_pos['joint4'], init_pos['joint5'], init_pos['joint6']]
    #joints_positions = [0] * number_of_joints
    angle_test_value = 0.7
    
    for i in range(number_of_joints):
        joints_positions = [init_pos['joint0'], init_pos['joint1'], init_pos['joint2'], init_pos['joint3'],
                            init_pos['joint4'], init_pos['joint5'], init_pos['joint6']] # TODO qui sotto andava cambiato?!
        #joints_positions = [0] * number_of_joints
        joints_positions[i] = angle_test_value
        joint_pos_dict = fetchmove.create_joints_dict(joints_positions)
        
        rospy.loginfo(tcolors.CYAN + "Executing JointConfig (INIT POS) " + str(i) + " ({})".format(joints_positions[i]) + tcolors.ENDC) # TODO vedi se va l'indexing così. perche prima lo ho indicizzato come dictionary
        result_list.append(fetchmove.set_trajectory_joints(joint_pos_dict))

    rospy.loginfo(tcolors.MAGENTA + "End MoveArmTest2, results==" + str(result_list) + tcolors.ENDC)
    

def MoveArmTest_OnePosition():
    
    fetchmove = FetchMoveitClass()
    
    position1 = [0.8, 0.0, 1.1] # TODO questo è uguale a quello sotto per ora. Prova altri valori
    orientation1 = [1., 0., 1., 0.]
    action1 = fetchmove.create_action(position1, orientation1)
    
    result_list = []
    
    rospy.loginfo(tcolors.CYAN + "Executing Action1" + " ({})".format(position1) + tcolors.ENDC)
    result_list.append(fetchmove.set_trajectory_ee(action1))
    rospy.loginfo(tcolors.CYAN + str(type(fetchmove.get_ee_pose())) + tcolors.ENDC)
    rospy.loginfo(tcolors.CYAN + str(position1) + " <==VS==> " + str(fetchmove.get_ee_pose()) + tcolors.ENDC)
    
    rospy.loginfo(tcolors.MAGENTA + "End MoveArmTest_OnePosition, results==" + str(result_list) + tcolors.ENDC)
    
    
def MoveArmTest_FetchEEPose():
    
    fetchmove = FetchMoveitClass()

    goal_ee_pos = rospy.get_param('/fetch/goal_ee_pos')
    position1 = [goal_ee_pos["x"], goal_ee_pos["y"], goal_ee_pos["z"]]
    #position1 = [0.8, 0.0, 1.1]
    orientation1 = [1., 0., 1., 0.]
    action1 = fetchmove.create_action(position1, orientation1)
    
    result_list = []
    
    rospy.loginfo(tcolors.CYAN + "Executing Action1 (GOAL EE POS)" + " ({})".format(position1) + tcolors.ENDC)
    result_list.append(fetchmove.set_trajectory_ee(action1))
    rospy.loginfo(tcolors.CYAN + str(type(fetchmove.get_ee_pose())) + tcolors.ENDC)
    rospy.loginfo(tcolors.CYAN + str(position1) + " <==VS==> " + str(fetchmove.get_ee_pose()) + tcolors.ENDC)
    
    rospy.loginfo(tcolors.MAGENTA + "End MoveArmTest_FetchEEPose, results==" + str(result_list) + tcolors.ENDC)

        
if __name__ == '__main__':
    
    rospy.init_node('fetch_moveit_test', anonymous=True, log_level=rospy.DEBUG)
    
    rospy.loginfo(tcolors.MAGENTA + "--- Starting MoveArmTest1() ---" + tcolors.ENDC)
    MoveArmTest1()

    rospy.loginfo(tcolors.MAGENTA + "--- Starting MoveArmTest2() ---" + tcolors.ENDC)
    MoveArmTest2()

    #rospy.loginfo(tcolors.MAGENTA + "--- Starting MoveArmTest_OnePosition() ---" + tcolors.ENDC)
    #MoveArmTest_OnePosition()

    rospy.loginfo(tcolors.MAGENTA + "--- Starting MoveArmTest_FetchEEPose() ---" + tcolors.ENDC)
    MoveArmTest_FetchEEPose()
    