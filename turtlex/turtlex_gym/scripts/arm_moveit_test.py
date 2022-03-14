#!/usr/bin/env python3

import rospy
import numpy as np

import sys
import moveit_commander
#import moveit_msgs.msg
import geometry_msgs.msg
import time
#from std_srvs.srv import Empty

from utils import tcolors

# TODO this code is untested and not properly documented for the current version of the repository

# TODO la prima classe è la stessa di quella in fetch_env. La seconda è inclusa al 100% nell'env di fetch_env.
# Sotto ci sono 5 test da far provare a turno (o anche insieme) e il main per farli girare.

class MoveTurtlexArm(object):
    
    def __init__(self):
        rospy.logdebug("In MoveTurtlexArm Class init...")
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.logdebug("moveit_commander initialised...")
        
        rospy.logdebug("Starting Robot Commander...")
        self.robot = moveit_commander.RobotCommander()
        rospy.logdebug("Starting Robot Commander...DONE")
        
        self.scene = moveit_commander.PlanningSceneInterface()  
        rospy.logdebug("PlanningSceneInterface initialised...DONE")
        self.group = moveit_commander.MoveGroupCommander("arm", wait_for_servers=30.0)
        rospy.logdebug("MoveGroupCommander for arm initialised...DONE")

        self.ee_group = moveit_commander.MoveGroupCommander("gripper", wait_for_servers=30.0)
        rospy.logdebug("MoveGroupCommander for gripper initialised...DONE")
        
    def ee_traj(self, pose):
        
        self.group.set_pose_target(pose) # TODO self.group.set_named_target("NOME") per usare le configurazioni mie
        
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
        self.group.set_joint_value_target(self.group_variable_values)
        result =  self.execute_trajectory()
        
        return result
        
    def execute_trajectory(self):
        
        #self.plan = self.group.plan()
        #result = self.group.go(wait=True)

        # TODO questo è il modo per verificare con moveit che il plan ci sia. Posso usarlo per il mio environment con le reti neurali
        plan = self.group.plan()
        #rospy.loginfo("\n\tRISULTATO PLANNING: " + str(plan) + "\n")
        if plan[0] == True:  # va bene anche, credo (ancor da testare): "if plan[1].points:"" ; o al massimo "if plan[1].joint_trajectory.points:"
            rospy.loginfo("Plan found")
            result = self.group.execute(plan[1])
        else:
            result = False
            rospy.logerr("Trajectory is empty. Planning was unsuccessful.")
        
        return result

    def ee_pose(self):
        
        gripper_pose = self.group.get_current_pose()
        #return gripper_pose.pose
        rospy.loginfo(tcolors.CYAN + "EE POSE ==> " + str(gripper_pose) + tcolors.ENDC)

        return gripper_pose
        
    def ee_rpy(self, request):
        
        gripper_rpy = self.group.get_current_rpy()
        #roll = gripper_rpy[0]
        #pitch = gripper_rpy[1]
        #yaw = gripper_rpy[2]
        #return [roll,pitch,yaw]

        return gripper_rpy


class TurtlexArmMoveitClass(object):
    
    def __init__(self):
        self.move_turtlex_arm_object = MoveTurtlexArm()
        
        self.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]

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
        
        result = self.move_turtlex_arm_object.ee_traj(ee_target)
        
        return result
        
    def set_trajectory_joints(self, initial_qpos):

        positions_array = [None] * 5
        positions_array[0] = initial_qpos["joint_1"]
        positions_array[1] = initial_qpos["joint_2"]
        positions_array[2] = initial_qpos["joint_3"]
        positions_array[3] = initial_qpos["joint_4"]
        positions_array[4] = initial_qpos["joint_5"]
 
        self.move_turtlex_arm_object.joint_traj(positions_array)
        
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
          joint_1: 0.0
          joint_2: 0.0
          joint_3: -1.5
          joint_4: 0.0
          joint_5: 1.5
        """
        
        assert len(joints_positions) == len(self.joint_names), "Wrong number of joints, there should be " + str(len(self.joint_names))
        joints_dict = dict(zip(self.joint_names, joints_positions))
        
        return joints_dict
        
    def get_ee_pose(self):
            
        return self.move_turtlex_arm_object.ee_pose()
        

def MoveArmTest1():
    
    turtlex_arm_move = TurtlexArmMoveitClass()
    
    position1 = [0.114, 0.194, 0.237] # questa dovrebbe essere ok per il turtlex_arm: vista da gazebo mettendo un oggetto e vedendo la sua pose
    orientation1 = [1., 0., 1., 0.]
    action1 = turtlex_arm_move.create_action(position1, orientation1)
    
    setup_ee_pos = rospy.get_param('/turtlex_arm/setup_ee_pos')
    rospy.logdebug(setup_ee_pos)
    rospy.logdebug(type(setup_ee_pos))
    position2 = [setup_ee_pos["x"], setup_ee_pos["y"], setup_ee_pos["z"]]
    #position2 = [0.598, 0.005, 0.9]
    orientation2 = [1., 0., 1., 0.]
    action2 = turtlex_arm_move.create_action(position2, orientation2)
    
    position3 = [-0.199, 0.272, 0.405] # questa dovrebbe essere ok per il turtlex_arm: presa da pose del gripper_rail di <random valid> su RViz
    orientation3 = [-0.591, -0.520, -0.034, 0.616]
    action3 = turtlex_arm_move.create_action(position3, orientation3)
    
    position1_bad = [0.498, 0.005, 0.9]
    orientation1_bad = [1., 0., 1., 0.]
    action1_bad = turtlex_arm_move.create_action(position1_bad, orientation1_bad)
    
    result_list = []
    
    rospy.loginfo(tcolors.CYAN + "Executing Action1" + " ({})".format(position1) + tcolors.ENDC)
    result_list.append(turtlex_arm_move.set_trajectory_ee(action1))
    rospy.loginfo(tcolors.CYAN + str(type(turtlex_arm_move.get_ee_pose())) + tcolors.ENDC)
    rospy.loginfo(tcolors.CYAN + str(position1) + " <==VS==> " + str(turtlex_arm_move.get_ee_pose()) + tcolors.ENDC)
    
    
    rospy.loginfo(tcolors.CYAN + "Executing Action 2 (SETUP EE _POS)" + " ({})".format(position2) + tcolors.ENDC)
    result_list.append(turtlex_arm_move.set_trajectory_ee(action2))

    rospy.loginfo(tcolors.CYAN + "Executing Action 3" + " ({})".format(position3) + tcolors.ENDC)
    result_list.append(turtlex_arm_move.set_trajectory_ee(action3))

    rospy.loginfo(tcolors.CYAN + "Executing Action 1 BAD" + " ({})".format(position1_bad) + tcolors.ENDC)
    result_list.append(turtlex_arm_move.set_trajectory_ee(action1_bad))
    
    rospy.loginfo(tcolors.MAGENTA + "End MoveArmTest1, results==" + str(result_list) + tcolors.ENDC)
    
    
def MoveArmTest2():
    
    turtlex_arm_move = TurtlexArmMoveitClass()

    result_list = []
    
    number_of_joints = len(turtlex_arm_move.get_joint_names())

    # TODO test aggiunta oggetto per vedere le coordinate del gripper... (non sembra andare, prova a cambiare frame_id)
    #box_pose = geometry_msgs.msg.PoseStamped()
    #box_pose.header.frame_id = "gripper_prismatic_joint_1"
    #box_pose.pose.orientation.w = 1.0
    #box_pose.pose.position.z = 0.11  # above the chosen frame
    #box_name = "box"
    #turtlex_arm_move.move_turtlex_arm_object.scene.add_box(box_name, box_pose, size=(0.075, 0.075, 0.075))
    
    init_joint_pos = rospy.get_param('/turtlex_arm/init_joint_pos')
    joints_positions = [init_joint_pos['joint_1'], init_joint_pos['joint_2'], init_joint_pos['joint_3'],
                        init_joint_pos['joint_4'], init_joint_pos['joint_5']]
    #joints_positions = [0] * number_of_joints
    angle_test_value = 0.7
    
    for i in range(number_of_joints):
        joints_positions = [init_joint_pos['joint_1'], init_joint_pos['joint_2'], init_joint_pos['joint_3'],
                            init_joint_pos['joint_4'], init_joint_pos['joint_5']] # TODO qui sotto andava cambiato?!
        #joints_positions = [0] * number_of_joints
        joints_positions[i] = angle_test_value
        joint_pos_dict = turtlex_arm_move.create_joints_dict(joints_positions)
        
        rospy.loginfo(tcolors.CYAN + "Executing JointConfig (INIT POS) " + str(i) + " ({})".format(joints_positions[i]) + tcolors.ENDC) # TODO vedi se va l'indexing così. perche prima lo ho indicizzato come dictionary
        result_list.append(turtlex_arm_move.set_trajectory_joints(joint_pos_dict))

    rospy.loginfo(tcolors.MAGENTA + "End MoveArmTest2, results==" + str(result_list) + tcolors.ENDC)


def MoveArmPicking():

    turtlex_arm_move = TurtlexArmMoveitClass()

    result_list = []

    number_of_joints = len(turtlex_arm_move.get_joint_names())

    # TODO turtlex_arm_move.move_turtlex_arm_object.group.set_goal_tolerance(0.01) # non sembra andare: https://answers.ros.org/question/391920

    print("\n" + str(turtlex_arm_move.move_turtlex_arm_object.group.get_end_effector_link()) + "\n") # = wrist_2_link
    
    #print("\nposizione rest\n" + str(turtlex_arm_move.move_turtlex_arm_object.group.get_current_pose()) + "\n")
    print("\nposizione rest\n" + str(turtlex_arm_move.move_turtlex_arm_object.group.get_current_pose().pose) + "\n")
    #print("\nposizione rest\n" + str(turtlex_arm_move.move_turtlex_arm_object.group.get_current_pose().pose.position) + "\n")
    init_joint_pos = rospy.get_param('/turtlex_arm/init_joint_pos')
    joints_positions = [init_joint_pos['joint_1'], init_joint_pos['joint_2'], init_joint_pos['joint_3'],
                        init_joint_pos['joint_4'], init_joint_pos['joint_5']]
    picking_joints_positions = [0.3, 0.9575, -0.251, 0.8364, 0.4]
    
    for i in range(number_of_joints):
        joints_positions[i] = picking_joints_positions[i]
        joint_pos_dict = turtlex_arm_move.create_joints_dict(joints_positions)
        
        rospy.loginfo(tcolors.CYAN + "Executing JointConfig (PICKING POS) " + str(i) + " ({})".format(joints_positions[i]) + tcolors.ENDC) # TODO vedi se va l'indexing così. perche prima lo ho indicizzato come dictionary
        result_list.append(turtlex_arm_move.set_trajectory_joints(joint_pos_dict))

    # TODO
    #turtlex_arm_move.move_turtlex_arm_object.group.set_named_target("rest_arm")
    #turtlex_arm_move.move_turtlex_arm_object.execute_trajectory()

    print("\nposizione pick di lato\n" + str(turtlex_arm_move.move_turtlex_arm_object.group.get_current_pose().pose) + "\n")

    turtlex_arm_move.move_turtlex_arm_object.ee_group.set_named_target("closed_gripper")
    turtlex_arm_move.move_turtlex_arm_object.ee_group.plan()
    turtlex_arm_move.move_turtlex_arm_object.ee_group.go(wait=True)

    print("\nposizione pick di lato - gripper aperto\n" + str(turtlex_arm_move.move_turtlex_arm_object.group.get_current_pose().pose) + "\n")

    rospy.loginfo(tcolors.MAGENTA + "End MoveArmPicking, results==" + str(result_list) + tcolors.ENDC)


def MoveArmTest_OnePosition():
    
    turtlex_arm_move = TurtlexArmMoveitClass()

    # TODO
    #turtlex_arm_move.move_turtlex_arm_object.group.set_planner_id("RRTConnect")
    #turtlex_arm_move.move_turtlex_arm_object.group.set_planning_time(15)

    # TODO Blocco per ottenere la posizione corrente dell'EE (È in riferimento a "base_footprint")
    """
    print("\n" + str(turtlex_arm_move.move_turtlex_arm_object.group.get_current_pose().pose) + "\n")
    
    risultato ottenuto:
    position:
    x: 0.08978966983946651
    y: 7.688869716872818e-05
    z: 0.30775837351632673
    orientation: 
    x: -0.00023758565085101746
    y: 0.9837234120687101
    z: 0.000145475901347114
    w: 0.17968909521090998
    """

    # TODO Questo per settare le goal tolerances
    """
    turtlex_arm_move.move_turtlex_arm_object.group.set_goal_position_tolerance(0.01)
    turtlex_arm_move.move_turtlex_arm_object.group.set_goal_orientation_tolerance(0.01)
    """    

    #position1 = [-0.199, 0.272, 0.405] # questa dovrebbe essere ok per il turtlex_arm: presa da pose del gripper_rail di <random valid> su RViz
    #orientation1 = [-0.591, -0.520, -0.034, 0.616]
    #position1 = [0.106, 0.011, 0.278] # questo preso da topic /gazebo/links_states (braccio in poszione di riposo)
    #orientation1 = [0.0, 0.983, 0.0, 0.183]
    #position1 = [0.08978966983946651, 7.688869716872818e-05, 0.80775837351632673] # questo basato sul codice get_current_pose().pose qui sopra (cambiata la z)
    #orientation1 = [-0.00023758565085101746, 0.9837234120687101, 0.000145475901347114, 0.17968909521090998]
    position1 = [0.2583668082644751, 0.07449562807698895, 0.14275216737937577] # questa basata su un posto raggiunto ("pick di lato") e poi letto con le api moveit
    orientation1 = [0.050066861120315095, 0.9986950337988192, 0.00344613205439315, 0.009469057872781227] 
    action1 = turtlex_arm_move.create_action(position1, orientation1)
    
    result_list = []
    
    rospy.loginfo(tcolors.CYAN + "Executing Action1" + " ({})".format(position1) + tcolors.ENDC)
    result_list.append(turtlex_arm_move.set_trajectory_ee(action1))
    rospy.loginfo(tcolors.CYAN + str(type(turtlex_arm_move.get_ee_pose())) + tcolors.ENDC)
    rospy.loginfo(tcolors.CYAN + str(position1) + " <==VS==> " + str(turtlex_arm_move.get_ee_pose()) + tcolors.ENDC)
    
    rospy.loginfo(tcolors.MAGENTA + "End MoveArmTest_OnePosition, results==" + str(result_list) + tcolors.ENDC)
    
    
def MoveArmTest_TurtlexArmEEPose():
    
    turtlex_arm_move = TurtlexArmMoveitClass()

    goal_ee_pos = rospy.get_param('/turtlex_arm/goal_ee_pos')
    position1 = [goal_ee_pos["x"], goal_ee_pos["y"], goal_ee_pos["z"]]
    #position1 = [0.8, 0.0, 1.1]
    orientation1 = [1., 0., 1., 0.]
    action1 = turtlex_arm_move.create_action(position1, orientation1)
    
    result_list = []
    
    rospy.loginfo(tcolors.CYAN + "Executing Action1 (GOAL EE POS)" + " ({})".format(position1) + tcolors.ENDC)
    result_list.append(turtlex_arm_move.set_trajectory_ee(action1))
    rospy.loginfo(tcolors.CYAN + str(type(turtlex_arm_move.get_ee_pose())) + tcolors.ENDC)
    rospy.loginfo(tcolors.CYAN + str(position1) + " <==VS==> " + str(turtlex_arm_move.get_ee_pose()) + tcolors.ENDC)
    
    rospy.loginfo(tcolors.MAGENTA + "End MoveArmTest_TurtlexArmEEPose, results==" + str(result_list) + tcolors.ENDC)

        
if __name__ == '__main__':
    
    rospy.init_node('turtlex_arm_moveit_test', anonymous=True, log_level=rospy.DEBUG)
    
    time.sleep(10)
    #rospy.wait_for_service('/gazebo/unpause_physics')
    #unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    #unpause()

    #rospy.loginfo(tcolors.MAGENTA + "--- Starting MoveArmTest1() ---" + tcolors.ENDC)
    #MoveArmTest1()

    #rospy.loginfo(tcolors.MAGENTA + "--- Starting MoveArmTest2() ---" + tcolors.ENDC)
    #MoveArmTest2()

    rospy.loginfo(tcolors.MAGENTA + "--- Starting MoveArmPicking() ---" + tcolors.ENDC)
    MoveArmPicking()

    rospy.loginfo(tcolors.MAGENTA + "--- Starting MoveArmTest_OnePosition() ---" + tcolors.ENDC)
    MoveArmTest_OnePosition()

    #rospy.loginfo(tcolors.MAGENTA + "--- Starting MoveArmTest_TurtlexArmEEPose() ---" + tcolors.ENDC)
    #MoveArmTest_TurtlexArmEEPose()
    