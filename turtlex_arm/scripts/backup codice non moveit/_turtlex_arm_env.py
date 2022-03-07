import rospy
import sys
import time
import numpy as np
from std_msgs.msg import Float64
import geometry_msgs.msg
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates #TODO
import tf
#from nav_msgs.msg import Odometry
#import trajectory_msgs.msg
from utils import tcolors

from openai_ros import robot_gazebo_env


class TurtlexArmEnv(robot_gazebo_env.RobotGazeboEnv):

    def __init__(self):
        rospy.logdebug("Entered TurltexArmEnv")
        
        self.controllers_list = []
        self.robot_name_space = ""
        self.reset_controls = False
        
        super(TurtlexArmEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=self.reset_controls,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")

        self.gazebo.unpauseSim()

        # We start all the ROS related subscribers and publishers
        self.joints_state_topic = '/joint_states'
        self.joints_name = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper_prismatic_joint_1", "gripper_prismatic_joint_2"]

        rospy.Subscriber(self.joints_state_topic, JointState, self.joints_state_callback)
        self.joints_state = JointState()

        self.joint_1_topic = ("/" + self.joints_name[0] + "_postion_controller/command")
        self.joint_2_topic = ("/" + self.joints_name[1] + "_postion_controller/command")
        self.joint_3_topic = ("/" + self.joints_name[2] + "_postion_controller/command")
        self.joint_4_topic = ("/" + self.joints_name[3] + "_postion_controller/command")
        self.joint_5_topic = ("/" + self.joints_name[4] + "_postion_controller/command")

        self.joint_1_pub = rospy.Publisher(self.joint_1_topic, Float64, queue_size=1)
        self.joint_2_pub = rospy.Publisher(self.joint_2_topic, Float64, queue_size=1)
        self.joint_3_pub = rospy.Publisher(self.joint_3_topic, Float64, queue_size=1)
        self.joint_4_pub = rospy.Publisher(self.joint_4_topic, Float64, queue_size=1)
        self.joint_5_pub = rospy.Publisher(self.joint_5_topic, Float64, queue_size=1)
        
        self._check_all_systems_ready()

        self.gazebo.pauseSim() # TODO forse da commentare

    # RobotGazeboEnv virtual methods
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True

    # TurtlexArmEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):

        self._check_joints_state_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_joints_state_ready(self):
        self.joints_state = None
        while self.joints_state is None and not rospy.is_shutdown():
            try:
                self.joints_state = rospy.wait_for_message(self.joints_state_topic, JointState, timeout=1.0)
                rospy.logdebug("Current " + str(self.joints_state_topic) + " READY => " + str(self.joints_state))

            except:
                rospy.logerr("Current " + str(self.joints_state_topic) + " not ready yet, retrying...")

        return self.joints_state
    
    def joints_state_callback(self, data):
        self.joints_state = data

    def get_joints_state(self):
        return self.joints_state
        
    def get_joint_names(self):
        return self.joints_state.name


    def move_arm(self, joints_orientation):
        """
        It will move the arm based on the joints orientation given.
        """
        #orientation = Float64() # TODO non so se va come ho messo sotto, magari serve
        rospy.logdebug("TurtlexArm joints_orientation >> " + str(joints_orientation))
        self.joint_1_pub.publish(joints_orientation[0])
        self.joint_2_pub.publish(joints_orientation[1])
        self.joint_3_pub.publish(joints_orientation[2])
        self.joint_4_pub.publish(joints_orientation[3])
        self.joint_5_pub.publish(joints_orientation[4])
        time.sleep(0.2) # TODO da cancellare? Forse serve per far andare a buon fine il movimento prima del gazebo.pauseSim(), forse no




    def set_trajectory_ee(self, action):
        """
        Sets the Pose of the EndEffector based on the action variable.
        The action variable contains the position and orientation of the EndEffector.
        See create_action
        """
        # Set up a trajectory message to publish.
        ee_target = geometry_msgs.msg.Pose()
        ee_target.orientation.w = 1.0
        ee_target.position.x = action[0]
        ee_target.position.y = action[1]
        ee_target.position.z = action[2]
        
        rospy.logdebug("Set EE Trajectory START: POSITION = " + str(ee_target.position))
        result = self.move_fetch_object.ee_traj(ee_target)
        rospy.logdebug("Set EE Trajectory END: RESULT = " + str(result))
        
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
        
    def create_joints_dict(self, joints_positions):
        """
        Based on the Order of the positions, they will be assigned to its joint name
        """
        
        assert len(joints_positions) == len(self.joints_name), "Wrong number of joints, there should be " + str(len(self.joints_name))
        joints_dict = dict(zip(self.joints_name, joints_positions))
        
        return joints_dict
        
    def get_ee_pose(self):
        """
        Returns geometry_msgs/PoseStamped
            std_msgs/Header header
              uint32 seq
              time stamp
              string frame_id
            geometry_msgs/Pose pose
              geometry_msgs/Point position
                float64 x
                float64 y
                float64 z
              geometry_msgs/Quaternion orientation
                float64 x
                float64 y
                float64 z
                float64 w
        """
        self.gazebo.unpauseSim()
        gripper_pose = self.move_fetch_object.ee_pose()
        self.gazebo.pauseSim()
        
        return gripper_pose
        
    def get_ee_rpy(self):
        
        gripper_rpy = self.move_fetch_object.ee_rpy()
        
        return gripper_rpy


    # Methods that the TrainingEnvironment will need to be defined here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment
    # ----------------------------

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
