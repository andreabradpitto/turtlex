import rospy
import time
from openai_ros import robot_gazebo_env
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class TurtlexEnv(robot_gazebo_env.RobotGazeboEnv):
    """
    Superclass for all Turtlex environments.
    """

    def __init__(self):
        """
        Initializes a new TurtlexEnv environment.
        Turtlex doesn't use controller_manager, therefore we wont reset the 
        controllers in the standard fashion. For the moment we wont reset them.
        
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        
        The Sensors: The sensors accesible are the ones considered useful for AI learning.
        
        Sensor Topic List:
        * /odom : Odometry readings of the Base of the Robot
        * /camera/depth/image_raw: 2d Depth image of the depth sensor.
        * /camera/depth/points: Pointcloud sensor readings
        * /camera/rgb/image_raw: RGB camera
        * /kobuki/laser/scan: Laser Readings
        
        Actuators Topic List: /cmd_vel, 
        
        Args:
        """

        rospy.logdebug("Entered TurltexEnv __init__")
        # Variables that we give through the constructor (i.e. the "__init__()""): none in this case

        # Internal variables
        self.controllers_list = []  # Does not have any accesibles
        self.robot_name_space = ""  # It does not use namespace
        self.reset_controls = False

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(TurtlexEnv, self).__init__(controllers_list=self.controllers_list,
                                         robot_name_space=self.robot_name_space,
                                         reset_controls=self.reset_controls,
                                         start_init_physics_parameters=False, # TODO prova a metterlo True
                                         reset_world_or_sim="WORLD")

        self.gazebo.unpauseSim()

        self.odometry_topic = "/odom"
        self.depth_raw_image_topic = "/camera/depth/image_raw"  # Sensor topic
        self.depth_points_image_topic = "/camera/depth/points"  # Sensor topic
        self.rgb_raw_image_topic = "/camera/rgb/image_raw"  # Sensor topic
        self.laser_scan_topic = "/kobuki/laser/scan"  # Sensor topic
        self.cmd_vel_topic = "/cmd_vel"  # Actuator topic

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber(self.odometry_topic, Odometry, self._odom_callback)
        #rospy.Subscriber(self.depth_raw_image_topic, Image, self._camera_depth_image_raw_callback)
        #rospy.Subscriber(self.depth_points_image_topic, PointCloud2, self._camera_depth_points_callback)
        #rospy.Subscriber(self.rgb_raw_image_topic, Image, self._camera_rgb_image_raw_callback)
        rospy.Subscriber(self.laser_scan_topic, LaserScan, self._laser_scan_callback)

        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)

        self._check_all_systems_ready()
        
        rospy.logdebug("Finished TurltexEnv __init__")


    # RobotGazeboEnv virtual methods
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are operational
        """
        self._check_all_sensors_ready()
        self._check_publishers_connection()

        return True

    # TurtlexEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):

        rospy.logdebug("START _check_all_sensors_ready")
        self._check_odom_ready()
        # We dont need to check for the moment, takes too long
        #self._check_camera_depth_image_raw_ready()
        #self._check_camera_depth_points_ready()
        #self._check_camera_rgb_image_raw_ready()
        self._check_laser_scan_ready()
        rospy.logdebug("END _check_all_sensors_ready")

    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("Waiting for " + str(self.odometry_topic) + " to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message(self.odometry_topic, Odometry, timeout=5.0)
                rospy.logdebug("Current " + str(self.odometry_topic) + " READY=>")

            except:
                rospy.logerr("Current " + str(self.odometry_topic) + " not ready yet, retrying for getting odom")

        return self.odom

    def _check_camera_depth_image_raw_ready(self):
        self.camera_depth_image_raw = None
        rospy.logdebug("Waiting for " + str(self.depth_raw_image_topic) + " to be READY...")
        while self.camera_depth_image_raw is None and not rospy.is_shutdown():
            try:
                self.camera_depth_image_raw = rospy.wait_for_message(self.depth_raw_image_topic, Image, timeout=5.0)
                rospy.logdebug("Current " + str(self.depth_raw_image_topic) + " READY=>")

            except:
                rospy.logerr("Current " + str(self.depth_raw_image_topic) + " not ready yet, retrying...")
        return self.camera_depth_image_raw

    def _check_camera_depth_points_ready(self):
        self.camera_depth_points = None
        rospy.logdebug("Waiting for " + str(self.depth_points_image_topic) + " to be READY...")
        while self.camera_depth_points is None and not rospy.is_shutdown():
            try:
                self.camera_depth_points = rospy.wait_for_message(self.depth_points_image_topic, PointCloud2, timeout=10.0)
                rospy.logdebug("Current " + str(self.depth_points_image_topic) + " READY=>")

            except:
                rospy.logerr("Current " + str(self.depth_points_image_topic) + " not ready yet, retrying...")
        return self.camera_depth_points

    def _check_camera_rgb_image_raw_ready(self):
        self.camera_rgb_image_raw = None
        rospy.logdebug("Waiting for " + str(self.rgb_raw_image_topic) + " to be READY...")
        while self.camera_rgb_image_raw is None and not rospy.is_shutdown():
            try:
                self.camera_rgb_image_raw = rospy.wait_for_message(self.rgb_raw_image_topic, Image, timeout=5.0)
                rospy.logdebug("Current " + str(self.rgb_raw_image_topic) + " READY=>")

            except:
                rospy.logerr("Current " + str(self.rgb_raw_image_topic) + " not ready yet, retrying...")
        return self.camera_rgb_image_raw

    def _check_laser_scan_ready(self):
        self.laser_scan = None
        rospy.logdebug("Waiting for " + str(self.laser_scan_topic) + " to be READY...")
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message(self.laser_scan_topic, LaserScan, timeout=5.0)
                rospy.logdebug("Current " + str(self.laser_scan_topic) + " READY=>")

            except:
                rospy.logerr("Current " + str(self.laser_scan_topic) + " not ready yet, retrying...")
        return self.laser_scan

    def _odom_callback(self, data):
        self.odom = data

    def _camera_depth_image_raw_callback(self, data):
        self.camera_depth_image_raw = data

    def _camera_depth_points_callback(self, data):
        self.camera_depth_points = data

    def _camera_rgb_image_raw_callback(self, data):
        self.camera_rgb_image_raw = data

    def _laser_scan_callback(self, data):
        self.laser_scan = data

    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10 Hz
        while self.cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No subscribers to self.cmd_vel_pub yet, so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("self.cmd_vel_pub publisher connected")

        rospy.logdebug("All publishers READY")
    
    # Methods that the TrainingEnvironment will need to be defined here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment
    # ----------------------------

    def _set_init_pose(self):
        """
        Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """
        Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """
        Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        """
        Acquires the current state/obervation
        """
        raise NotImplementedError()

    def _is_done(self, observations):
        """
        Checks if episode done based on observations given.
        """
        raise NotImplementedError()
        
    # Methods that the TrainingEnvironment will need
    # ----------------------------

    def move_base(self, linear_speed, angular_speed, running_step=0.2, epsilon=0.05, update_rate=10, min_laser_distance=-1):
        """
        This will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return: 
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("Turtlex Base Twist Cmd >> " + str(cmd_vel_value))
        self._check_publishers_connection()
        self.cmd_vel_pub.publish(cmd_vel_value)
        time.sleep(running_step)

    def get_odom(self):
        return self.odom
        
    def get_camera_depth_image_raw(self):
        return self.camera_depth_image_raw
        
    def get_camera_depth_points(self):
        return self.camera_depth_points
        
    def get_camera_rgb_image_raw(self):
        return self.camera_rgb_image_raw
        
    def get_laser_scan(self):
        return self.laser_scan
