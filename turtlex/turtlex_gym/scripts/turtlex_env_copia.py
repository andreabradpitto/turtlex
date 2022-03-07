import rospy
import time
import numpy as np
from openai_ros import robot_gazebo_env
#from std_msgs.msg import Float64
#from sensor_msgs.msg import JointState
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
        
        The Sensors: The sensors accesible are the ones considered usefull for AI learning.
        
        Sensor Topic List:
        * /odom : Odometry readings of the Base of the Robot
        * /camera/depth/image_raw: 2d Depth image of the depth sensor.
        * /camera/depth/points: Pointcloud sensor readings
        * /camera/rgb/image_raw: RGB camera
        * /kobuki/laser/scan: Laser Readings
        
        Actuators Topic List: /cmd_vel, 
        
        Args:
        """

        rospy.logdebug("Start TurtlexEnv INIT...")
        # Variables that we give through the constructor (i.e. the "__init__()""): none in this case

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(TurtlexEnv, self).__init__(controllers_list=self.controllers_list,
                                         robot_name_space=self.robot_name_space,
                                         reset_controls=False,
                                         start_init_physics_parameters=False, # TODO prova a metterlo True
                                         reset_world_or_sim="WORLD")

        self.gazebo.unpauseSim()

        self.odometry_topic = "/odom"
        self.depth_raw_image_topic = "/camera/depth/image_raw"
        self.depth_points_image_topic = "/camera/depth/points"
        self.rgb_raw_image_topic = "/camera/rgb/image_raw"
        self.laser_scan_topic = "/kobuki/laser/scan"
        self.cmd_vel_topic = "/cmd_vel"

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber(self.odometry_topic, Odometry, self._odom_callback)
        #rospy.Subscriber(self.depth_raw_image_topic, Image, self._camera_depth_image_raw_callback)
        #rospy.Subscriber(self.depth_points_image_topic, PointCloud2, self._camera_depth_points_callback)
        #rospy.Subscriber(self.rgb_raw_image_topic, Image, self._camera_rgb_image_raw_callback)
        rospy.Subscriber(self.laser_scan_topic, LaserScan, self._laser_scan_callback)

        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)

        #self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        self._check_publishers_connection()

        #self.gazebo.pauseSim() # TODO rimosso dopo aver creato i topic come attributi. Non so perche ma poi ho messo il super in
                               # turtlex_office all'inizio e ora se faccio andare mi si pausa la simulazione: per questo ho commentato qui
        
        rospy.logdebug("Finished TurtlexEnv INIT...")

    # RobotGazeboEnv virtual methods
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True

    # TurtlexEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):

        rospy.logdebug("START ALL SENSORS READY")
        self._check_odom_ready()
        # We dont need to check for the moment, takes too long
        #self._check_camera_depth_image_raw_ready()
        #self._check_camera_depth_points_ready()
        #self._check_camera_rgb_image_raw_ready()
        self._check_laser_scan_ready()
        rospy.logdebug("ALL SENSORS READY")

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
                rospy.logerr("Current " + str(self.depth_raw_image_topic) + " not ready yet, retrying for getting camera_depth_image_raw")
        return self.camera_depth_image_raw

    def _check_camera_depth_points_ready(self):
        self.camera_depth_points = None
        rospy.logdebug("Waiting for " + str(self.depth_points_image_topic) + " to be READY...")
        while self.camera_depth_points is None and not rospy.is_shutdown():
            try:
                self.camera_depth_points = rospy.wait_for_message(self.depth_points_image_topic, PointCloud2, timeout=10.0)
                rospy.logdebug("Current " + str(self.depth_points_image_topic) + " READY=>")

            except:
                rospy.logerr("Current " + str(self.depth_points_image_topic) + " not ready yet, retrying for getting camera_depth_points")
        return self.camera_depth_points

    def _check_camera_rgb_image_raw_ready(self):
        self.camera_rgb_image_raw = None
        rospy.logdebug("Waiting for " + str(self.rgb_raw_image_topic) + " to be READY...")
        while self.camera_rgb_image_raw is None and not rospy.is_shutdown():
            try:
                self.camera_rgb_image_raw = rospy.wait_for_message(self.rgb_raw_image_topic, Image, timeout=5.0)
                rospy.logdebug("Current " + str(self.rgb_raw_image_topic) + " READY=>")

            except:
                rospy.logerr("Current " + str(self.rgb_raw_image_topic) + " not ready yet, retrying for getting camera_rgb_image_raw")
        return self.camera_rgb_image_raw

    def _check_laser_scan_ready(self):
        self.laser_scan = None
        rospy.logdebug("Waiting for " + str(self.laser_scan_topic) + " to be READY...")
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message(self.laser_scan_topic, LaserScan, timeout=5.0)
                rospy.logdebug("Current " + str(self.laser_scan_topic) + " READY=>")

            except:
                rospy.logerr("Current " + str(self.laser_scan_topic) + " not ready yet, retrying for getting laser_scan")
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
        rate = rospy.Rate(10) # 10 Hz
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
        
    # Methods that the TrainingEnvironment will need
    # ----------------------------

    def move_base(self, linear_speed, angular_speed, epsilon=0.05, update_rate=10, min_laser_distance=-1):
        """
        It will move the base based on the linear and angular speeds given.
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
        time.sleep(0.2)
        #time.sleep(0.02)
        """
        self.wait_until_twist_achieved(cmd_vel_value,
                                        epsilon,
                                        update_rate,
                                        min_laser_distance)
        """
                        
    
    def wait_until_twist_achieved(self, cmd_vel_value, epsilon, update_rate, min_laser_distance=-1):
        """
        We wait for the cmd_vel twist given to be reached by the robot reading
        from the odometry.
        :param cmd_vel_value: Twist we want to wait to reach.
        :param epsilon: Error acceptable in odometry readings.
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        rospy.logwarn("START wait_until_twist_achieved...")
        
        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0
        epsilon = 0.05
        
        rospy.logdebug("Desired Twist Cmd>>" + str(cmd_vel_value))
        rospy.logdebug("epsilon>>" + str(epsilon))
        
        linear_speed = cmd_vel_value.linear.x
        angular_speed = cmd_vel_value.angular.z
        
        linear_speed_plus = linear_speed + epsilon
        linear_speed_minus = linear_speed - epsilon
        angular_speed_plus = angular_speed + epsilon
        angular_speed_minus = angular_speed - epsilon
        
        while not rospy.is_shutdown():
            
            crashed_into_something = self.has_crashed(min_laser_distance)
            
            current_odometry = self._check_odom_ready()
            odom_linear_vel = current_odometry.twist.twist.linear.x
            odom_angular_vel = current_odometry.twist.twist.angular.z
            
            rospy.logdebug("Linear VEL=" + str(odom_linear_vel) + ", ?RANGE=[" + str(linear_speed_minus) + ","+str(linear_speed_plus)+"]")
            rospy.logdebug("Angular VEL=" + str(odom_angular_vel) + ", ?RANGE=[" + str(angular_speed_minus) + ","+str(angular_speed_plus)+"]")
            
            linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (odom_linear_vel > linear_speed_minus)
            angular_vel_are_close = (odom_angular_vel <= angular_speed_plus) and (odom_angular_vel > angular_speed_minus)
            
            if linear_vel_are_close and angular_vel_are_close:
                rospy.logwarn("Reached Velocity!")
                end_wait_time = rospy.get_rostime().to_sec()
                break
            
            if crashed_into_something:
                rospy.logerr("Turtlex has crashed, stopping movement!")
                break
            
            rospy.logwarn("Not there yet, keep waiting...")
            rate.sleep()
        delta_time = end_wait_time- start_wait_time
        rospy.logdebug("[Wait Time=" + str(delta_time)+"]")
        
        rospy.logwarn("END wait_until_twist_achieved...")
        
        return delta_time
        
    def has_crashed(self, min_laser_distance):
        """
        It states based on the laser scan if the robot has crashed or not.
        Crashed means that the minimum laser reading is lower than the
        min_laser_distance value given.
        If min_laser_distance == -1, it returns always false, because its the way
        to deactivate this check.
        """
        robot_has_crashed = False
        
        if min_laser_distance != -1:
            laser_data = self.get_laser_scan()
            for i, item in enumerate(laser_data.ranges):
                if item == float ('Inf') or np.isinf(item):
                    pass
                elif np.isnan(item):
                   pass
                else:
                    # Has a Non Infinite or Nan Value
                    if (item < min_laser_distance):
                        rospy.logerr("Turtlex HAS CRASHED >>> item=" + str(item)+"< "+str(min_laser_distance))
                        robot_has_crashed = True
                        break
        return robot_has_crashed
        

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
        
    def reinit_sensors(self):
        """
        This method is for the tasks so that when resetting the episode
        the sensors values are forced to be updated with the real data
        """
