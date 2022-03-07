import rospy
import numpy
from gym import spaces
import turtlex_env
from gym.envs.registration import register
from geometry_msgs.msg import Point

timestep_limit_per_episode = 10000 # Can be any Value

register(
        id='MyTurtlexOffice-v0',
        entry_point='turtlex_office:MyTurtlexOfficeEnv',
        max_episode_steps=timestep_limit_per_episode,
    )

class MyTurtlexOfficeEnv(turtlex_env.TurtlexEnv):
    def __init__(self):
        """
        This Task Env is designed for having the Turtlex in some kind of maze.
        It will learn how to move around the maze without crashing.
        """
        
        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlex/n_actions')
        self.action_space = spaces.Discrete(number_actions)
        
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)
        
        
        #number_observations = rospy.get_param('/turtlex/n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """
        # TODO ora mi pare ne abbiamo 14 noi: 10 settori, 2 per dove siamo, 2 per la destinazione. ce anche un parametro nel .yaml
        
        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlex/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlex/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlex/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlex/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlex/init_linear_turn_speed')
        
        #self.new_ranges = rospy.get_param('/turtlex/new_ranges')
        self.min_range = rospy.get_param('/turtlex/min_range')
        #self.max_laser_value = rospy.get_param('/turtlex/max_laser_value')
        #self.min_laser_value = rospy.get_param('/turtlex/min_laser_value')
        
        # Get Desired Point to Get
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/turtlex/desired_pose/x")
        self.desired_point.y = rospy.get_param("/turtlex/desired_pose/y")
        self.desired_point.z = rospy.get_param("/turtlex/desired_pose/z")

        self.n_sectors = rospy.get_param("/turtlex/n_sectors")
        
        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self._check_laser_scan_ready()
        self.max_laser_value = laser_scan.range_max # Hokuyo sensor's range_max = 30.0
        self.min_laser_value = laser_scan.range_min # Hokuyo sensor's range_min = 0.10000000149011612
        #num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges) # TODO RIPARATO DA ME mettendo int
        #high = numpy.full((num_laser_readings), self.max_laser_value)
        #low = numpy.full((num_laser_readings), self.min_laser_value)
        high = numpy.full((self.n_sectors), self.max_laser_value)
        low = numpy.full((self.n_sectors), self.min_laser_value)

        # We only use two integers
        self.observation_space = spaces.Box(numpy.float32(low), numpy.float32(high))
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        # TODO questo era l'output per qlearn_wall. Penso pero che fosse errato perche doveva venire (5,) e non 144: errore in "num_laser_radings"?
        #ACTION SPACES TYPE===>Discrete(3)
        #OBSERVATION SPACES TYPE===>Box(0.0, 6.0, (144,), float32)
        
        # Rewards
        self.forwards_reward = rospy.get_param("/turtlex/forwards_reward")
        self.turn_reward = rospy.get_param("/turtlex/turn_reward")
        self.end_episode_points = rospy.get_param("/turtlex/end_episode_points")

        self.cumulated_steps = 0.0

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(MyTurtlexOfficeEnv, self).__init__()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set Done to false, because it is calculated asyncronously
        self._episode_done = False
        
        odometry = self.get_odom()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(odometry.pose.pose.position)


    def _set_action(self, action): # TODO qui devo cambiare la logica delle azioni: invece di tradurre in uno dei 3 casi,
        # devo fare si che action contenga i valori di velocita lineare ed angolare
        """
        This set action will Set the linear and angular speed of the turtlex
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1: #LEFT
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2: #RIGHT
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action = "TURN_RIGHT"

        
        # We tell Turtlex the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)
        
        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtlexEnv API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()
        
        #discretized_laser_scan = self.discretize_observation(laser_scan, self.new_ranges)
        discretized_laser_scan = self.compute_sector_readings(laser_scan, self.n_sectors)
                                                                
                                                                
        # We get the odometry so that Turtlex knows where it is
        odometry = self.get_odom()
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y

        # We round to only two decimals to avoid very big Observation space
        odometry_array = [round(x_position, 2),round(y_position, 2)]
        # TODO dovrebbe andare cosi, perche la posizione desiderata la aggiungo allo stesso modo fatto qui,
        # ma nel file della fsm o il launcher del learning
        # Oppure passo qui il target a partire dall fsm o dal launcher e salvo le coordinate come self.qualcosa (forse meglio cosi)

        # We only want the X and Y position and the Yaw

        observations = discretized_laser_scan + odometry_array

        rospy.logdebug("Observations==>"+str(observations))
        rospy.logdebug("END Get Observation ==>")
        return observations
        

    def _is_done(self, observations):
        
        if self._episode_done:
            rospy.logerr("Turtlex is Too Close to wall==>")
        else:
            rospy.logerr("Turtlex didnt crash at least ==>")
       
       
            current_position = Point()
            current_position.x = observations[-2]
            current_position.y = observations[-1]
            current_position.z = 0.0
            
            MAX_X = 6.0
            MIN_X = -1.0
            MAX_Y = 3.0
            MIN_Y = -3.0
            
            # We see if we are outside the Learning Space
            
            if current_position.x <= MAX_X and current_position.x > MIN_X:
                if current_position.y <= MAX_Y and current_position.y > MIN_Y:
                    rospy.logdebug("Turtlex Position is OK ==>["+str(current_position.x)+","+str(current_position.y)+"]")
                    
                    # We see if it got to the desired point
                    if self.is_in_desired_position(current_position):
                        self._episode_done = True
                    
                    
                else:
                    rospy.logerr("Turtlex too Far in Y Pos ==>"+str(current_position.x))
                    self._episode_done = True
            else:
                rospy.logerr("Turtlex too Far in X Pos ==>"+str(current_position.x))
                self._episode_done = True

        return self._episode_done


    def _compute_reward(self, observations, done):

        current_position = Point()
        current_position.x = observations[-2]
        current_position.y = observations[-1]
        current_position.z = 0.0

        distance_from_des_point = self.get_distance_from_desired_point(current_position)
        distance_difference =  distance_from_des_point - self.previous_distance_from_des_point


        if not done:
            
            if self.last_action == "FORWARDS":
                reward = self.forwards_reward
            else:
                reward = self.turn_reward
                
            # If there has been a decrease in the distance to the desired point, we reward it
            if distance_difference < 0.0:
                rospy.logwarn("DECREASE IN DISTANCE GOOD")
                reward += self.forwards_reward
            else:
                rospy.logerr("INCREASE IN DISTANCE BAD")
                reward += 0
                
        else:
            
            if self.is_in_desired_position(current_position):
                reward = self.end_episode_points
            else:
                reward = -1*self.end_episode_points


        self.previous_distance_from_des_point = distance_from_des_point


        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        
        return reward


    # Internal TaskEnv Methods
    """
    def discretize_observation(self,data,new_ranges):
        
        #Discards all the laser readings that are not multiple in index of new_ranges
        #value.
        
        self._episode_done = False
        
        discretized_ranges = []
        mod = len(data.ranges)/new_ranges
        
        rospy.logdebug("data=" + str(data))
        rospy.logwarn("new_ranges=" + str(new_ranges))
        rospy.logwarn("mod=" + str(mod))
        
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))
                    
                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logwarn("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    

        return discretized_ranges
    """

    def compute_sector_readings(self, data, n_sectors):
        """
        Create n_sectors sectors and compute the average of their laser readings,
        then return those averages as a list
        """
        self._episode_done = False
        
        readings_buffer = []
        sector_readings_avg = []
        n_readings_per_sector = len(data.ranges) / n_sectors
        
        rospy.logdebug("data=" + str(data))
        rospy.logwarn("n_sectors=" + str(n_sectors))
        rospy.logwarn("n_readings_per_sector=" + str(n_readings_per_sector))
        
        for item in data.ranges:

            if item == float ('Inf') or numpy.isinf(item):
                readings_buffer.append(self.max_laser_value)
            elif numpy.isnan(item):
                readings_buffer.append(self.min_laser_value)
            else:
                readings_buffer.append(item) # TODO qui ho rimosso cast a int. Va bene? O devo per lo meno fare round a tipo la 2nda cifra decimale?
                
            if (self.min_range > item > 0): # TODO is ""> 0" superfluous? or even wrong?
                rospy.logerr("done Validation >>> item=" + str(item) + " < " + str(self.min_range))
                self._episode_done = True
            else:
                rospy.logwarn("NOT done Validation >>> item=" + str(item) + " < " + str(self.min_range))

            if (len(readings_buffer) == n_readings_per_sector):

                sector_readings_avg.append(sum(readings_buffer) / n_readings_per_sector)
                readings_buffer = []


        return sector_readings_avg
        
        
    def is_in_desired_position(self,current_position, epsilon=0.05):
        """
        It return True if the current position is similar to the desired poistion
        """
        
        is_in_desired_pos = False
        
        
        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon
        
        x_current = current_position.x
        y_current = current_position.y
        
        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)
        
        is_in_desired_pos = x_pos_are_close and y_pos_are_close
        
        return is_in_desired_pos
        
        
    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)
    
        return distance
    
    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))
    
        distance = numpy.linalg.norm(a - b)
    
        return distance
