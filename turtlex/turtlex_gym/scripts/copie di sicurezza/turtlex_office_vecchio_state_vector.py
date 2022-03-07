import rospy
import numpy as np
from gym import spaces
import turtlex_env
from gym.envs.registration import register
from geometry_msgs.msg import Point

import random

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
        #number_actions = rospy.get_param('/turtlex/n_actions')
        #self.action_space = spaces.Discrete(number_actions)
        self.action_v_min = rospy.get_param("/turtlex/action_v_min")
        self.action_w_min = rospy.get_param("/turtlex/action_w_min")
        self.action_v_max = rospy.get_param("/turtlex/action_v_max")
        self.action_w_max = rospy.get_param("/turtlex/action_w_max")
        action_low = np.array([self.action_v_min, self.action_w_min])
        action_high = np.array([self.action_v_max, self.action_w_max])
        self.action_space = spaces.Box(np.float32(action_low), np.float32(action_high))
        
        # We set the reward range, which is not compulsory
        self.reward_range = (-np.inf, np.inf)
        
        #number_observations = rospy.get_param('/turtlex/n_observations')
        
        # Actions and Observations
        #self.linear_forward_speed = rospy.get_param('/turtlex/linear_forward_speed')
        #self.linear_turn_speed = rospy.get_param('/turtlex/linear_turn_speed')
        #self.angular_speed = rospy.get_param('/turtlex/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlex/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlex/init_linear_turn_speed')
        
        #self.new_ranges = rospy.get_param('/turtlex/new_ranges')
        self.min_range = rospy.get_param('/turtlex/min_range')
        #self.max_laser_value = rospy.get_param('/turtlex/max_laser_value')
        #self.min_laser_value = rospy.get_param('/turtlex/min_laser_value')

        self.world_x_max = rospy.get_param("/turtlex/world_bounds/x_max")
        self.world_x_min = rospy.get_param("/turtlex/world_bounds/x_min")
        self.world_y_max = rospy.get_param("/turtlex/world_bounds/y_max")
        self.world_y_min = rospy.get_param("/turtlex/world_bounds/y_min")

        # Get Desired Point to Get
        self.desired_point = Point()
        #self.desired_point.x = rospy.get_param("/turtlex/desired_pose/x")
        #self.desired_point.y = rospy.get_param("/turtlex/desired_pose/y")
        #self.desired_point.z = rospy.get_param("/turtlex/desired_pose/z")

        self.last_goal_idx = -1
        self.goal_idx = 0 # TODO poi metti -1 anche qui così si triggera la scelta casuale anche del primo goal
        self.goal_x_list = rospy.get_param("/turtlex/desired_pose/x")
        self.goal_y_list = rospy.get_param("/turtlex/desired_pose/y")
        #self.goal_yaw_list = rospy.get_param("/turtlex/desired_pose/Y") # yaw

        self.n_sectors = rospy.get_param("/turtlex/n_sectors")

        #self.past_action = np.array([0.,0.]) # aggiunto da me per non dover passare due valori
                                              # (azione de-normalizz. + azione passata) ad env.step()

        self.still_robot_counter = 0
        
        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self._check_laser_scan_ready()
        self.max_laser_value = laser_scan.range_max # Hokuyo sensor's range_max = 30.0
        self.min_laser_value = laser_scan.range_min # Hokuyo sensor's range_min = 0.10000000149011612
        #obs_high = np.full((self.new_ranges), self.max_laser_value)
        #obs_low = np.full((self.new_ranges), self.min_laser_value)
        obs_high = np.full((self.n_sectors), self.max_laser_value)
        obs_low = np.full((self.n_sectors), self.min_laser_value)

        goal_x_max = max(self.goal_x_list)
        goal_x_min = min(self.goal_x_list)
        goal_y_max = max(self.goal_y_list)
        goal_y_min = min(self.goal_y_list)
        obs_high = np.concatenate((obs_high, [self.world_x_max, self.world_y_max, goal_x_max, goal_y_max]))
        obs_low = np.concatenate((obs_low, [self.world_x_min, self.world_x_min, goal_x_min, goal_y_min]))

        # We only use two integers
        self.observation_space = spaces.Box(np.float32(obs_low), np.float32(obs_high))
        
        rospy.logdebug("ACTION SPACES TYPE ===> " + str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE ===> " + str(self.observation_space))
        # Questo era l'output per qlearn_wall:
        #ACTION SPACES TYPE===>Discrete(3)
        #OBSERVATION SPACES TYPE===>Box(0.0, 6.0, (5,), float32)
        
        # Rewards
        #self.forwards_reward = rospy.get_param("/turtlex/forwards_reward")
        #self.turn_reward = rospy.get_param("/turtlex/turn_reward")
        self.end_episode_points = rospy.get_param("/turtlex/end_episode_points")

        self.decrease_dist_reward = rospy.get_param("/turtlex/decrease_goal_distance")
        self.increase_dist_reward = rospy.get_param("/turtlex/increase_goal_distance")

        self.cumulated_steps = 0

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(MyTurtlexOfficeEnv, self).__init__()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base(self.init_linear_forward_speed, self.init_linear_turn_speed, epsilon=0.05, update_rate=10)

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

        self.still_robot_counter = 0
        
        while(self.goal_idx == self.last_goal_idx):
            self.goal_idx = random.randrange(0, len(self.goal_x_list))

        self.desired_point.x = self.goal_x_list[self.goal_idx]
        self.desired_point.y = self.goal_y_list[self.goal_idx]
        #self.desired_point.z = 0
        #self.desired_heading = self.goal_yaw_list[self.goal_idx] # In caso guarda getOdometry() di environment_stage_1.py

        rospy.logdebug(f"New goal index: {self.goal_idx} | last goal index: {self.last_goal_idx}")
        rospy.logdebug("desired_point.x: " + str(self.desired_point.x))
        rospy.logdebug("desired_point.y: " + str(self.desired_point.y))

        self.last_goal_idx = self.goal_idx

        odometry = self.get_odom()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(odometry.pose.pose.position)


    def _set_action(self, action): # I switched from a pool of discrete actions to a continuous action space
        """
        This set action will Set the linear and angular speed of the turtlex
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        linear_speed = action[0]
        angular_speed = action[1]
        
        rospy.logdebug("Start Set Action ==> " + str(action))
        """
        # We convert the actions to speed movements to send to the parent class TurtlexEnv
        if action == 0: #FORWARDS
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1: #LEFT
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2: #RIGHT
            linear_speed = self.linear_turn_speed
            angular_speed = -1 * self.angular_speed
            self.last_action = "TURN_RIGHT"
        """
        
        # We tell Turtlex the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)
        
        rospy.logdebug("END Set Action ==> " + str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robot's observations
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

        goal_array = [self.desired_point.x, self.desired_point.y]

        observations = discretized_laser_scan + odometry_array + goal_array

        # TODO MOLTO ATTENTO qui peche in getState() di environment_stage_1.py il return observations e fatto da:
        # scan_range + [heading, current_distance]
        # in cui, in scan_range, sono appesi in fondo i due valori della velocita dell'azione precedente:
        # 10 laser scans + 2 valori last actions (vel. lin. e vel. ang.) + 1 yaw risp. al goal + 1 distanza corrente dal goal

        rospy.logdebug("Observations ==> " + str(observations))
        rospy.logdebug("END Get Observation ==>")
        #return observations
        return np.asarray(observations) # Ok questo ripsetto al vecchio (linea sopra)?
        #TODO ATTENTO a questa conversione np.asarray + 2 float32() in altro file!
        

    def _is_done(self, observations):
        
        if self._episode_done:
            rospy.logerr("Turtlex is Too Close to wall, episode ended ==>")
        else:
            rospy.logerr("Turtlex has not crashed this step ==>")
       
            current_position = Point()
            current_position.x = observations[-4] # cambiato perche ho cambiato l'obs/state array! (1/2)
            current_position.y = observations[-3]
            current_position.z = 0.0

            # We check if it got to the desired point # Questo sostiuisce tutto quello sotto
            if self.is_in_desired_position(current_position):
                self._episode_done = True
            
            """
            # We check if we are outside the Learning Space
            if current_position.x <= self.world_x_max and current_position.x > self.world_x_min:
                if current_position.y <= self.world_y_max and current_position.y > self.world_y_min:
                    rospy.logdebug("Turtlex Position is OK ==> [" + str(current_position.x) + "," + str(current_position.y) + "]")
                    
                    # We check if it got to the desired point
                    if self.is_in_desired_position(current_position):
                        self._episode_done = True
                    
                else:
                    rospy.logerr("Turtlex too Far in Y Pos ==> " + str(current_position.x))
                    self._episode_done = True
            else:
                rospy.logerr("Turtlex too Far in X Pos ==> " + str(current_position.x))
                self._episode_done = True
            """

        return self._episode_done


    def _compute_reward(self, observations, done):

        current_position = Point()
        current_position.x = observations[-4] # cambiato perche ho cambiato l'obs/state array! (2/2)
        current_position.y = observations[-3]
        current_position.z = 0.0

        distance_from_des_point = self.get_distance_from_desired_point(current_position)
        distance_difference =  distance_from_des_point - self.previous_distance_from_des_point


        if not done:
            """
            if self.last_action == "FORWARDS":
                reward = self.forwards_reward
            else:
                reward = self.turn_reward
            """    
            # If there has been a decrease in the distance to the desired point, we reward it
            if distance_difference < 0.0:
                rospy.logwarn(f"Decreased distance from goal | reward delta: {self.decrease_dist_reward}")
                #reward += self.forwards_reward
                reward = self.decrease_dist_reward
                self.still_robot_counter = 0
            elif distance_difference > 0.0:
                rospy.logerr(f"Increased distance from goal | reward delta: {self.increase_dist_reward}")
                #reward += 0
                reward = self.increase_dist_reward
                self.still_robot_counter = 0
            #elif 0.0 <= distance_difference <= 0.0: # questo assolutamente da ottimizare. Deve essere un else alla fine
            else:
                self.still_robot_counter += self.still_robot_counter
                if self.still_robot_counter < 5:
                    reward = 0
                else:
                    reward = self.increase_dist_reward * self.still_robot_counter

        else:
            if self.is_in_desired_position(current_position):
                reward = self.end_episode_points # the robot has reached the goal
            else:
                reward = -1 * self.end_episode_points # the robot has crashed


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
                if item == float ('Inf') or np.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif np.isnan(item):
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
        rospy.logdebug("n_sectors=" + str(n_sectors))
        rospy.logdebug("n_readings_per_sector=" + str(n_readings_per_sector))
        
        for item in data.ranges:

            if item == float ('Inf') or np.isinf(item):
                readings_buffer.append(self.max_laser_value)
            elif np.isnan(item):
                readings_buffer.append(self.min_laser_value)
            else:
                readings_buffer.append(round(item, 2)) # Qui ho rimosso cast a int e messo un round
                
            if (self.min_range > item > 0):
                rospy.logerr("done Validation >>> item = " + str(item) + " < " + str(self.min_range))
                self._episode_done = True
            #else:
                #rospy.logwarn("NOT done Validation >>> item = " + str(item) + " < " + str(self.min_range))

            if (len(readings_buffer) == n_readings_per_sector):

                sector_readings_avg.append(sum(readings_buffer) / n_readings_per_sector)
                readings_buffer = []


        return sector_readings_avg
        
        
    def is_in_desired_position(self, current_position, epsilon=0.05):
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
        distance = self.get_distance_from_point(current_position, self.desired_point)
    
        return distance
    
    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = np.array((pstart.x, pstart.y, pstart.z))
        b = np.array((p_end.x, p_end.y, p_end.z))
    
        distance = np.linalg.norm(a - b)
    
        return distance
