import rospy
import numpy as np
from gym import spaces
import turtlex_env
from gym.envs.registration import register
from geometry_msgs.msg import Point

import random
import math
from tf.transformations import euler_from_quaternion
from collections import deque
from utils import tcolors


register(
        id='MyTurtlexOffice-v0',
        entry_point='turtlex_office:MyTurtlexOfficeEnv',
        max_episode_steps=10000)

class MyTurtlexOfficeEnv(turtlex_env.TurtlexEnv):

    def __init__(self):
        """
        This Task Env is designed for having the Turtlex in an office.
        It will learn how to move around the maze without crashing.
        """
        
        self.get_params()

        super(MyTurtlexOfficeEnv, self).__init__()

        # We set the reward range, which is not compulsory
        self.reward_range = (-np.inf, np.inf)

        action_low = np.array([self.action_v_min, self.action_w_min])
        action_high = np.array([self.action_v_max, self.action_w_max])
        self.action_space = spaces.Box(np.float32(action_low), np.float32(action_high))

        laser_scan = self._check_laser_scan_ready()
        self.max_laser_value = laser_scan.range_max # Hokuyo sensor's range_max = 30.0
        self.min_laser_value = laser_scan.range_min # Hokuyo sensor's range_min = 0.10000000149011612
        #obs_high = np.full((self.new_ranges), self.max_laser_value)
        #obs_low = np.full((self.new_ranges), self.min_laser_value)
        obs_high = np.full((self.n_sectors), self.max_laser_value)
        obs_low = np.full((self.n_sectors), self.min_laser_value)

        #goal_x_max = max(self.goal_x_list)
        #goal_x_min = min(self.goal_x_list)
        #goal_y_max = max(self.goal_y_list)
        #goal_y_min = min(self.goal_y_list)
        #obs_high = np.concatenate((obs_high, [self.world_x_max, self.world_y_max, goal_x_max, goal_y_max]))
        #obs_low = np.concatenate((obs_low, [self.world_x_min, self.world_x_min, goal_x_min, goal_y_min]))

        self.desired_point = Point()

        max_point = Point()
        max_point.x = self.world_x_max
        max_point.y = self.world_y_max
        min_point = Point()
        min_point.x = self.world_x_min
        min_point.y = self.world_y_min
        max_distance = self.get_distance_from_point(max_point, min_point) # TODO questo non e detto sia proprio la max distance. Va bene per il box?
        obs_high = np.concatenate((obs_high, action_high, [math.pi, max_distance]))
        obs_low = np.concatenate((obs_low, action_low, [-math.pi, 0.0]))

        self.observation_space = spaces.Box(np.float32(obs_low), np.float32(obs_high))

        """
        self.observation_space = spaces.Dict(dict(
            laser_readings=spaces.Box(self.min_laser_value, self.max_laser_value, shape=obs['laser_readings'].shape, dtype='float32'),
            past_action=spaces.Box(action_low, action_high, shape=obs['past_action'].shape, dtype='float32'),
            heading=spaces.Box(-math.pi, math.pi, shape=obs['heading'].shape, dtype='float32'),
            goal_dist=spaces.Box(0.0, max_distance, shape=obs['goal_dist'].shape, dtype='float32')))
        """

        #self.last_goal_idx = -1
        #self.goal_idx = -1
        self.goal_to_solve_idx = 0

        self.overall_reward = 0 # sum of the rewards of all the previous and current episodes
        self.overall_steps = 0 # sum of the steps of all the previous and current episodes

        self.consecutive_goals = 0
        self.consecutive_goal_threshold = 10
        #self.score_hist_length = 30
        #self.score_history = deque(maxlen=self.score_hist_length)

        if not(self.is_training):
        #if (True):
            self.goal_x_list, self.goal_y_list = self.gen_test_goals(self.testing_goals)
            self.solved_counter = 0

        rospy.loginfo("Action space types ===> " + str(self.action_space))
        rospy.loginfo("Observations space types ===> " + str(self.observation_space))

    def get_params(self):
        """
        Acquire configuration parameters
        """

        self.is_training = rospy.get_param("/turtlex/training")
        self.testing_goals = rospy.get_param("/turtlex/testing_goals")

        #number_actions = rospy.get_param('/turtlex/n_actions')
        #self.action_space = spaces.Discrete(number_actions)
        self.action_v_min = rospy.get_param("/turtlex/action_v_min")
        self.action_w_min = rospy.get_param("/turtlex/action_w_min")
        self.action_v_max = rospy.get_param("/turtlex/action_v_max")
        self.action_w_max = rospy.get_param("/turtlex/action_w_max")

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

        self.world_x_max = rospy.get_param("/turtlex/world_bounds/x_max") # self.world_bounds = rospy.get_param("/turtlex/world_bounds")
        self.world_x_min = rospy.get_param("/turtlex/world_bounds/x_min")
        self.world_y_max = rospy.get_param("/turtlex/world_bounds/y_max")
        self.world_y_min = rospy.get_param("/turtlex/world_bounds/y_min")

        # Get the desired point to reach
        #self.desired_point = Point()
        #self.desired_point.x = rospy.get_param("/turtlex/desired_pose/x")
        #self.desired_point.y = rospy.get_param("/turtlex/desired_pose/y")
        #self.desired_point.z = rospy.get_param("/turtlex/desired_pose/z")

        self.goal_x_list = rospy.get_param("/turtlex/desired_pose/x") # self.goal_list = rospy.get_param("/turtlex/desired_pose")
        self.goal_y_list = rospy.get_param("/turtlex/desired_pose/y")
        #self.goal_yaw_list = rospy.get_param("/turtlex/desired_pose/Y") # yaw

        self.n_sectors = rospy.get_param("/turtlex/n_sectors")

        self.round_value = rospy.get_param("/turtlex/rounding_value")
        self.max_idle_steps = rospy.get_param("/turtlex/max_idle_steps")

        # Rewards
        #self.forwards_reward = rospy.get_param("/turtlex/forwards_reward")
        #self.turn_reward = rospy.get_param("/turtlex/turn_reward")
        self.end_episode_points = rospy.get_param("/turtlex/end_episode_points")
        self.decrease_dist_reward = rospy.get_param("/turtlex/decrease_goal_distance")
        self.increase_dist_reward = rospy.get_param("/turtlex/increase_goal_distance")

        self.max_steps = rospy.get_param("/turtlex/nsteps")

        self.running_step = rospy.get_param("/turtlex/running_step")

        self.test_areas = rospy.get_param("/turtlex/test_areas")

    def gen_test_goals(self, goal_number):
        """
        Sample goal_number random goals from a pool of reachable locations of the office
        """

        goal_x_array = []
        goal_y_array = []

        for elem in range(goal_number):
            chosen_area = random.randrange(0, self.test_areas)
            if chosen_area == 0:
                goal_x_array.append(round(random.uniform( -2.0,  4.0), self.round_value))
                goal_y_array.append(round(random.uniform( -2.0,  1.0), self.round_value))
            elif chosen_area == 1:
                goal_x_array.append(round(random.uniform( -4.0, -2.0), self.round_value))
                goal_y_array.append(round(random.uniform( -2.0, -0.5), self.round_value))
            elif chosen_area == 2:
                goal_x_array.append(round(random.uniform( -4.0, -2.0), self.round_value))
                goal_y_array.append(round(random.uniform(  0.5,  1.5), self.round_value))
            elif chosen_area == 3:
                goal_x_array.append(round(random.uniform( -4.0, -3.5), self.round_value))
                goal_y_array.append(round(random.uniform( -2.0,  1.0), self.round_value))
            elif chosen_area == 4:
                goal_x_array.append(round(random.uniform( -0.5,  4.0), self.round_value))
                goal_y_array.append(round(random.uniform(-12.0, -4.0), self.round_value))
            elif chosen_area == 5:
                goal_x_array.append(round(random.uniform( -4.0,  0.0), self.round_value))
                goal_y_array.append(round(random.uniform(-12.0, -8.5), self.round_value))
            elif chosen_area == 6:
                goal_x_array.append(round(random.uniform( -4.0,  0.0), self.round_value))
                goal_y_array.append(round(random.uniform( -6.5, -4.0), self.round_value))
            else:
                goal_x_array.append(round(random.uniform( -4.0, -3.2), self.round_value))
                goal_y_array.append(round(random.uniform(-12.0, -4.0), self.round_value))

        #goal_x_array = [-3.0, -3.5, -3.0,  0.0,  3.2,  3.0]
        #goal_y_array = [-5.0, -7.0,-11.5,-10.0, -8.0,-11.5]

        return goal_x_array, goal_y_array

    def _set_init_pose(self):
        """
        Sets the Robot in its init pose
        """
        self.move_base(self.init_linear_forward_speed, self.init_linear_turn_speed, self.running_step, epsilon=0.05, update_rate=10)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start of an episode.
        :return:
        """

        # Set Done to false, because it is calculated asyncronously
        self.episode_done = False
        self.step_counter = 0
        self.episode_reward = 0 # this has the same function of env.cumulated_episode_reward, but I may need a copy here that gets "updated faster"
        #self.still_robot_counter = 0

        #while(self.goal_idx == self.last_goal_idx):
        #    self.goal_idx = random.randrange(0, len(self.goal_x_list))
        #while(self.goal_idx == self.last_goal_idx):
        #    if self.goal_idx != (len(self.goal_x_list) - 1):
        #        self.goal_idx += 1
        #    else:
        #        self.goal_idx = 0

        #self.desired_point.x = self.goal_x_list[self.goal_idx]
        #self.desired_point.y = self.goal_y_list[self.goal_idx]
        self.desired_point.x = self.goal_x_list[self.goal_to_solve_idx]
        self.desired_point.y = self.goal_y_list[self.goal_to_solve_idx]
        #self.desired_point.z = 0
        #self.desired_heading = self.goal_yaw_list[self.goal_idx]

        #rospy.logdebug(f"New goal index: {self.goal_idx} | last goal index: {self.last_goal_idx}")
        rospy.logdebug("desired_point.x: " + str(self.desired_point.x))
        rospy.logdebug("desired_point.y: " + str(self.desired_point.y))

        #self.last_goal_idx = self.goal_idx

        self.odometry = self.get_odom()
        self.previous_distance_from_des_point = self.get_distance_from_point(self.odometry.pose.pose.position, self.desired_point)
        self.previous_distance_from_des_point = round(self.previous_distance_from_des_point, self.round_value)
        #self.previous_action = [0.,0.]
        #self.current_action = [0.,0.]
        self.previous_action = np.array([0.,0.])
        self.current_action = np.array([0.,0.])

    def _set_action(self, action): # I switched from a pool of discrete actions to a continuous action space
        """
        This set action will Set the linear and angular speed of the turtlex
        based on the action number given.
        :param action: The action integer that sets what movement to do next.
        """

        self.step_counter += 1

        rospy.logdebug("Start Set Action ==> " + str(action))

        linear_speed = action[0]
        angular_speed = action[1]

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
        
        self.current_action = action

        # We tell Turtlex the linear and angular speed to execute
        self.move_base(linear_speed, angular_speed, self.running_step, epsilon=0.05, update_rate=10)
        
        rospy.logdebug("End Set Action ==> " + str(action))

    def _get_obs(self):
        """
        Here we state what sensor data defines our robot's observations
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")

        # We get the laser scan data
        laser_scan = self.get_laser_scan()
        
        discretized_laser_scan = self.compute_sector_averages(laser_scan, self.n_sectors)
        
        # We get the odometry so that Turtlex knows where it is
        self.odometry = self.get_odom()
        #x_position = odometry.pose.pose.position.x
        #y_position = odometry.pose.pose.position.y

        # We round to only two decimals to avoid very big Observation space
        #odometry_array = [round(x_position, self.round_value),round(y_position, self.round_value)]

        #goal_array = [self.desired_point.x, self.desired_point.y]

        #observations = discretized_laser_scan + odometry_array + goal_array

        heading = self.compute_heading_to_point(self.odometry, self.desired_point)
        goal_distance = self.get_distance_from_point(self.odometry.pose.pose.position, self.desired_point)

        for elem in self.previous_action:
            discretized_laser_scan.append(elem)
        self.previous_action = self.current_action

        observations = discretized_laser_scan + [round(heading, self.round_value), round(goal_distance, self.round_value)]

        #rospy.logdebug("Observations ==> " + str(observations))
        rospy.logdebug("Observations ==> " + str(np.asarray(observations)))
        rospy.logdebug("END Get Observation ==>")

        #return observations
        return np.asarray(observations) #TODO attento a questa conversione np.asarray + 2 float32() in altro file! + in _init_env_variables qui
        

    def _is_done(self, observations):
        
        if self.episode_done:
            rospy.loginfo(tcolors.MAGENTA + "Turtlex is too close to an obstacle, episode ended ==>" + tcolors.ENDC)
        else:
            rospy.loginfo(tcolors.MAGENTA + "Turtlex has not crashed this step ==>" + tcolors.ENDC)
       
            #current_position = Point()
            #current_position.x = observations[-4] # cambiato perche ho cambiato l'obs/state array! (1/2)
            #current_position.y = observations[-3]
            #current_position.z = 0.0

            # We check if it got to the desired point # Questo sostiuisce tutto quello sotto
            if self.is_in_desired_position(self.odometry.pose.pose.position):
            #if self.is_in_desired_position(current_position):
                self.episode_done = True

            
            """
            # We check if we are outside the Learning Space
            if current_position.x <= self.world_x_max and current_position.x > self.world_x_min:
                if current_position.y <= self.world_y_max and current_position.y > self.world_y_min:
                    rospy.logdebug("Turtlex Position is OK ==> [" + str(current_position.x) + "," + str(current_position.y) + "]")
                    
                    # We check if it got to the desired point
                    if self.is_in_desired_position(current_position):
                        self.episode_done = True
                    
                else:
                    rospy.loginfo(tcolors.MAGENTA + "Turtlex too Far in Y Pos ==> " + str(current_position.x) + tcolors.ENDC)
                    self.episode_done = True
            else:
                rospy.loginfo(tcolors.MAGENTA + "Turtlex too Far in X Pos ==> " + str(current_position.x) + tcolors.ENDC)
                self.episode_done = True
            """

        return self.episode_done


    def _compute_reward(self, observations, done):

        #current_position = Point()
        #current_position.x = observations[-4] # cambiato perche ho cambiato l'obs/state array! (2/2)
        #current_position.y = observations[-3]
        #current_position.z = 0.0

        goal_distance_difference =  observations[-1] - self.previous_distance_from_des_point # observations[-1] contains the current distance
        self.previous_distance_from_des_point = observations[-1]

        if not done:

            # If there has been a decrease in the distance from the desired point, we reward it
            if goal_distance_difference < 0.0:
                reward = self.decrease_dist_reward
                #self.still_robot_counter = 0
                #rospy.loginfo(tcolors.CYAN + f"Decreased distance from goal {self.desired_point.x, self.desired_point.y} | reward: {reward}" + tcolors.ENDC)
                rospy.loginfo(tcolors.CYAN + f"Action's outcome: decreased distance from goal {self.desired_point.x, self.desired_point.y}" +
                                             f" | {self.goal_to_solve_idx + 1}/{len(self.goal_x_list)}" + tcolors.ENDC)
            elif goal_distance_difference > 0.0:
                reward = self.increase_dist_reward
                #self.still_robot_counter = 0
                #rospy.loginfo(tcolors.CYAN + f"Increased distance from goal {self.desired_point.x, self.desired_point.y} | reward: {reward}" + tcolors.ENDC)
                rospy.loginfo(tcolors.CYAN + f"Action's outcome: increased distance from goal {self.desired_point.x, self.desired_point.y}" +
                                             f" | {self.goal_to_solve_idx + 1}/{len(self.goal_x_list)}" + tcolors.ENDC)
            else:
                reward = self.increase_dist_reward
                #self.still_robot_counter += 1
                #if self.still_robot_counter < self.max_idle_steps:
                    #reward = 0
                    #reward = self.increase_dist_reward
                    #rospy.loginfo(tcolors.CYAN + f"Distance from goal unchanged {self.desired_point.x, self.desired_point.y} | reward: {reward}" + tcolors.ENDC)
                #else:
                    #reward = self.increase_dist_reward * self.still_robot_counter
                    #rospy.loginfo(tcolors.CYAN + f"Distance from goal unchanged {self.desired_point.x, self.desired_point.y} | reward: {reward}" + tcolors.ENDC)
                rospy.loginfo(tcolors.CYAN + f"Action's outcome: distance from goal unchanged {self.desired_point.x, self.desired_point.y}" +
                                             f" | {self.goal_to_solve_idx + 1}/{len(self.goal_x_list)}" + tcolors.ENDC)

            #self.episode_reward += reward

            if self.step_counter == self.max_steps:
                if self.is_training:
                    self.consecutive_goals = 0
                    #reward = 0
                    #self.score_history.append(reward)
                    #self.score_history.append(self.episode_reward)
                else:
                    self.goal_to_solve_idx += 1
                    if self.goal_to_solve_idx == len(self.goal_x_list):
                            self.goal_to_solve_idx = 0

        else:
            if self.is_in_desired_position(self.odometry.pose.pose.position):
            #if self.is_in_desired_position(current_position):
                #rospy.loginfo(tcolors.CYAN + f"The robot has reached the goal {self.desired_point.x, self.desired_point.y) | reward: {self.end_episode_points}" + tcolors.ENDC)
                rospy.loginfo(tcolors.CYAN + f"Action's outcome: the robot has reached the goal {self.desired_point.x, self.desired_point.y}" + tcolors.ENDC)
                reward = self.end_episode_points
                #self.episode_reward += self.end_episode_points

                if self.is_training:
                    self.consecutive_goals += 1
                    if (self.consecutive_goals == self.consecutive_goal_threshold):
                        self.consecutive_goals = 0
                        self.goal_to_solve_idx += 1
                    #self.score_history.append(self.episode_reward)
                    #if (self.queue_avg(self.score_history) >= .9 * self.end_episode_points) and len(self.score_history) == self.score_hist_length:
                        #self.deque_over_value = self.queue_avg(self.score_history) # save the final average
                        #self.score_history.clear()
                        #self.goal_to_solve_idx += 1
                        if self.goal_to_solve_idx == len(self.goal_x_list):
                            self.goal_to_solve_idx = 0
                else:
                    self.solved_counter += 1
                    self.goal_to_solve_idx += 1
                    if self.goal_to_solve_idx == len(self.goal_x_list):
                        self.goal_to_solve_idx = 0
            else:
                #rospy.loginfo(tcolors.CYAN + f"The robot got too close to an obstacle | reward: {-1 * self.end_episode_points}" + tcolors.ENDC)
                rospy.loginfo(tcolors.CYAN + f"Action's outcome: the robot got too close to an obstacle" + tcolors.ENDC)
                reward = -1 * self.end_episode_points
                #self.episode_reward += -1 * self.end_episode_points

                if self.is_training:
                    self.consecutive_goals = 0
                    #self.score_history.append(self.episode_reward)
                else:
                    self.goal_to_solve_idx += 1
                    if self.goal_to_solve_idx == len(self.goal_x_list):
                        self.goal_to_solve_idx = 0

        rospy.logdebug("reward = " + str(reward))
        self.overall_reward += reward
        rospy.logdebug("overall_reward = " + str(self.overall_reward))
        self.episode_reward += reward
        rospy.logdebug("episode_reward = " + str(self.episode_reward))
        self.overall_steps += 1
        rospy.logdebug("overall_steps = " + str(self.overall_steps))
        
        return reward


    # Internal TaskEnv Methods

    def compute_sector_averages(self, data, n_sectors) -> list:
        """
        Create n_sectors sectors and compute the average of their laser readings,
        then return those averages as a list
        """
        #self.episode_done = False
        
        readings_buffer = []
        sector_readings_avg = []
        n_readings_per_sector = len(data.ranges) / n_sectors
        
        rospy.logdebug("data = " + str(data))
        rospy.logdebug("n_sectors = " + str(n_sectors))
        rospy.logdebug("n_readings_per_sector = " + str(n_readings_per_sector))
        
        for item in data.ranges:

            if item == float ('Inf') or np.isinf(item):
                readings_buffer.append(self.max_laser_value)
            elif np.isnan(item):
                readings_buffer.append(self.min_laser_value)
            else:
                readings_buffer.append(round(item, self.round_value)) # Qui ho rimosso cast a int e messo un round
                
            if (self.min_range > item > 0):
                rospy.loginfo(tcolors.MAGENTA + "Object too close >>> item = " + str(item) + " < " + str(self.min_range) + tcolors.ENDC)
                self.episode_done = True
            #else:
                #rospy.loginfo(tcolors.CYAN + "NOT done Validation >>> item = " + str(item) + " < " + str(self.min_range) + tcolors.ENDC)

            if (len(readings_buffer) == n_readings_per_sector):
                sector_readings_avg.append(sum(readings_buffer) / n_readings_per_sector)
                readings_buffer = []

        return sector_readings_avg
        
        
    def is_in_desired_position(self, current_position, epsilon=0.05) -> bool:
        """
        Returns True if the current position is similar to the desired poistion
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

    
    def get_distance_from_point(self, pstart, p_end) -> float:
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = np.array((pstart.x, pstart.y, pstart.z))
        b = np.array((p_end.x, p_end.y, p_end.z))
    
        distance = np.linalg.norm(a - b)
    
        #return round(distance, self.round_value)
        return distance


    def compute_heading_to_point(self, odometry, goal_point) -> float:
        position = odometry.pose.pose.position
        orientation = odometry.pose.pose.orientation

        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(goal_point.y - position.y, goal_point.x - position.x)

        heading = goal_angle - yaw

        #if heading > math.pi:
        while(heading > math.pi):
            heading -= 2 * math.pi

        #elif heading < -math.pi:
        while(heading < math.pi):
            heading += 2 * math.pi

        #return round(heading, self.round_value)
        return heading

    def queue_avg(self, queue):
        total = 0

        for elem in queue:
            total += elem

        return total / len(queue)
