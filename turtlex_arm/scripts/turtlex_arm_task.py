import rospy
import numpy as np
from gym import utils
from gym import spaces
from gym.envs.registration import register
#from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point

import random
from collections import deque
from utils import tcolors

import turtlex_arm_env #from openai_ros.robot_envs import fetch_env


register(
        id='TurtlexArmTask-v0',
        entry_point='turtlex_arm_task:TurtlexArmTaskEnv',
        max_episode_steps=10000)

class TurtlexArmTaskEnv(turtlex_arm_env.TurtlexArmEnv, utils.EzPickle):

    def __init__(self):
        
        rospy.logdebug("Entered TurtlexArmTaskEnv")

        self.get_params()

        turtlex_arm_env.TurtlexArmEnv.__init__(self)

        # We set the reward range, which is not compulsory
        self.reward_range = (-np.inf, np.inf)

        action_low = np.array(self.joints_min_pos)
        action_high = np.array(self.joints_max_pos)
        self.action_space = spaces.Box(np.float32(action_low), np.float32(action_high))
        #self.action_space = spaces.Discrete(self.n_actions)

        ee_min_x = self.ee_bounds["x"][0]
        ee_max_x = self.ee_bounds["x"][1]
        ee_min_y = self.ee_bounds["y"][0]
        ee_max_y = self.ee_bounds["y"][1]
        ee_min_z = self.ee_bounds["z"][0]
        ee_max_z = self.ee_bounds["z"][1]

        goal_low_pos = np.array([ee_min_x, ee_min_y, ee_min_z])
        goal_high_pos = np.array([ee_max_x, ee_max_y, ee_max_z])

        observations_high_dist = np.array([self.max_distance])
        observations_low_dist = np.array([0.0])

        obs_low = np.concatenate((action_low, goal_low_pos, observations_low_dist))
        obs_high = np.concatenate((action_high, goal_high_pos, observations_high_dist))

        self.observation_space = spaces.Box(np.float32(obs_low), np.float32(obs_high)) # era 3D position ee (3), current distance from goal (1) = 4 observations; ora sono 9

        self.goal_to_solve_idx = 0

        self.desired_ee_goal = Point()

        self.overall_reward = 0 # sum of the rewards of all the previous and current episodes
        self.overall_steps = 0 # sum of the steps of all the previous and current episodes
        self.score_hist_length = 30
        self.score_history = deque(maxlen=self.score_hist_length)

        if not self.is_training:
            self.solved_counter = 0

        rospy.loginfo("Action space types ===> " + str(self.action_space))
        rospy.loginfo("Observations space types ===> " + str(self.observation_space))


    def get_params(self):
        # Acquire configuration parameters

        self.joints_min_pos = rospy.get_param("/turtlex_arm/joints_min_pos")
        self.joints_max_pos = rospy.get_param("/turtlex_arm/joints_max_pos")

        #self.n_actions = rospy.get_param('/turtlex_arm/n_actions')
        self.n_observations = rospy.get_param('/turtlex_arm/n_observations')

        self.round_value = rospy.get_param("/turtlex_arm/rounding_value")

        self.step_punishment = rospy.get_param('/turtlex_arm/step_punishment')
        self.closer_reward = rospy.get_param('/turtlex_arm/closer_reward')
        self.impossible_movement_punishment = rospy.get_param('/turtlex_arm/impossible_movement_punishment')
        self.reached_goal_reward = rospy.get_param('/turtlex_arm/reached_goal_reward')

        self.init_joint_pos = rospy.get_param('/turtlex_arm/init_joint_pos')
        #self.setup_ee_pos = rospy.get_param('/turtlex_arm/setup_ee_pos')
        self.goal_ee_pos = rospy.get_param('/turtlex_arm/ee_goals')
        #self.goal_ee_pos_x = rospy.get_param('/turtlex_arm/ee_goals/x')
        #self.goal_ee_pos_y = rospy.get_param('/turtlex_arm/ee_goals/y')
        #self.goal_ee_pos_z = rospy.get_param('/turtlex_arm/ee_goals/z')
        self.ee_bounds = rospy.get_param('/turtlex_arm/ee_bounds')
        self.max_distance = rospy.get_param('/turtlex_arm/max_distance')

        self.max_steps = rospy.get_param("/turtlex_arm/max_episode_steps")

        self.is_training = rospy.get_param('/turtlex_arm/training')

        
        #self.desired_position = [self.goal_ee_pos["x"], self.goal_ee_pos["y"], self.goal_ee_pos["z"]]
        #self.gripper_rotation = [1., 0., 1., 0.]

    def _set_init_pose(self):
        """
        Sets the robot in its init pose
        The Simulation will be unpaused for this purpose
        """
        # Check because it seems its not being used
        rospy.logdebug("self.init_joint_pos=" + str(self.init_joint_pos))

        # Init Joint Pose
        #rospy.logdebug("Moving to SETUP Joints")
        #self.movement_result = self.set_trajectory_joints(self.init_joint_pos)

        # Get the desired point to reach
        #self.desired_point = Point()
        #self.desired_point.x = self.goal_ee_pos_x[0]
        #self.desired_point.y = self.goal_ee_pos_y[0]
        #self.desired_point.z = self.goal_ee_pos_z[0]

        rospy.logdebug("Moving to INIT POSE Position")

        #action = self.create_action(self.init_joint_pos)
        #self.movement_result = self.set_trajectory_ee(action)

        init_joint_pos_list = []
        for value in self.init_joint_pos.values():
            init_joint_pos_list.append(value)
        self.action_result = self.move_joints(init_joint_pos_list)

        # if self.action_result:
        #     self.joints_pos = copy.deepcopy(self.action_result)
        # else:
        #     #assert False, "Desired INIT POSE is not possible"
        #     print("Desired INIT POSE is not possible")

        # TODO
        if not self.action_result:
        #    assert False, "Desired INIT POSE is not possible"
            print("Desired INIT POSE is not possible")
        
        rospy.logdebug("Init Pose Results ==> " + str(self.action_result))

        return self.action_result


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start of an episode.
        The simulation will be paused, therefore all the data retrieved has to come 
        from a system that does not need the simulation to be running, like variables where the 
        callbacks have stored last known sensor data.
        :return:
        """
        rospy.logdebug("Init Env Variables...")

        # Set Done to false, because it is calculated asyncronously
        self.episode_done = False
        self.step_counter = 0

        self.desired_ee_goal.x = self.goal_ee_pos["x"][self.goal_to_solve_idx]
        self.desired_ee_goal.y = self.goal_ee_pos["y"][self.goal_to_solve_idx]
        self.desired_ee_goal.z = self.goal_ee_pos["z"][self.goal_to_solve_idx]
        #self.desired_ee_goal_vector = [self.desired_ee_goal.x, self.desired_ee_goal.y, self.desired_ee_goal.z]
        #self.gripper_rotation = [1., 0., 1., 0.]

        rospy.logdebug("desired_ee_goal.x: " + str(self.desired_ee_goal.x))
        rospy.logdebug("desired_ee_goal.y: " + str(self.desired_ee_goal.y))
        rospy.logdebug("desired_ee_goal.z: " + str(self.desired_ee_goal.z))

        #self.gripper_position = self.move_turtlex_arm_object.ee_pose().position
        self.gripper_position = self.get_ee_pose().position
        self.prev_dist_from_des_pos_ee = self.compute_euclidean_dist(self.desired_ee_goal, self.gripper_position)
        self.prev_dist_from_des_pos_ee = round(self.prev_dist_from_des_pos_ee, self.round_value)
        rospy.logdebug("INIT DISTANCE FROM GOAL ==> " + str(self.prev_dist_from_des_pos_ee))

        rospy.logdebug("Init Env Variables...DONE")
    

    def _set_action(self, action):
        
        self.step_counter += 1

        self.action_result = self.move_joints(action)

        # Apply action to the simulation
        #action_end_effector = self.create_action(gripper_target, self.gripper_rotation)
        #self.movement_result = self.set_trajectory_ee(action_end_effector)

        # If the joints configuration was successful, we replace the last one with the new one
        #if self.action_result:
        #    self.joints_pos = copy.deepcopy(self.action_result)
        #else:
            #rospy.loginfo(tcolors.MAGENTA + "Impossible joints configuration from action: " + str(action) + tcolors.ENDC)
            #self.episode_done = True # questo va bene qui ma cerco di mettere tutto in _is_done()
        
        rospy.loginfo(tcolors.CYAN + "END Set Action ==> " + str(action) + tcolors.ENDC)

    def _get_obs(self):
        """
        It returns the Position of the TCP/EndEffector as observation,
        as well as the distance from the desired point
        At the moment, orientation is not taken into account
        """

        rospy.logdebug("Start Get Observation ==>")

        #grip_pos = self.get_ee_pose().position
        #grip_pos_array = [grip_pos.pose.position.x, grip_pos.pose.position.y, grip_pos.pose.position.z]
        #obs = grip_pos_array

        #self.gripper_position = self.move_turtlex_arm_object.ee_pose().position
        self.gripper_position = self.get_ee_pose().position
        dist_from_des_pos_ee = self.compute_euclidean_dist(self.desired_ee_goal, self.gripper_position)
        dist_from_des_pos_ee = round(dist_from_des_pos_ee, self.round_value)

        #obs = [self.gripper_position.x, self.gripper_position.y, self.gripper_position.z] # TODO anche questo può avere senso: in caso cambia numero obs in parametri .yaml da 9 a 7
        #obs = self.joints_pos # TODO ricontrolla soluzione alternativa a joints_pos
        #obs = list(self.get_joints().position[2:]) # skip the position of the two gripper prismatic joints
        all_joints_pos = self.get_joints().position # TODO questa riga e quella sotto sostituiscono quella sopra: forse questo attenua/risolve il bug che la lista delle joint pos mi veniva da 7 lo stesso, con le 2 pos del gripper in coda
        obs = list(all_joints_pos[2:])

        #new_dist_from_des_pos_ee = self.compute_euclidean_dist(self.desired_position,grip_pos_array)

        obs.extend([self.desired_ee_goal.x, self.desired_ee_goal.y, self.desired_ee_goal.z])

        obs.append(dist_from_des_pos_ee)

        rospy.logdebug("Observations ==> " + str(obs))
        #rospy.logdebug("Observations ==> " + str(np.asarray(obs)))
        rospy.logdebug("END Get Observation ==>")

        return obs
        #return np.asarray(obs)

    def _is_done(self, observations):
        
        """
        If the latest action didn't succeed, it means that the position asked was imposible, therefore the episode must end.
        It will also end if it reaches its goal.
        """

        if self.action_result:
            if self.calculate_if_done(self.desired_ee_goal, self.gripper_position):
                rospy.loginfo(tcolors.MAGENTA + "Turtlex_arm has reached the goal ==>" + tcolors.ENDC)
                self.episode_done = True
            else:
                rospy.loginfo(tcolors.MAGENTA + "Turtlex_arm has not failed this step ==>" + tcolors.ENDC)
        else:
            rospy.loginfo(tcolors.MAGENTA + "Turtlex_arm attempted a wrong joint configuration, episode ended ==>" + tcolors.ENDC)
            self.episode_done = True

        return self.episode_done
        
    def _compute_reward(self, observations, done):
        """
        We punish each step that it passes without achieveing the goal.
        Punishes differently if it reached a position that is imposible to move to.
        Rewards getting to a position close to the goal.
        """
        goal_distance_difference =  observations[-1] - self.prev_dist_from_des_pos_ee # observations[-1] contains the current distance, "dist_from_des_pos_ee"

        if not self.episode_done:

            # If there has been a decrease in the distance from the desired point, we reward it
            if goal_distance_difference < 0.0:
                reward = self.closer_reward
                rospy.loginfo(tcolors.CYAN + f"Action's outcome: decreased distance from goal \
                              {self.desired_ee_goal.x, self.desired_ee_goal.y, self.desired_ee_goal.z}" + tcolors.ENDC)
            else:
                reward = self.step_punishment
                rospy.loginfo(tcolors.CYAN + f"Action's outcome: increased or unchanged distance from goal \
                              {self.desired_ee_goal.x, self.desired_ee_goal.y, self.desired_ee_goal.z}" + tcolors.ENDC)

            if self.step_counter == self.max_steps:
                if self.is_training:
                    self.score_history.append(reward)
                else:
                    self.goal_to_solve_idx += 1
                    if self.goal_to_solve_idx == len(self.goal_ee_pos["x"]):
                            self.goal_to_solve_idx = 0

        else:
            if self.action_result: # TODO in caso sostituire con un self.goal_reached (da aggiungere anche in init_env_variables() come False e come True in _is_done())
                rospy.loginfo(tcolors.CYAN + f"Action's outcome: the robot has reached the goal \
                              {self.desired_ee_goal.x, self.desired_ee_goal.y, self.desired_ee_goal.z}" + tcolors.ENDC)
                reward = self.reached_goal_reward

                if self.is_training:
                    self.score_history.append(reward)
                    if (self.queue_avg(self.score_history) >= .9 * self.end_episode_points):
                        self.score_history.clear()
                        self.goal_to_solve_idx += 1
                        if self.goal_to_solve_idx == len(self.goal_x_list):
                            self.goal_to_solve_idx = 0
                else:
                    self.solved_counter += 1
                    self.goal_to_solve_idx += 1
                    if self.goal_to_solve_idx == len(self.goal_x_list):
                            self.goal_to_solve_idx = 0
            else:
                rospy.loginfo(tcolors.CYAN + f"Action's outcome: the robot has failed this episode" + tcolors.ENDC)
                reward = self.impossible_movement_punishment

                if self.is_training:
                    self.score_history.append(reward)
                else:
                    self.goal_to_solve_idx += 1
                    if self.goal_to_solve_idx == len(self.goal_x_list):
                            self.goal_to_solve_idx = 0

        # Update the previous distance
        self.prev_dist_from_des_pos_ee = observations[-1]
        rospy.logdebug("Updated distance from GOAL = " + str(self.prev_dist_from_des_pos_ee))

        rospy.logdebug(">>> REWARD >>> " + str(reward))
        
        return reward

    def calculate_if_done(self, desired_ee_position, current_ee_pos):
        """
        It calculates whether the episode is over or not
        """
        done = False

        desired_ee_position = [desired_ee_position.x, desired_ee_position.y, desired_ee_position.z]
        current_ee_pos = [current_ee_pos.x, current_ee_pos.y, current_ee_pos.z]

        position_similar = np.all(np.isclose(desired_ee_position, current_ee_pos, atol=1e-02))

        if position_similar:
            done = True
            rospy.logdebug("Desired position reached!")
            
        return done

    def compute_euclidean_dist(self, p1, p2):
        """
        Calculates the Euclidean distance between two Point() inputs
        """
        v1 = [p1.x, p1.y, p1.z]
        v2 = [p2.x, p2.y, p2.z]       

        dist = np.linalg.norm(np.array(v1) - np.array(v2))
        return dist

    def queue_avg(self, queue):
        total = 0

        for elem in queue:
            total += elem

        return total / len(queue)
