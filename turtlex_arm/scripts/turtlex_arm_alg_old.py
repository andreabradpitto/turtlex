#!/usr/bin/env python3

import rospy
# Inspired by https://keon.io/deep-q-learning/
import random
import gym
import math
import rospkg
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import model_from_yaml
import time
from std_msgs.msg import Float32
from utils import tcolors

# import our training environment
import turtlex_arm_task #from openai_ros.task_envs.fetch import fetch_test_task


class DQNRobotSolver(): # TODO nel paper da Armando si dice che questa (DQN) presenta problemi, meglio la loro (DDPG)
    def __init__(self, environment_name, n_observations, n_actions, max_env_steps=None, gamma=1.0, epsilon=1.0,
                 epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64, monitor=False, outdir='../monitor_data'):

        self.memory = deque(maxlen=100000)
        self.env = gym.make(environment_name)
        if monitor: self.env = gym.wrappers.Monitor(self.env, outdir, force=True)

        self.input_dim = n_observations
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        self.model = Sequential()
        
        self.model.add(Dense(24, input_dim=self.input_dim, activation='tanh'))
        self.model.add(Dense(48, activation='tanh'))
        self.model.add(Dense(self.n_actions, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon, do_train, step=0):
        
        if do_train and (np.random.random() <= epsilon):
            # We return a random sample form the available action space
            rospy.loginfo(tcolors.MAGENTA + ">>>>> Chosen Random ACTION" + tcolors.ENDC)
            action_chosen = self.env.action_space.sample()
             
        else:
            # We return the best known prediction based on the state
            action_chosen = np.argmax(self.model.predict(state))
        
        if do_train:
            rospy.loginfo(tcolors.MAGENTA + "LEARNING Action=" + str(action_chosen) + ", Epsilon=" + str(round(epsilon, 3)) +
                          ", Step=" + str(step) + tcolors.ENDC)
        else:
            rospy.loginfo(tcolors.MAGENTA + "RUNNING Action=" + str(action_chosen) + ", Epsilon=" + str(round(epsilon, 3)) +
                          ", Step=" + str(step) + tcolors.ENDC)
        
        return action_chosen
        

    def get_epsilon(self, t):
        new_epsilon = max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))
        #new_epsilon = self.epsilon
        return new_epsilon

    def preprocess_state(self, state):
        return np.reshape(state, [1, self.input_dim])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self, num_episodes, do_train=False):

        start_time = time.time()

        rospy.logdebug(f"env.action_space.high: {self.env.action_space.high}")
        rospy.logdebug(f"env.action_space.low: {self.env.action_space.low}")
        rospy.logdebug(f"env.observation_space.high: {self.env.observation_space.high}")
        rospy.logdebug(f"env.observation_space.low: {self.env.observation_space.low}")

        highest_reward = 0

        for ep in range(1, num_episodes + 1):

            rospy.loginfo(tcolors.CYAN + "######################## Beginning episode => " + str(ep) + tcolors.ENDC)

            if monitor: self.env.stats_recorder.done = None

            init_state = self.env.reset()

            state = self.preprocess_state(init_state)
            done = False

            cumulated_ep_reward = 0

            for step in range(1, max_ep_steps + 1):

                rospy.loginfo(tcolors.CYAN + "############### Starting Step => " + str(step) + tcolors.ENDC)

                #self.env.render() # openai_ros does not support render for the moment
                action = self.choose_action(state, self.get_epsilon(ep), do_train, step)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                
                if do_train:
                    # If we are training, we want to remember what happened and process it
                    self.remember(state, action, reward, next_state, done)
                
                state = next_state

                cumulated_ep_reward += reward
                if highest_reward < cumulated_ep_reward:
                    highest_reward = cumulated_ep_reward

                rospy.loginfo(tcolors.CYAN + "# State used for the action       => [" + str(', '.join(map(str, state))) + "]" + tcolors.ENDC)
                #rospy.loginfo(tcolors.CYAN + "# Action performed                => [" + str(', '.join(map(str, action))) + "]" + tcolors.ENDC)
                rospy.loginfo(tcolors.CYAN + "# Action performed                => " + str(action) + tcolors.ENDC)
                rospy.loginfo(tcolors.CYAN + "# Reward that action generated    => " + str(reward) + tcolors.ENDC)
                rospy.loginfo(tcolors.CYAN + "# Cumulated episode reward        => " + str(cumulated_ep_reward) + tcolors.ENDC)
                rospy.loginfo(tcolors.CYAN + "# Starting state of the next step => [" + str(', '.join(map(str, next_state))) + "]" + tcolors.ENDC)

                if done:
                    rospy.loginfo(tcolors.CYAN + f"Episode {ep} done" + tcolors.ENDC)
                    rospy.loginfo(tcolors.CYAN + "############### END Step => " + str(step) + tcolors.ENDC)
                    break
                else:
                    rospy.loginfo(tcolors.CYAN + f"Episode {ep} NOT done" + tcolors.ENDC)
                    rospy.loginfo(tcolors.CYAN + "############### END Step => " + str(step) + tcolors.ENDC)

            m, s = divmod(int(time.time() - start_time), 60)
            h, m = divmod(m, 60)
            rospy.loginfo(tcolors.MAGENTA + "Episode: " + str(ep) + " | cumulated_ep_reward: " + str(cumulated_ep_reward) + " | highest_reward: " +
                          str(highest_reward) + " | Final step: " + str(step) + " | Time: %d:%02d:%02d" % (h, m, s) + "\n\n" + tcolors.ENDC)

            if do_train:
                self.replay(self.batch_size)
                if ep % 20 == 0:
                    self.save(ep, outdir)

        return ep
        
    def save(self, model_name, models_dir_path="/tmp"):
        """
        Saves the current model
        """
        
        model_name_yaml_format = model_name + ".yaml"
        model_name_HDF5_format = model_name + ".h5"
        
        model_name_yaml_format_path = os.path.join(models_dir_path, model_name_yaml_format)
        model_name_HDF5_format_path = os.path.join(models_dir_path, model_name_HDF5_format)
        
        # serialize model to YAML
        model_yaml = self.model.to_yaml()
        
        with open(model_name_yaml_format_path, "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5: http://www.h5py.org/
        self.model.save_weights(model_name_HDF5_format_path)
        rospy.loginfo('''

        ====================================
                Saved model to disk    
        ====================================

        ''')
        
    def load(self, model_name, models_dir_path="/tmp"):
        """
        Loads a previously saved model
        """
        
        model_name_yaml_format = model_name + ".yaml"
        model_name_HDF5_format = model_name + ".h5"
        
        model_name_yaml_format_path = os.path.join(models_dir_path, model_name_yaml_format)
        model_name_HDF5_format_path = os.path.join(models_dir_path, model_name_HDF5_format)
        
        # load YAML and create model
        yaml_file = open(model_name_yaml_format_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self.model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        self.model.load_weights(model_name_HDF5_format_path)
        rospy.loginfo('''

        ====================================
               Loaded model from disk    
        ====================================

        ''')



if __name__ == '__main__':

    rospy.init_node('turtlex_arm_algorithm', anonymous=True, log_level=rospy.INFO)
    # logging hierarchy: CRITICAL > ERROR > WARNING > INFO > DEBUG; the chosen log level outputs that levels and all the ones above it

    # qlearn parameters
    alpha = rospy.get_param('/turtlex_arm/alpha')
    alpha_decay = rospy.get_param('/turtlex_arm/alpha_decay')
    gamma =  rospy.get_param('/turtlex_arm/gamma')
    epsilon = rospy.get_param('/turtlex_arm/epsilon')
    epsilon_log_decay = rospy.get_param('/turtlex_arm/epsilon_decay')
    epsilon_min = rospy.get_param('/turtlex_arm/epsilon_min')

    batch_size = rospy.get_param('/turtlex_arm/batch_size')

    monitor = rospy.get_param('/turtlex_arm/monitor')

    n_actions = rospy.get_param('/turtlex_arm/n_actions')    
    n_observations = rospy.get_param('/turtlex_arm/n_observations')

    max_ep_steps = rospy.get_param('/turtlex_arm/max_episode_steps')

    is_training = rospy.get_param('/turtlex_arm/training')
    if is_training:
        episode_num = rospy.get_param('/turtlex_arm/episodes_training') # Get the number of episodes for training
    else:
        test_loops = rospy.get_param('/turtlex_arm/test_loops')
        episode_num = test_loops * len(rospy.get_param('/turtlex_arm/ee_goals/x')) # Get the number of episodes for testing

    # Get the world name
    world_name = rospy.get_param('/turtlex_arm/world_name')

    rospackage_name = "turtlex_arm"
    model_name = "turtlex_arm_dqn"
    environment_name = 'TurtlexArmTask-v0'

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    outdir = pkg_path + '/scripts/arm_dqn/' + world_name
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        rospy.loginfo("Created folder=" + str(outdir))

    agent = DQNRobotSolver(environment_name, n_observations, n_actions, max_ep_steps, gamma, epsilon, epsilon_min,
                           epsilon_log_decay, alpha, alpha_decay, batch_size, monitor, '../monitor_data')

    agent.run(num_episodes=episode_num, do_train=is_training)
    
    if is_training:
        agent.save(model_name, outdir)
