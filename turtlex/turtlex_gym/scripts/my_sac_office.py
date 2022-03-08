#!/usr/bin/env python3

import rospy
import numpy as np
import random
import copy
#from std_msgs.msg import Float32
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import torch.nn as nn
from torch.optim import Adam

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
#---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))

import gym
import time
#import functools
#from algs import qlearn # training algorithm
#from task_envs import turtlex_office # training environment
from gym import wrappers
#import rospkg
import turtlex_office # training environment
from utils import tcolors

#import matplotlib.pyplot as plt
#import seaborn as sns
#import pandas as pd


# The replay buffer is the agent's memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def action_unnormalized(action, high, low):
    action = low + (action + 1.0) * 0.5 * (high - low)
    #action = np.clip(action, low, high)
    return action

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, critic_hidden_dim):

        super(QNetwork, self).__init__()
        
        # Q1
        self.linear1_q1 = nn.Linear(state_dim + action_dim, critic_hidden_dim)
        self.linear2_q1 = nn.Linear(critic_hidden_dim, critic_hidden_dim)
        self.linear3_q1 = nn.Linear(critic_hidden_dim, critic_hidden_dim)
        self.linear4_q1 = nn.Linear(critic_hidden_dim, 1)
        
        # Q2
        self.linear1_q2 = nn.Linear(state_dim + action_dim, critic_hidden_dim)
        self.linear2_q2 = nn.Linear(critic_hidden_dim, critic_hidden_dim)
        self.linear3_q2 = nn.Linear(critic_hidden_dim, critic_hidden_dim)
        self.linear4_q2 = nn.Linear(critic_hidden_dim, 1)
        
        self.apply(weights_init_)
        
    def forward(self, state, action):
        x_state_action = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1_q1(x_state_action))
        x1 = F.relu(self.linear2_q1(x1))
        x1 = F.relu(self.linear3_q1(x1))
        x1 = self.linear4_q1(x1)
        
        x2 = F.relu(self.linear1_q2(x_state_action))
        x2 = F.relu(self.linear2_q2(x2))
        x2 = F.relu(self.linear3_q2(x2))
        x2 = self.linear4_q2(x2)
        
        return x1, x2

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, actor_hidden_dim, log_std_min=-20, log_std_max=2):

        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(state_dim, actor_hidden_dim)
        self.linear2 = nn.Linear(actor_hidden_dim, actor_hidden_dim)

        self.mean_linear = nn.Linear(actor_hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(actor_hidden_dim, action_dim)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        #action = torch.tanh(x_t)
        action = (2 * torch.sigmoid(2 * x_t)) - 1
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std

class SAC(object):
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=1e-2, alpha=0.2, actor_hidden_dim=256, critic_hidden_dim=256, lr=0.0003):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        # self.action_range = [action_space.low, action_space.high]
        self.lr=lr

        self.target_update_interval = 1
        #self.automatic_entropy_tuning = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(state_dim, action_dim, critic_hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        self.critic_target = QNetwork(state_dim, action_dim, critic_hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        #self.value = ValueNetwork(state_dim, value_hidden_dim).to(device=self.device)
        #self.value_target = ValueNetwork(state_dim, value_hidden_dim).to(self.device)
        #self.value_optim = Adam(self.value.parameters(), lr=self.lr)
        #hard_update(self.value_target, self.value)
        
        self.target_entropy = - torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        #print('entropy', self.target_entropy)
        rospy.loginfo('Entropy: ' + str(self.target_entropy))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

        self.policy = PolicyNetwork(state_dim, action_dim, actor_hidden_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        
        
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
            #action = torch.tanh(action)
            action = (2 * torch.sigmoid(2 * action)) - 1
        action = action.detach().cpu().numpy()[0]
        return action
    
    # def rescale_action(self, action):
    #     return action * (self.action_range[1] - self.action_range[0]) / 2.0 +\
    #             (self.action_range[1] + self.action_range[0]) / 2.0
    
    def update_parameters(self, memory, batch_size):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            #vf_next_target = self.value_target(next_state_batch)
            #next_q_value = reward_batch + (1 - done_batch) * self.gamma * (vf_next_target)
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _, _ = self.policy.sample(state_batch) # pi, log_pi, mean, log_std = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 
        # Regularization Loss
        #reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
        #policy_loss += reg_loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        #vf = self.value(state_batch)
        
        #with torch.no_grad():
        #    vf_target = min_qf_pi - (self.alpha * log_pi)

        #vf_loss = F.mse_loss(vf, vf_target) # 

        #self.value_optim.zero_grad()
        #vf_loss.backward()
        #self.value_optim.step()
        
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        #alpha_tlogs = self.alpha.clone() # For TensorboardX logs

        #if updates % self.target_update_interval == 0:
        soft_update(self.critic_target, self.critic, self.tau)

        #return vf_loss.item(), qf1_loss.item(), qf2_loss.item(), policy_loss.item()
    
    # Save model parameters
    def save_models(self, episode_count, world_name):
        torch.save(self.policy, dirPath + '/nav_sac/' + world_name + '/' + str(episode_count) + '_policy_net.pth')
        torch.save(self.critic, dirPath + '/nav_sac/' + world_name + '/' + str(episode_count) + '_value_net.pth')
        # hard_update(self.critic_target, self.critic)
        # torch.save(soft_q_net.state_dict(), dirPath + '/nav_sac/' + world_name + '/' + str(episode_count) + 'soft_q_net.pth')
        # torch.save(target_value_net.state_dict(), dirPath + '/nav_sac/' + world_name + '/' + str(episode_count) + 'target_value_net.pth')
        rospy.loginfo('''

        ====================================
             The models have been saved     
        ====================================

        ''')
    
    # Load model parameters
    def load_models(self, episode, world_name):
        self.policy = torch.load(dirPath + '/nav_sac/' + world_name + '/' + str(episode) + '_policy_net.pth', map_location=self.device)
        self.critic = torch.load(dirPath + '/nav_sac/' + world_name + '/' + str(episode) + '_value_net.pth', map_location=self.device)
        hard_update(self.critic_target, self.critic)
        # soft_q_net.load_state_dict(torch.load(dirPath + '/nav_sac/' + world_name + '/'+str(episode) + 'soft_q_net.pth'))
        # target_value_net.load_state_dict(torch.load(dirPath + '/nav_sac/' + world_name + '/'+str(episode) + 'target_value_net.pth'))
        rospy.loginfo('''

        ====================================
             The models have been loaded    
        ====================================

        ''')



if __name__ == '__main__':

    rospy.init_node('turtlex_sac_office', anonymous=True, log_level=rospy.INFO) # change from rospy.WARN to rospy.DEBUG to view all the prints
    # logging hierarchy: CRITICAL > ERROR > WARNING > INFO > DEBUG; the chosen log level outputs that level and all the ones above it

    # Create the Gym environment
    env = gym.make('MyTurtlexOffice-v0')
    rospy.loginfo("Gym environment created")

    #is_training = rospy.get_param("/turtlex/training")
    is_training = env.is_training

    batch_size  = rospy.get_param("/turtlex/batch_size")

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the "config" directory
    # They are loaded at runtime by the launch file
    state_dim  = rospy.get_param("/turtlex/n_sectors") + 4
    actor_hidden_dim = rospy.get_param("/turtlex/actor_hidden_dim")
    critic_hidden_dim = rospy.get_param("/turtlex/critic_hidden_dim")
    action_dim = rospy.get_param("/turtlex/n_actions")

    # SAC parameters
    gamma = rospy.get_param("/turtlex/gamma")
    tau = rospy.get_param("/turtlex/tau")
    alpha = rospy.get_param("/turtlex/alpha")
    learning_rate = rospy.get_param("/turtlex/learning_rate")

    max_episodes = rospy.get_param("/turtlex/nepisodes")
    max_steps = rospy.get_param("/turtlex/nsteps")

    # Bounds for the actions (linear and angular velocities)
    action_v_min = rospy.get_param("/turtlex/action_v_min")
    action_w_min = rospy.get_param("/turtlex/action_w_min")
    action_v_max = rospy.get_param("/turtlex/action_v_max")
    action_w_max = rospy.get_param("/turtlex/action_w_max")

    # Get the world name
    world_name = rospy.get_param("/turtlex/world_name")

    goal_reached_reward = rospy.get_param("/turtlex/end_episode_points")

    load_model = rospy.get_param("/turtlex/load_model")

    replay_buffer_size = rospy.get_param("/turtlex/replay_buffer_size")

    # Create the agent and the replay buffer
    agent = SAC(state_dim, action_dim, gamma, tau, alpha, actor_hidden_dim, critic_hidden_dim, learning_rate)
    replay_buffer = ReplayBuffer(replay_buffer_size)
    if load_model != False:
        agent.load_models(load_model, world_name)

    rospy.loginfo('State Dimension: ' + str(state_dim))
    rospy.loginfo('Action Dimension: ' + str(action_dim))
    rospy.loginfo('Action Minimums: ' + str(action_v_min) + ' m/s and ' + str(action_w_min) + ' rad/s')
    rospy.loginfo('Action Maximums: ' + str(action_v_max) + ' m/s and ' + str(action_w_max) + ' rad/s')

    # Set the logging system
    #rospack = rospkg.RosPack()
    #pkg_path = rospack.get_path('turtlex_gym')
    #outdir = pkg_path + '/scripts/nav_sac/' + world_name + '/sac_training_results'
    outdir = dirPath + '/nav_sac/' + world_name + '/sac_training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    #pub_result = rospy.Publisher('/result', Float32, queue_size=5)
    #result = Float32()

    before_training = 4

    rospy.logdebug(f"env.action_space.high: {env.action_space.high}")
    rospy.logdebug(f"env.action_space.low: {env.action_space.low}")
    rospy.logdebug(f"env.observation_space.high: {env.observation_space.high}")
    rospy.logdebug(f"env.observation_space.low: {env.observation_space.low}")

    highest_reward = 0

    """
    logger = dict(episode=[],reward=[])
    plt.ion()
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    """

    if not is_training:
        max_episodes = env.testing_goals

    start_time = time.time()

    for ep in range(1, max_episodes + 1):

        rospy.loginfo(tcolors.CYAN + "######################## Beginning episode => " + str(ep) + tcolors.ENDC)

        env.stats_recorder.done = None # CORRETTO DA ME per bug Monitor; andava bene anche "env.stats_recorder.steps = 0", ma sarebbe stato meno corretto

        #done = False
        state = env.reset()

        if is_training and not ep % 10 == 0 and len(replay_buffer) > before_training * batch_size:
            rospy.loginfo(tcolors.CYAN + 'Episode ' + str(ep) + ': TRAINING' + tcolors.ENDC)
        else:
            if len(replay_buffer) > before_training * batch_size:
                rospy.loginfo(tcolors.CYAN + 'Episode ' + str(ep) + ': EVALUATING' + tcolors.ENDC)
            else:
                rospy.loginfo(tcolors.CYAN + 'Episode ' + str(ep) + ': ADDING TO MEMORY' + tcolors.ENDC)

        for step in range(1, max_steps + 1):

            rospy.loginfo(tcolors.CYAN + "############### Starting Step => " + str(step) + tcolors.ENDC)

            state = np.float32(state) # attento qui, sotto, unnorm_action, ed a return np.asarray(observations) in _get_obs() + 2 in _init_env_variables

            if is_training and not ep % 10 == 0:
                action = agent.select_action(state)
            else:
                action = agent.select_action(state, eval=True)

            if not is_training:
                action = agent.select_action(state, eval=True)

            # De-normalize the action, so that env.step() gets passed the actual linear and angular velocities
            # Normalization was performed in [-1, 1]
            #unnorm_action = [action_unnormalized(action[0], action_v_max, action_v_min), action_unnormalized(action[1], action_w_max, action_w_min)]
            unnorm_action = np.array([action_unnormalized(action[0], action_v_max, action_v_min),
                                      action_unnormalized(action[1], action_w_max, action_w_min)])

            next_state, reward, done, info = env.step(unnorm_action)

            if highest_reward < env.overall_reward:
                highest_reward = env.overall_reward

            next_state = np.float32(next_state) # attento qui, sopra, unnorm_action, ed a return np.asarray(observations) in _get_obs() + 2 in _init_env_variables

            #rospy.loginfo(tcolors.CYAN + "# State used for the action       => [" + str(', '.join(map(str, state))) + "]" + tcolors.ENDC)
            ##rospy.loginfo(tcolors.CYAN + "# Action performed (normalized)   => [" + str(', '.join(map(str, action))) + "]" + tcolors.ENDC)
            #rospy.loginfo(tcolors.CYAN + "# Action performed (unnormalized) => [" + str(', '.join(map(str, unnorm_action))) + "]" + tcolors.ENDC)
            rospy.loginfo(tcolors.CYAN + "# Reward that action generated    => " + str(reward) + tcolors.ENDC)
            rospy.loginfo(tcolors.CYAN + "# Cumulated episode reward        => " + str(env.episode_reward) + tcolors.ENDC)
            #rospy.loginfo(tcolors.CYAN + "# Overall reward                  => " + str(env.overall_reward) + tcolors.ENDC)
            #rospy.loginfo(tcolors.CYAN + "# Starting state of the next step => [" + str(', '.join(map(str, next_state))) + "]" + tcolors.ENDC)

            if not ep % 10 == 0 or not len(replay_buffer) > before_training * batch_size:
                if reward == goal_reached_reward:
                    rospy.loginfo(tcolors.MAGENTA + '\n\n\n\n\t\t\t-------- GOAL REACHED ----------\n\n\n' + tcolors.ENDC)
                    for _ in range(3):
                        replay_buffer.push(state, action, reward, next_state, done)
                else:
                    replay_buffer.push(state, action, reward, next_state, done)
            
            if len(replay_buffer) > before_training * batch_size and is_training and not ep % 10 == 0:
                agent.update_parameters(replay_buffer, batch_size)

            state = copy.deepcopy(next_state)

            if done:
                rospy.loginfo(tcolors.CYAN + f"Episode {ep}: done" + tcolors.ENDC)
                #rospy.loginfo(tcolors.CYAN + "############### END Step => " + str(step) + tcolors.ENDC)

                """
                # save reward logs
                ax1.cla()
                logger['episode'] = range(1, max_episodes + 1)
                logger['reward'].append(reward)
                df = pd.DataFrame(logger)
                sns.lineplot(ax=ax1, x='episode', y='reward', data=df)
                """

                break
            else:
                rospy.loginfo(tcolors.CYAN + f"Episode {ep}: NOT done" + tcolors.ENDC)
                #rospy.loginfo(tcolors.CYAN + "############### END Step => " + str(step) + tcolors.ENDC)

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)

        #rospy.loginfo(tcolors.MAGENTA + "\nEpisode: " + str(ep) + " | Reward: " + str(env.episode_reward) + " | Final step: " + str(step) +
        #              " | Avg. reward: " + str(env.episode_reward / step) + " | Time: %d:%02d:%02d" % (h, m, s) + "\n" + tcolors.ENDC)
        rospy.loginfo(tcolors.MAGENTA + "Episode: " + str(ep) + " | Overall reward: " + str(env.overall_reward) + " | Final step: " + str(step) +
                      " | Time: %d:%02d:%02d" % (h, m, s) + "\n\n" + tcolors.ENDC)

        """
        if ep % 10 == 0:
            if len(replay_buffer) > before_training * batch_size:
                #result = env.episode_reward
                result = env.overall_reward
                pub_result.publish(result)
        """

        #pub_result.publish(env.queue_avg(env.score_history))

        if ep % 20 == 0 and is_training:
            agent.save_models(ep, world_name)

    #if highest_reward < env.episode_reward:
    #    highest_reward = env.episode_reward

    if not is_training:
        rospy.loginfo(f"\nTest results: {env.solved_counter} / {max_episodes}\n")

    #agent.save_models(max_episodes - 1, world_name)
    rospy.loginfo(tcolors.CYAN + "\n\n| gamma: " + str(gamma) + "| tau: " + str(tau) + "| alpha: " + str(alpha) + "| learning_rate: " +
                    str(learning_rate) + "| max_episodes: " + str(max_episodes) + "| highest_reward: " + str(highest_reward) + "\n\n" + tcolors.ENDC)

    env.close() # https://stackoverflow.com/questions/64679139
