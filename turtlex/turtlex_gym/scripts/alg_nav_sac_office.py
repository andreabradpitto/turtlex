#!/usr/bin/env python3

# partly inspired by https://github.com/dranaju/project

import rospy
import random
import copy
import torch
import rospkg
import gym
import time
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
import torch.nn as nn
from torch.optim import Adam
from utils import tcolors
import rosnode
import task_nav_office  # task environment


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
        self.lr=lr

        self.target_update_interval = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(state_dim, action_dim, critic_hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        self.critic_target = QNetwork(state_dim, action_dim, critic_hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)
        
        self.target_entropy = - torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
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
            action = (2 * torch.sigmoid(2 * action)) - 1
        action = action.detach().cpu().numpy()[0]
        return action
    
    def update_parameters(self, memory, batch_size):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
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

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        soft_update(self.critic_target, self.critic, self.tau)
    
    # Save model parameters
    def save_models(self, path, episode_count):
        torch.save(self.policy, path + '/' + str(episode_count) + '_policy_net.pth')
        torch.save(self.critic, path + '/' + str(episode_count) + '_value_net.pth')
        rospy.loginfo('''

        ====================================
             The models have been saved     
        ====================================

        ''')
    
    # Load model parameters
    def load_models(self, path, episode):
        self.policy = torch.load(path + '/' + str(episode) + '_policy_net.pth', map_location=self.device)
        self.critic = torch.load(path + '/' + str(episode) + '_value_net.pth', map_location=self.device)
        hard_update(self.critic_target, self.critic)
        rospy.loginfo('''

        ====================================
             The models have been loaded    
        ====================================

        ''')



if __name__ == '__main__':

    rospy.init_node('turtlex_sac_office', anonymous=True, log_level=rospy.INFO)
    # logging hierarchy: CRITICAL > ERROR > WARNING > INFO > DEBUG; the chosen log level outputs that level and all the ones above it

    while('/spawn_turtlex_model' in rosnode.get_node_names()):
        pass

    # Create the Gym environment
    env = gym.make('TaskNavOffice-v0')
    rospy.loginfo("Gym environment created")

    is_training = env.is_training

    batch_size  = rospy.get_param("/turtlex_nav/batch_size")

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the "config" directory
    # They are loaded at runtime by the launch file
    state_dim  = rospy.get_param("/turtlex_nav/n_sectors") + 4
    actor_hidden_dim = rospy.get_param("/turtlex_nav/actor_hidden_dim")
    critic_hidden_dim = rospy.get_param("/turtlex_nav/critic_hidden_dim")
    action_dim = rospy.get_param("/turtlex_nav/n_actions")

    # SAC parameters
    gamma = rospy.get_param("/turtlex_nav/gamma")
    tau = rospy.get_param("/turtlex_nav/tau")
    alpha = rospy.get_param("/turtlex_nav/alpha")
    learning_rate = rospy.get_param("/turtlex_nav/learning_rate")

    max_episodes = rospy.get_param("/turtlex_nav/nepisodes")
    max_steps = rospy.get_param("/turtlex_nav/nsteps")

    monitor = rospy.get_param('/turtlex_nav/monitor')

    # Bounds for the actions (linear and angular velocities)
    action_v_min = rospy.get_param("/turtlex_nav/action_v_min")
    action_w_min = rospy.get_param("/turtlex_nav/action_w_min")
    action_v_max = rospy.get_param("/turtlex_nav/action_v_max")
    action_w_max = rospy.get_param("/turtlex_nav/action_w_max")

    # Get the world name
    world_name = rospy.get_param("/turtlex_nav/world_name")

    goal_reached_reward = rospy.get_param("/turtlex_nav/end_episode_points")

    load_model = rospy.get_param("/turtlex_nav/load_model")

    replay_buffer_size = rospy.get_param("/turtlex_nav/replay_buffer_size")

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('results')
    gym_outdir = pkg_path + '/gym/' + world_name + '_nav_sac'
    nets_outdir = pkg_path + '/nets_train/' + world_name + '_nav_sac'

    if monitor: env = gym.wrappers.Monitor(env, gym_outdir, force=True)

    # Create the agent and the replay buffer
    agent = SAC(state_dim, action_dim, gamma, tau, alpha, actor_hidden_dim, critic_hidden_dim, learning_rate)
    replay_buffer = ReplayBuffer(replay_buffer_size)
    if load_model != False:
        agent.load_models(nets_outdir, load_model)

    rospy.logdebug('State Dimension: ' + str(state_dim))
    rospy.logdebug('Action Dimension: ' + str(action_dim))
    rospy.logdebug('Action Minimums: ' + str(action_v_min) + ' m/s and ' + str(action_w_min) + ' rad/s')
    rospy.logdebug('Action Maximums: ' + str(action_v_max) + ' m/s and ' + str(action_w_max) + ' rad/s')

    before_training = 4

    rospy.logdebug(f"env.action_space.high: {env.action_space.high}")
    rospy.logdebug(f"env.action_space.low: {env.action_space.low}")
    rospy.logdebug(f"env.observation_space.high: {env.observation_space.high}")
    rospy.logdebug(f"env.observation_space.low: {env.observation_space.low}")

    highest_reward = 0

    if not is_training:
        max_episodes = env.testing_goals

    start_time = time.time()

    for ep in range(1, max_episodes + 1):

        rospy.loginfo(tcolors.CYAN + "######################## Beginning episode => " + str(ep) + tcolors.ENDC)

        if monitor: env.stats_recorder.done = None

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

            state = np.float32(state)

            #env.render()  # openai_ros does not support render for the moment

            if is_training and not ep % 10 == 0:
                action = agent.select_action(state)
            else:
                action = agent.select_action(state, eval=True)

            if not is_training:
                action = agent.select_action(state, eval=True)

            # De-normalize the action, so that env.step() gets passed the actual linear and angular velocities
            # Normalization was performed in [-1, 1]
            unnorm_action = np.array([action_unnormalized(action[0], action_v_max, action_v_min),
                                      action_unnormalized(action[1], action_w_max, action_w_min)])

            next_state, reward, done, info = env.step(unnorm_action)

            if highest_reward < env.overall_reward:
                highest_reward = env.overall_reward

            next_state = np.float32(next_state)

            #rospy.loginfo(tcolors.CYAN + "# State used for the action       => [" + str(', '.join(map(str, state))) + "]" + tcolors.ENDC)
            ##rospy.loginfo(tcolors.CYAN + "# Action performed (normalized)   => [" + str(', '.join(map(str, action))) + "]" + tcolors.ENDC)
            #rospy.loginfo(tcolors.CYAN + "# Action performed (unnormalized) => [" + str(', '.join(map(str, unnorm_action))) + "]" + tcolors.ENDC)
            rospy.loginfo(tcolors.CYAN + "# Reward that action generated    => " + str(reward) + tcolors.ENDC)
            rospy.loginfo(tcolors.CYAN + "# Cumulated episode reward        => " + str(env.episode_reward) + tcolors.ENDC)
            #rospy.loginfo(tcolors.CYAN + "# Overall reward                  => " + str(env.overall_reward) + tcolors.ENDC)
            #rospy.loginfo(tcolors.CYAN + "# Starting state of the next step => [" + str(', '.join(map(str, next_state))) + "]" + tcolors.ENDC)

            if not ep % 10 == 0 or not len(replay_buffer) > before_training * batch_size:
                if reward == goal_reached_reward:
                    for _ in range(3):
                        replay_buffer.push(state, action, reward, next_state, done)
                else:
                    replay_buffer.push(state, action, reward, next_state, done)
            
            if len(replay_buffer) > before_training * batch_size and is_training and not ep % 10 == 0:
                agent.update_parameters(replay_buffer, batch_size)

            state = copy.deepcopy(next_state)

            if done:
                rospy.loginfo(tcolors.CYAN + f"Episode {ep}: done" + tcolors.ENDC)

                break
            else:
                rospy.loginfo(tcolors.CYAN + f"Episode {ep}: NOT done" + tcolors.ENDC)

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.loginfo(tcolors.MAGENTA + "Episode: " + str(ep) + " | Overall reward: " + str(env.overall_reward) + " | highest_reward: " +
                        str(highest_reward) + " | Final step: " + str(step) + " | Time: %d:%02d:%02d" % (h, m, s) + "\n\n" + tcolors.ENDC)


        if ep % 20 == 0 and ep != max_episodes and is_training:
            agent.save_models(nets_outdir, ep)

    if not is_training:
        rospy.loginfo(f"\nTest results: {env.solved_counter} / {max_episodes}\n")
    else:
        agent.save_models(nets_outdir, max_episodes)

    env.close()  # Known issue: https://stackoverflow.com/questions/64679139
