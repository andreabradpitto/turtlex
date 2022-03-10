#!/usr/bin/env python3

# Party inspired by https://github.com/CUN-bjy/gym-ddpg-keras

import rospy
import gym
import rospkg
import os
import time
import numpy as np
from ddpg import ActorNet, CriticNet
from ddpg_utils import MemoryBuffer, OrnsteinUhlenbeckProcess
from utils import tcolors
import rosnode
import task_arm_office  # import the training environment


class ddpgAgent():
    """
    Deep Deterministic Policy Gradient(DDPG) Agent
    """
    def __init__(self, env_, is_discrete=False, batch_size=100, w_per=True, buffer_size=20000, outdir='./ddpg_data'):

        # gym environments
        self.env = env_
        self.discrete = is_discrete
        self.obs_dim = env_.observation_space.shape[0]
        self.act_dim = env_.action_space.n if is_discrete else env_.action_space.shape[0]

        self.outdir = outdir

        self.action_bound = (env_.action_space.high - env_.action_space.low) / 2 if not is_discrete else 1.
        self.action_shift = (env_.action_space.high + env_.action_space.low) / 2 if not is_discrete else 0.

        # initialize actor & critic and their targets
        self.discount_factor = 0.99
        self.actor = ActorNet(self.obs_dim, self.act_dim, self.action_bound, lr_=1e-4, tau_=1e-3)
        self.critic = CriticNet(self.obs_dim, self.act_dim, lr_=1e-3, tau_=1e-3, discount_factor=self.discount_factor)

        # Experience Buffer
        self.buffer = MemoryBuffer(buffer_size, with_per=w_per)
        self.with_per = w_per
        self.batch_size = batch_size

        # OU-Noise-Process
        self.noise = OrnsteinUhlenbeckProcess(size=self.act_dim)

    ###################################################
    # Network Related
    ###################################################
    def make_action(self, obs, t, noise=True):
        """ predict next action from Actor's Policy
        """
        action_ = self.actor.predict(obs)[0]
        a = np.clip(action_ + self.noise.generate(t) if noise else 0, -self.action_bound, self.action_bound)
        return a

    def update_networks(self, obs, acts, critic_target):
        """ Train actor & critic from sampled experience
        """

        self.critic.train(obs, acts, critic_target)  # Update the critic

        # get next action and Q-value Gradient
        n_actions = self.actor.network.predict(obs)
        q_grads = self.critic.Qgradient(obs, n_actions)

        # update actor
        self.actor.train(obs,self.critic.network, q_grads)

        # update target networks
        self.actor.target_update()
        self.critic.target_update()

    def replay(self, replay_num_):
        if self.with_per and (self.buffer.size() <= self.batch_size): return

        for _ in range(replay_num_):

            # sample from buffer
            states, actions, rewards, dones, new_states, idx = self.sample_batch(self.batch_size)

            new_states = new_states.tolist() # Fix to "ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type list)"

            # get target q-value using target network
            q_vals = self.critic.target_predict([new_states, self.actor.target_predict(new_states)])

            # bellman iteration for target critic value
            critic_target = np.asarray(q_vals)
            for i in range(q_vals.shape[0]):
                if dones[i]:
                    critic_target[i] = rewards[i]
                else:
                    critic_target[i] = self.discount_factor * q_vals[i] + rewards[i]

                if self.with_per:
                    self.buffer.update(idx[i], abs(q_vals[i] - critic_target[i]))

            # train(or update) the actor & critic and target networks
            self.update_networks(states, actions, critic_target)


    ####################################################
    # Buffer Related
    ####################################################

    def memorize(self,obs,act,reward,done,new_obs):
        """store experience in the buffer
        """
        if self.with_per:
            q_val = self.critic.network([np.expand_dims(obs, axis=0), self.actor.predict(obs)])
            next_action = self.actor.target_network.predict(np.expand_dims(new_obs, axis=0))
            q_val_t = self.critic.target_predict([np.expand_dims(new_obs, axis=0), next_action])
            new_val = reward + self.discount_factor * q_val_t
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0			

        self.buffer.memorize(obs,act,reward, done,new_obs, td_error)

    def sample_batch(self, batch_size):
        """ Sampling from the batch
        """
        return self.buffer.sample_batch(batch_size)

    ###################################################
    # Save & Load Networks
    ###################################################
    def save_weights(self, episode):
        """ Agent's Weights Saver
        """
        self.actor.save_network(self.outdir, episode)
        self.critic.save_network(self.outdir, episode)
        rospy.loginfo('''

        ====================================
                Saved model to disk    
        ====================================

        ''')

    def load_weights(self, episode):
        """ Agent's Weights Loader
        """
        self.actor.load_network(self.outdir, episode)
        self.critic.load_network(self.outdir, episode)
        rospy.loginfo('''

        ====================================
               Loaded model from disk    
        ====================================

        ''')



if __name__ == '__main__':

    rospy.init_node('turtlex_arm_algorithm', anonymous=True, log_level=rospy.DEBUG)
    # logging hierarchy: CRITICAL > ERROR > WARNING > INFO > DEBUG; the chosen log level outputs that levels and all the ones above it

    while('/spawn_turtlex_model' in rosnode.get_node_names()):
        pass

    batch_size = rospy.get_param('/turtlex_arm/batch_size')
    buffer_size = rospy.get_param('/turtlex_arm/replay_buffer_size')

    monitor = rospy.get_param('/turtlex_arm/monitor')

    max_ep_steps = rospy.get_param('/turtlex_arm/max_episode_steps')

    load_model = rospy.get_param("/turtlex_arm/load_model")

    n_actions = rospy.get_param('/turtlex_arm/n_actions')
    n_observations = rospy.get_param('/turtlex_arm/n_observations')

    is_training = rospy.get_param('/turtlex_arm/training')
    if is_training:
        episode_num = rospy.get_param('/turtlex_arm/episodes_training') # Get the number of episodes for training
    else:
        test_loops = rospy.get_param('/turtlex_arm/test_loops')
        episode_num = test_loops * len(rospy.get_param('/turtlex_arm/ee_goals/x')) # Get the number of episodes for testing

    # Get the world name
    world_name = rospy.get_param('/turtlex_arm/world_name')

    rospackage_name = "turtlex_gym"
    environment_name = 'TaskArmOffice-v0'

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    outdir = pkg_path + '/training_results/' + world_name + '_arm_ddpg'

    if not os.path.exists(outdir):
        os.makedirs(outdir)
        rospy.loginfo("Created folder=" + str(outdir))

    env = gym.make(environment_name)
    if monitor: env = gym.wrappers.Monitor(env, outdir, force=True)

    try:
        # Ensure that the action bound is symmetric
        assert (np.all(env.action_space.high + env.action_space.low) == 0)
        is_discrete = False
        rospy.loginfo('Continuous Action Space')
    except AttributeError:
        is_discrete = True
        rospy.logerr('Discrete Action Space')

    # Create Agent model
    agent = ddpgAgent(env, batch_size=batch_size, w_per=False, is_discrete=is_discrete, buffer_size=buffer_size, outdir=outdir)

    if load_model != False:
        agent.load_weights(load_model)

    rospy.logdebug('State Dimension: ' + str(n_actions))
    rospy.logdebug('Action Dimension: ' + str(n_observations))

    rospy.logdebug(f"env.action_space.high: {env.action_space.high}")
    rospy.logdebug(f"env.action_space.low: {env.action_space.low}")
    rospy.logdebug(f"env.observation_space.high: {env.observation_space.high}")
    rospy.logdebug(f"env.observation_space.low: {env.observation_space.low}")

    highest_reward = 0

    start_time = time.time()

    for ep in range(1, episode_num + 1):

        rospy.loginfo(tcolors.CYAN + "######################## Beginning episode => " + str(ep) + tcolors.ENDC)

        if monitor: env.stats_recorder.done = None

        #done = False
        state = env.reset()

        cumulated_ep_reward = 0

        for step in range(1, max_ep_steps + 1):

            rospy.loginfo(tcolors.CYAN + "############### Starting Step => " + str(step) + tcolors.ENDC)

            #env.render()  # openai_ros does not support render for the moment
            action = agent.make_action(state, step)
            next_state, reward, done, _ = env.step(action)
            
            if is_training:
                agent.memorize(state, action, reward, done, next_state)  # store the results into buffer

            state = next_state

            cumulated_ep_reward += reward
            if highest_reward < cumulated_ep_reward:
                highest_reward = cumulated_ep_reward

            if is_training:
                agent.replay(1)

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

        if is_training and ep % 20 == 0 and ep != episode_num:
            agent.save_weights(ep)

    if not is_training:
        rospy.loginfo(f"\nTest results: {env.solved_counter} / {episode_num}\n")
    else:
        agent.save_weights(episode_num)

    env.close()  # Known issue: https://stackoverflow.com/questions/64679139
