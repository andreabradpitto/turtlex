import numpy as np
import tensorflow as tf
from keras.initializers import GlorotNormal
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, BatchNormalization, Activation, Lambda, Concatenate
from keras.regularizers import l2


class ActorNet():
	"""
	Actor Network for DDPG
	"""
	def __init__(self, in_dim, out_dim, act_range, lr_, tau_):
		self.obs_dim = in_dim
		self.act_dim = out_dim
		self.act_range = act_range
		self.lr = lr_; self.tau = tau_

		# Initialize actor network and target
		self.network = self.create_network()
		self.target_network = self.create_network()

		# Initialize optimizer
		self.optimizer = Adam(self.lr)

		# Copy the weights for initialization
		weights_ = self.network.get_weights()
		self.target_network.set_weights(weights_)


	def create_network(self):
		"""
		Create a Actor Network Model using Keras
		"""
		# Unput layer (observations)
		input_ = Input(shape=self.obs_dim)

		# Gidden layer 1
		h1_ = Dense(30, kernel_initializer=GlorotNormal())(input_)
		h1_b = BatchNormalization()(h1_)
		h1 = Activation('relu')(h1_b)

		# Hidden_layer 2
		h2_ = Dense(30, kernel_initializer=GlorotNormal())(h1)
		h2_b = BatchNormalization()(h2_)
		h2 = Activation('relu')(h2_b)

		# Output layer (actions)
		output_ = Dense(self.act_dim, kernel_initializer=GlorotNormal())(h2)
		output_b = BatchNormalization()(output_)
		output = Activation('sigmoid')(2 * output_b)
		output = (2 * output) - 1
		scalar = self.act_range * np.ones(self.act_dim)
		out = Lambda(lambda i: i * scalar)(output)

		return Model(input_, out)

	def train(self, obs, critic, q_grads):
		"""
		Training Actor's Weights
		"""
		with tf.GradientTape() as tape:
			actions = self.network(obs)
			actor_loss = - tf.reduce_mean(critic([obs,actions]))
		actor_grad = tape.gradient(actor_loss,self.network.trainable_variables)
		self.optimizer.apply_gradients(zip(actor_grad,self.network.trainable_variables))

	def target_update(self):
		"""
		Soft target update for training target actor network
		"""
		weights, weights_t = self.network.get_weights(), self.target_network.get_weights()
		for i in range(len(weights)):
			weights_t[i] = self.tau * weights[i] + (1 - self.tau) * weights_t[i]
		self.target_network.set_weights(weights_t)

	def predict(self, obs):
		"""
		Predict function for Actor Network
		"""
		return self.network.predict(np.expand_dims(obs, axis=0))

	def target_predict(self, new_obs):
		"""
		Predict function for Target Actor Network
		"""
		return self.target_network.predict(new_obs)

	def save_network(self, path, ep):
		self.network.save_weights(path + '/actor_' + str(ep) + '.h5')
		self.target_network.save_weights(path + '/actor_t_' + str(ep) + '.h5')

	def load_network(self, path, ep):
		self.network.load_weights(path + '/actor_' + str(ep) + '.h5')
		self.target_network.load_weights(path + '/actor_t_' + str(ep) + '.h5')
		print(self.network.summary())


class CriticNet():
	"""
	Critic Network for DDPG
	"""
	def __init__(self, in_dim, out_dim, lr_, tau_, discount_factor):
		self.obs_dim = in_dim
		self.act_dim = out_dim
		self.lr = lr_; self.discount_factor=discount_factor;self.tau = tau_

		# Initialize critic network and target
		self.network = self.create_network()
		self.target_network = self.create_network()

		self.optimizer = Adam(self.lr)

		# Copy the weights for initialization
		weights_ = self.network.get_weights()
		self.target_network.set_weights(weights_)

		self.critic_loss = None

	def create_network(self):
		"""
		Create a Critic Network Model using Keras
		as a Q-value approximator function
		"""
		# Input layer (observations and actions)
		input_obs = Input(shape=self.obs_dim)
		input_act = Input(shape=(self.act_dim,))
		inputs = [input_obs, input_act]
		concat = Concatenate(axis=-1)(inputs)

		# Hidden layer 1
		h1_ = Dense(300, kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01))(concat)
		h1_b = BatchNormalization()(h1_)
		h1 = Activation('relu')(h1_b)

		# Hidden_layer 2
		h2_ = Dense(400, kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01))(h1)
		h2_b = BatchNormalization()(h2_)
		h2 = Activation('relu')(h2_b)

		# Output layer (actions)
		output_ = Dense(1, kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01))(h2)
		output_b = BatchNormalization()(output_)
		output = Activation('linear')(output_b)

		return Model(inputs, output)

	def Qgradient(self, obs, acts):
		acts = tf.convert_to_tensor(acts)
		with tf.GradientTape() as tape:
			tape.watch(acts)
			q_values = self.network([obs,acts])
			q_values = tf.squeeze(q_values)
		return tape.gradient(q_values, acts)

	def train(self, obs, acts, target):
		"""
		Train Q-network for critic on sampled batch
		"""
		with tf.GradientTape() as tape:
			q_values = self.network([obs, acts], training=True)
			td_error = q_values - target
			critic_loss = tf.reduce_mean(tf.math.square(td_error))
			tf.print("critic loss :",critic_loss)
			self.critic_loss = float(critic_loss)

		critic_grad = tape.gradient(critic_loss, self.network.trainable_variables)  # Compute critic gradient
		self.optimizer.apply_gradients(zip(critic_grad, self.network.trainable_variables))

	def predict(self, obs):
		"""
		Predict Q-value from approximation function(Q-network)
		"""
		return self.network.predict(obs)

	def target_predict(self, new_obs):
		"""
		Predict target Q-value from approximation function(Q-network)
		"""
		return self.target_network.predict(new_obs)

	def target_update(self):
		"""
		Soft target update for training target critic network
		"""
		weights, weights_t = self.network.get_weights(), self.target_network.get_weights()
		for i in range(len(weights)):
			weights_t[i] = self.tau*weights[i] + (1-self.tau)*weights_t[i]
		self.target_network.set_weights(weights_t)

	def save_network(self, path, ep):
		self.network.save_weights(path + '/critic_' + str(ep) + '.h5')
		self.target_network.save_weights(path + '/critic_t_' + str(ep) + '.h5')

	def load_network(self, path, ep):
		self.network.load_weights(path + '/critic_' + str(ep) + '.h5')
		self.target_network.load_weights(path + '/critic_t_' + str(ep) + '.h5')
		print(self.network.summary())
