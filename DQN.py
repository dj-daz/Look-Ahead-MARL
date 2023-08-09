from collections import deque
import time
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
import random

class DQNAgent:
	def __init__(self, size, action_size, agentId, ability=0, LOADMODEL = "", MODEL_NAME = ""):
		self.SIZE = size
		self.ACTION_SIZE = action_size
		self.ability = ability

		self.LOAD_MODEL = LOADMODEL
		model_name = MODEL_NAME + '-' + agentId

		self.DISCOUNT = 0.99
		self.REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
		self.MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
		self.MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
		self.UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

		# create main model
		self.model = self.create_model()

		# create target model
		self.target_model = self.create_model()
		self.target_model.set_weights(self.model.get_weights())
		self.target_update_counter = 0 # counter to keep track of target model updates

		 # create array with last n samples for training
		self.replay_memory = deque(maxlen = self.REPLAY_MEMORY_SIZE)

		# custom tensorboard object
		self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{model_name}-{int(time.time())}")

	def create_model(self):

		if self.LOAD_MODEL is not "":
			print(f"Loading {self.LOAD_MODEL}")
			model = load_model(self.LOAD_MODEL)
			print(f"Model {self.LOAD_MODEL} loaded!")

		else:
			model = Sequential()
			model.add(Conv2D(256, (3,3), input_shape = (self.SIZE, self.SIZE, 1)))
			model.add(Activation("relu"))
			model.add(MaxPooling2D(2,2))
			model.add(Dropout(0.2))

			model.add(Conv2D(256, (3,3)))
			model.add(Activation("relu"))
			model.add(MaxPooling2D(2,2))
			model.add(Dropout(0.2))

			model.add(Flatten())
			model.add(Dense(64))

			model.add(Dense(self.ACTION_SIZE, activation = "linear"))
			model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

		return model

	def update_replay_memory(self, transition):
	# Adds step's data to a memory replay array
	# transition: (observation space, action, reward, new observation space, done)
		self.replay_memory.append(transition)

	def get_qs(self, state):
		# predict Q value from model
		return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

	def train(self, terminal_state, step):
	# transition: (observation space, action, reward, new observation space, done)

		tic = time.perf_counter()

		# Start training only if certain number of samples is already saved
		if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
			return

		# Get a minibatch of random samples from memory replay table
		minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

		# Get current states from minibatch, then query NN model for Q values
		# transition[0]: state
		current_states = np.array([transition[0] for transition in minibatch])/255
		current_qs_list = self.model.predict(current_states)

		# Get future states from minibatch, then query NN target model for Q values
		# transition[3]: new states
		new_current_states = np.array([transition[3] for transition in minibatch])/255
		future_qs_list = self.target_model.predict(new_current_states)

		States = [] # states
		Qs = [] # stores q values

		# update Q values
		for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
			if not done:
				max_future_q = np.max(future_qs_list[index])
				new_q = reward + self.DISCOUNT * max_future_q # Q value calculation
			else:
				new_q = reward

			current_qs = current_qs_list[index]
			current_qs[action] = new_q # Q value update

			# store updated states and q values
			States.append(current_state)
			Qs.append(current_qs)

		# update model
		self.model.fit(np.array(States)/255, np.array(Qs), batch_size = self.MINIBATCH_SIZE, verbose = 0, shuffle = False, callbacks = [self.tensorboard] if terminal_state else None)

		#update to determine if we want to update target model yet
		if terminal_state:
			self.target_update_counter += 1

		if self.target_update_counter > self.UPDATE_TARGET_EVERY:
			self.target_model.set_weights(self.model.get_weights())
			self.target_update_counter = 0

		toc = time.perf_counter()


# Own tensorboard class for stats output
class ModifiedTensorBoard(TensorBoard):

	# Overriding init to set initial step and writer (we want one log file for all .fit() calls)
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.step = 1
		self.writer = tf.summary.FileWriter(self.log_dir)

	# Overriding this method to stop creating default log writer
	def set_model(self, model):
		pass

	# Overrided, saves logs with our step number
	# (otherwise every .fit() will start writing from 0th step)
	def on_epoch_end(self, epoch, logs=None):
		self.update_stats(**logs)

	# Overrided
	# We train for one batch only, no need to save anything at epoch end
	def on_batch_end(self, batch, logs=None):
		pass

	# Overrided, so won't close writer
	def on_train_end(self, _):
		pass

	# Custom method for saving own metrics
	# Creates writer, writes custom metrics and closes writer
	def update_stats(self, **stats):
		self._write_logs(stats, self.step)
