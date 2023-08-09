## This version is the DQN with randomised task generation
# Has travel, reward, bid and task history metrics output.
# Introducing new types of robots and orders

import string
import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import xlwt
from xlwt import Workbook
import copy
from copy import deepcopy

# LOAD_MODEL_A = ""
# LOAD_MODEL_B = ""
# LOAD_MODEL_C = ""
# LOAD_MODEL_D = ""

# LOAD_MODEL = "" # or filepath None

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 1000
AGENTS = 4 # number of agents
TASKS = 20 # number of tasks to be completed

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 10  # episodes
SHOW_PREVIEW = False  # True for render
LOG_TASKS = True  # True to log tasks to Excel file

m = 1
mm = 1
mmm = 1
mmmm = 1

TRAIN = True

if TRAIN:
	LOAD_OR_TRAIN = 'TRAIN'
else:
	LOAD_OR_TRAIN = 'LOAD'

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Uncomment if want to specify CPU

# ks = [1,3,5,10,20]
# ns = [0,2]

# ks = [1,5,10,20]
# ns = [0,1,2,3,4]

ks = [5]
ns = [0]

alphabet = list(string.ascii_uppercase)
abilities = [2, 1, 0, 2, 2, 1, 0, 2, 2, 1, 0, 2, 2, 1, 0, 2, 2, 1, 0, 2, 2, 1, 0, 2]


## DEFINE ENV CLASS

class Env:
	SIZE = 10 # size of environment
	RETURN_IMAGES = True # use image to observe environment
	OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 1)
	ACTION_SPACE_SIZE = 12  # Number of possible actions
	NUM_ORDERS = TASKS  # Number of orders per episode
	MAX_EPS = NUM_ORDERS * 10  # Max steps per episode

	# Rewards
	TASK_COMPLETE_REWARD = 100
	LATE_PENALTY = -60
	MOVE_PENALTY = -1
	LEAD_TIME_PENALTY_FACTOR = 1.5
	MA_TASK_COMPLETE = 50

	IDLE_THRESHOLD = 50

	ROBOT = 1
	TASK_START = 2
	TASK_END = 3
	OFFERED_START = 4
	OFFERED_END = 5
	d = {1: (1),
		 2: (2),
		 3: (3),
		 4: (4),
		 5: (5)}

	def reset(self, agentList):

		self.count = 0
		self.robotList = [] # stores list of robots
		self.orderList = [] # stores list of orders
		self.episode_step = 0 # tracks episode step

		id = 1

		# Initialise robot list and order list

		for agent in agentList:
			robot = Robot(self.SIZE, id, agent.ability)
			order = Order(id=id + 10)
			id += 1
			self.robotList.append(robot)
			self.orderList.append(order)


		observations = [] # stores observations seen by each agent

		# Observation is communicated as an array with the robot and order locations as non-zeros
		for i in range(len(self.robotList)):
			if self.RETURN_IMAGES:
				observation = np.array(self.get_image(i))
			else:
				startO = Obj(self.orderList[i], "start")
				endO = Obj(self.orderList[i], "end")
				observation = (self.robotList[i] - startO) + (self.robotList[i] - endO)

			observations.append(observation)

		return observations

	def step(self, actions, offered_task=None):
		# In each step in the episode, this function is called to determine the rewards that should be assigned
		# returns new obersevations, rewards, and dones

		new_observations = [] # stores new observations seen by each agent

		for i in range(len(self.robotList)):
			if self.RETURN_IMAGES:
				new_observation = np.array(self.get_image(i, offered_task))
			else:
				startO = Obj(self.orderList[i], "start")
				endO = Obj(self.orderList[i], "end")
				new_observation = (self.robotList[i] - startO) + (self.robotList[i] - endO)
			new_observations.append(new_observation)

		self.episode_step += 1
		rewards = [] #stores reward obtained by each robot
		dones = [] # append true when episode step reaches max_eps, append false otherwise for each robot. 
		add = 0

		for i in range(len(self.robotList)):
			robot = self.robotList[i]
			order = self.orderList[i]

			robot.action(actions[i]) # stores bid for action taken

			if robot.getPos() == order.getEnd(): # if task completed
				# EDIT HERE FOR REWARD
				add += 1
				reward = self.TASK_COMPLETE_REWARD
				if self.episode_step > order.deadline:
					reward += self.LATE_PENALTY
				# print("Task complete")
				self.count += 1
			else:
				reward = self.MOVE_PENALTY

			done = False
			if self.episode_step >= self.MAX_EPS:
				done = True

			rewards.append(reward + (add * self.MA_TASK_COMPLETE))
			dones.append(done)

		return new_observations, rewards, dones

	def render(self):
		# This function displays the environment while running
		# Requires SHOW_PREVIEW to be True

		dispEnv = np.zeros((self.SIZE, self.SIZE))

		for i in range(len(self.robotList)):
			robot = self.robotList[i]
			order = self.orderList[i]

			string = ''

			if robot.currentOrder is not None:
				string += str(robot.id) + ' is assigned order ' + str(robot.currentOrder.id)

			if len(robot.nextOrders) > 0:
				for o in robot.nextOrders:
					string += " and order " + str(o.id)

			print(string)

			# for x in range(self.SIZE):
			# 	for y in range(self.SIZE):
			# 		if x == robot.x and y == robot.y:
			# 			dispEnv[x][y] = robot.id
			# 		elif x == order.x_start and y == order.y_start:
			# 			if not order.id == None:
			# 				dispEnv[x][y] = order.id
			# 		elif x == order.x_end and y == order.y_end:
			# 			if not order.id == None:
			# 				dispEnv[x][y] = order.id

			for x in range(self.SIZE):
				for y in range(self.SIZE):
					if x == robot.x and y == robot.y:
						dispEnv[x][y] = robot.id
					if robot.currentOrder is not None:
						if x == robot.currentOrder.x_start and y == robot.currentOrder.y_start:
							if not robot.currentOrder.id == None:
								dispEnv[x][y] = robot.currentOrder.id
						elif x == robot.currentOrder.x_end and y == robot.currentOrder.y_end:
							if not robot.currentOrder.id == None:
								dispEnv[x][y] = robot.currentOrder.id

		print(dispEnv)
		enter = input("Enter to continue")

	def log_location(self):
		# This function logs the movement of all the robots to a txt file
		# Requires LOG_LOCATION to be true
		dispEnv = np.zeros((self.SIZE, self.SIZE))

		for i in range(len(self.robotList)):
			# print(self.robotList[i].id)
			robot = self.robotList[i]
			order = self.orderList[i]

			for x in range(self.SIZE):
				for y in range(self.SIZE):
					if x == robot.x and y == robot.y:
						dispEnv[x][y] = robot.id
					elif x == order.x_start and y == order.y_start:
						if not order.id == None:
							dispEnv[x][y] = -order.id
					elif x == order.x_end and y == order.y_end:
						if not order.id == None:
							dispEnv[x][y] = -300 - order.id

		return dispEnv

	def get_image(self, index, offered_task=None):
		# Is called to get the observation required for each step

		env = np.zeros((self.SIZE, self.SIZE, 1), dtype=np.uint8)  # starts an rbg of our size

		robot = self.robotList[index]
		order = self.orderList[index]

		env[robot.x][robot.y] = self.d[self.ROBOT] # locate robot

		if robot.currentOrder is not None:
			env[robot.currentOrder.x_end][robot.currentOrder.y_end] = self.d[self.TASK_END]
		else:
			env[robot.x][robot.y] = self.d[self.TASK_END]

		# env[order.x_start][order.y_start] = self.d[self.TASK_START]
		# env[order.x_end][order.y_end] = self.d[self.TASK_END]

		if offered_task is not None:
			env[offered_task.x_start][offered_task.y_start] = d[OFFERED_START] # start position of task
			env[offered_task.x_end][offered_task.y_end] = d[OFFERED_END] # end position of task

		# img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
		# return img
		return env


# Create models folder
if not os.path.isdir('models'):
	os.makedirs('models')


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


class DQNAgent:
	def __init__(self, agentId, LOADMODEL="", ability=0):
		
		self.LOAD_MODEL = LOADMODEL
		self.string = MODEL_NAME + '-' + agentId
		self.ability = ability
		
		# create main model
		self.model = self.create_model()

		# create target model
		self.target_model = self.create_model()
		self.target_model.set_weights(self.model.get_weights())
		self.target_update_counter = 0 # counter to keep track of target model updates

		# create array with last n samples for training
		self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

		# custom tensorboard object
		self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.string}-{int(time.time())}")


	def create_model(self):

		if self.LOAD_MODEL is not "":
			print(f"Loading {self.LOAD_MODEL}")
			model = load_model(self.LOAD_MODEL)
			print(f"Model {self.LOAD_MODEL} loaded!")

		else:
			model = Sequential()
			model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
			model.add(Activation("relu"))
			model.add(MaxPooling2D(2, 2))
			model.add(Dropout(0.2))

			model.add(Conv2D(256, (3, 3)))
			model.add(Activation("relu"))
			model.add(MaxPooling2D(2, 2))
			model.add(Dropout(0.2))

			model.add(Flatten())
			model.add(Dense(64))

			model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))
			model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

		return model

	def update_replay_memory(self, transition):
		# Adds step's data to a memory replay array
		# transition: (observation space, action, reward, new observation space, done)
		self.replay_memory.append(transition)

	def get_qs(self, state):
		# predict Q value from model
		return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

	def train(self, terminal_state, step):
		# transition: (observation space, action, reward, new observation space, done)

		tic = time.perf_counter()

		# Start training only if certain number of samples is already saved
		if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
			return

		# Get a minibatch of random samples from memory replay table
		minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

		# Get current states from minibatch, then query NN model for Q values
		# transition[0]: state
		current_states = np.array([transition[0] for transition in minibatch]) / 255
		current_qs_list = self.model.predict(current_states)

		# Get future states from minibatch, then query NN target model for Q values
		# transition[3]: new states
		new_current_states = np.array([transition[3] for transition in minibatch]) / 255
		future_qs_list = self.target_model.predict(new_current_states)

		X = []  # images/states
		y = []  # stores q values

		# update Q values
		for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
			if not done:
				max_future_q = np.max(future_qs_list[index])
				new_q = reward + DISCOUNT * max_future_q # Q value calculation
			else:
				new_q = reward

			current_qs = current_qs_list[index]
			current_qs[action] = new_q # Q value update

			# store updated states and q values
			X.append(current_state)
			y.append(current_qs)

		# update model
		self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
					   callbacks=[self.tensorboard] if terminal_state else None)

		# update to determine if we want to update target model yet
		if terminal_state:
			self.target_update_counter += 1

		if self.target_update_counter > UPDATE_TARGET_EVERY:
			self.target_model.set_weights(self.model.get_weights())
			self.target_update_counter = 0

		toc = time.perf_counter()


class Obj:
	def __init__(self, order, start_or_end):
		if start_or_end == "start":
			self.x = order.x_start
			self.y = order.y_start
		elif start_or_end == "end":
			self.x = order.x_end
			self.y = order.y_end


class Robot:
	def __init__(self, size, id, ability=0):
		self.size = size # environment size
		self.na = 0

		# initialise robot on board
		self.x = np.random.randint(0, size - 1)
		self.y = np.random.randint(0, size - 1)

		# initialise task location
		self.task_x_end = self.x
		self.task_y_end = self.y

		self.task_x_start = None
		self.task_y_start = None

		# set robot ability and queue length

		self.ability = ability
		self.queueLength = n

		# orders handling
		self.currentOrder = None
		self.nextOrder = None
		self.nextOrders = []

		# robot status
		self.hasMoved = False
		self.complete = False
		self.pickedUp = False

		self.x_1 = self.x
		self.y_1 = self.y

		self.bidAmount = 0 # bid made for task

		self.id = id

		self.travel = 0

	def __str__(self):
		return f"Robot ({self.x}, {self.y})"

	def __sub__(self, other):
		return (self.x - other.x, self.y - other.y)

	def __eq__(self, other):
		return self.x == other.x and self.y == other.y

	def action(self, choice):
		if self.currentOrder is not None and (len(self.nextOrders) >= self.queueLength):
			self.bid(0)
		elif choice == 0:
			self.bid(0)
		elif choice == 1:
			self.bid(0.1)
		elif choice == 2:
			self.bid(0.2)
		elif choice == 3:
			self.bid(0.3)
		elif choice == 4:
			self.bid(0.4)
		elif choice == 5:
			self.bid(0.5)
		elif choice == 6:
			self.bid(0.6)
		elif choice == 7:
			self.bid(0.7)
		elif choice == 8:
			self.bid(0.8)
		elif choice == 9:
			self.bid(0.9)
		elif choice == 10:
			self.bid(1)

	def bid(self, bidAmount):
		self.bidAmount = bidAmount

	def move(self, completedOrders, step):

		self.complete = False

		if not self.currentOrder == None:

			self.hasMoved = True

			if self.pickedUp: # if order picked, set target as task end location
				target_x = self.task_x_end
				target_y = self.task_y_end
			else: # else pick up task by setting target as start location
				target_x = self.task_x_start
				target_y = self.task_y_start

			self.x_1 = deepcopy(self.x)
			self.y_1 = deepcopy(self.y)

			# move towards target
			if (self.x < target_x):
				self.x = self.x + 1
				self.travel += 1
			elif (self.x > target_x):
				self.x = self.x - 1
				self.travel += 1
			elif (self.y < target_y):
				self.y = self.y + 1
				self.travel += 1
			elif (self.y > target_y):
				self.y = self.y - 1
				self.travel += 1
			elif (self.x == self.task_x_start) and (self.y == self.task_y_start):
				if SHOW_PREVIEW:
					print(f"Order {self.currentOrder.id} picked up!")
				self.pickedUp = True
			else: # robot location is equal to task end location, aka taka complete!
				if SHOW_PREVIEW:
					print(f"Order {self.currentOrder.id} complete!")
				self.currentOrder.orderCompleted = step
				completedOrders.append(self.currentOrder)
				self.complete = True
				self.pickedUp = False
				self.currentOrder = None
				self.hasMoved = False

				# if self.nextOrder is not None:
				# 	self.assignOrder(self.nextOrder)
				# 	self.nextOrder = None
				# 	self.n = 0
				# 	self.pickedUp = False
				# 	self.hasMoved = False

				if len(self.nextOrders) > 0: # if there are orders in the queue, assign them
					self.assignOrder(self.nextOrders[0])
					self.nextOrders.remove(self.nextOrders[0])
					self.pickedUp = False
					self.hasMoved = False

				return True

			return False

	def assignOrder(self, order):

		if self.currentOrder == None: # set order as current if no order currently
			self.complete = False
			self.pickedUp = False
			self.currentOrder = order

			self.task_x_start = order.x_start
			self.task_y_start = order.y_start
			self.task_x_end = order.x_end
			self.task_y_end = order.y_end

		else: # else append order to list
			self.nextOrders.append(order)

	def isIdle(self, iteration):
		if iteration >= 5:
			if (self.x == self.x_1) and (self.y == self.y_1):
				return True
			else:
				return False
		else:
			return False

	def getPos(self):
		return [self.x, self.y]


class Order:
	def __init__(self, id=None, x_start=None, y_start=None, x_end=None, y_end=None, deadline=None, iteration=None,
				 ability_required=0):
		self.x_start = x_start
		self.y_start = y_start
		self.x_end = x_end
		self.y_end = y_end
		self.deadline = deadline
		self.empty = False
		self.id = id
		self.orderCreated = iteration
		self.orderStarted = None
		self.orderCompleted = None
		self.ability_required = ability_required
		self.agentID = None
		self.agentPos = [None, None]
		self.agentQ = None

	def getEnd(self):
		return [self.x_end, self.y_end]

	def display(self):
		print(f"ID: {self.id}")
		print(f"x_start: {self.x_start}")
		print(f"y_start: {self.y_start}")
		print(f"x_end: {self.x_end}")
		print(f"y_end: {self.y_end}")


# Helper function to generate the random parameters for the tasks
def generateParam(iteration, SIZE, ability_range):
	x_start = random.randint(0, SIZE - 1)
	y_start = random.randint(0, SIZE - 1)
	x_end = random.randint(0, SIZE - 1)
	y_end = random.randint(0, SIZE - 1)
	deadline = iteration + (iteration * random.randint(35, 50))
	ability_required = random.randint(0, ability_range - 1)

	return [x_start, y_start, x_end, y_end, deadline, ability_required]


# Helper function to find the index of the highest bidder
def argMax(bidList):
	maxBid = bidList[0]
	index = 0

	i = 0

	for bid in bidList:

		if bid > maxBid:
			maxBid = bid
			index = i

		i = i + 1

	return index


# Helper function to check if all bids are zero
def checkBids(bidList):
	for bid in bidList:
		if not (bid == 0):
			return False

	return True


for k in ks:
	for n in ns:

		# Create environment variable
		env = Env()

		MODEL_NAME = LOAD_OR_TRAIN + str(AGENTS) + "A_" + str(env.NUM_ORDERS) + "T_K" + str(k) + "_N" + str(n)
		ORDERS_AHEAD = k

		random.seed(1)

		# For stats
		ep_rewards = [-200]
		ag_rewards = []

		idleCount = 0
		m = 1
		mm = 1
		mmm = 1
		mmmm = 1

		# Create agent list
		agentList = []

		for i in range(AGENTS):
			agentList.append(DQNAgent(alphabet[i], "", abilities[i]))
			ag_rewards.append([-200])

		# Define agents of specific abilities
		# agentList[0] = DQNAgent('A', "", 2)
		# agentList[1] = DQNAgent('B', "", 1)
		# agentList[2] = DQNAgent('C', "", 0)
		# agentList[3] = DQNAgent('D', "", 2)

		# Load model for each agent 
		# agentList[0] = DQNAgent('A', 'C:\TempDataMMOH360\models\TRAIN_K1_N2-A__26432.50max_7392.45avg__255.00min__1605508878.model', 2)
		# agentList[1] = DQNAgent('B', 'C:\TempDataMMOH360\models\TRAIN_K1_N2-B__26432.50max_7392.45avg__255.00min__1605508878.model', 1)
		# agentList[2] = DQNAgent('C', 'C:\TempDataMMOH360\models\TRAIN_K1_N2-C__26432.50max_7392.45avg__255.00min__1605508878.model', 0)
		# agentList[3] = DQNAgent('D', 'C:\TempDataMMOH360\models\TRAIN_K1_N2-D__26432.50max_7392.45avg__255.00min__1605508878.model', 2)

		leadTimes = []

		# Get initial states
		current_states = env.reset(agentList)

		# Set up blank order for orders that will appear randomly throughout the episode
		orderA = Order()

		j = 0
		ep = 0

		if LOG_TASKS:
			tasklog = Workbook()
			sheet = tasklog.add_sheet('Sheet 1')
			sheets = tasklog.add_sheet('Sheet 2')
			sheetss = tasklog.add_sheet('Sheet 3')
			sheetsss = tasklog.add_sheet('Sheet 4')
			sheet.write(0, 0, "Episode")
			sheet.write(0, 1, "Start Pos")
			sheet.write(0, 2, "End Pos")
			sheet.write(0, 3, "Creation time")
			sheet.write(0, 4, "Deadline")
			sheet.write(0, 5, "Ability")
			sheet.write(0, 6, "Start time")
			sheet.write(0, 7, "Completion time")
			sheet.write(0, 8, "Agent ID")
			sheet.write(0, 9, "Agent Position at assignment")
			sheet.write(0, 10, "Agent Queue Length at assignment")
			sheets.write(0, 0, "Episode")
			sheets.write(0, 1, "Start Pos")
			sheets.write(0, 2, "End Pos")
			sheets.write(0, 3, "Creation time")
			sheets.write(0, 4, "Deadline")
			sheets.write(0, 5, "Start time")
			sheets.write(0, 6, "Completion time")
			sheets.write(0, 7, "Agent ID")
			sheets.write(0, 8, "Agent Position at assignment")
			sheets.write(0, 9, "Agent Queue Length at assignment")
			sheetss.write(0, 0, "Episode")
			sheetss.write(0, 1, "Start Pos")
			sheetss.write(0, 2, "End Pos")
			sheetss.write(0, 3, "Creation time")
			sheetss.write(0, 4, "Deadline")
			sheetss.write(0, 5, "Start time")
			sheetss.write(0, 6, "Completion time")
			sheetss.write(0, 7, "Agent ID")
			sheetss.write(0, 8, "Agent Position at assignment")
			sheetss.write(0, 9, "Agent Queue Length at assignment")
			sheetss.write(0, 10, "Exit reason")
			sheetsss.write(0, 0, "Episode")
			sheetsss.write(0, 1, "Start Pos")
			sheetsss.write(0, 2, "End Pos")
			sheetsss.write(0, 3, "Creation time")
			sheetsss.write(0, 4, "Deadline")
			sheetsss.write(0, 5, "Start time")
			sheetsss.write(0, 6, "Completion time")
			sheetsss.write(0, 7, "Agent ID")
			sheetsss.write(0, 8, "Agent Position at assignment")
			sheetsss.write(0, 9, "Agent Queue Length at assignment")
			sheetsss.write(0, 10, "Exit reason")

		# For each episode
		for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episode"):

			t_ep_start = time.time()

			tin = time.perf_counter()  # Used for timing purposes
			j += 1
			n_step = 0

			for agent in agentList:
				agent.tensorboard.step = episode

			episode_rewards = np.zeros(len(agentList))
			global_reward = 0
			reward = 0
			step = 1
			current_states = env.reset(agentList)

			task_counts = []
			travel_norms = []

			# Initialise robot/agent stats to zero
			for robot in env.robotList:
				robot.travel = 0
				travel_norms.append(0)
				task_counts.append(0)

			done = False
			bid = None
			bidListExport = []

			num_tasks_complete = False
			enough_tasks = 0
			task_created = 0
			exit = ""

			orderQueue = []  # Queue of orders that are available to be bid on
			completedOrders = []  # List of all completed orders per episode - used for logging
			assignedOrders = []  # List of all assigned order per episode
			createdOrders = []  # List of all orders created in the episode
			missedOrders = []

			# Generating 10 orders to be in the queue from the beginning
			for i in range(10):
				# for i in range(1):
				params = generateParam(0, env.SIZE, 3)
				orderQueue.append(
					Order(id=i + 100, x_start=params[0], y_start=params[1], x_end=params[2], y_end=params[3],
						  deadline=params[4], iteration=0, ability_required=params[5]))
				createdOrders.append(
					Order(id=i + 100, x_start=params[0], y_start=params[1], x_end=params[2], y_end=params[3],
						  deadline=params[4], iteration=0, ability_required=params[5]))

			if (ep == 32):
				SHOW_PREVIEW = False
			else:
				SHOW_PREVIEW = False

			idle_count = 0

			# For each step
			while not done:

				idle = []

				for robot in env.robotList:
					if not robot.isIdle(n_step):
						idle.append(False)
					else:
						idle.append(True)

				if False in idle:
					idle_count = 0
				else:
					idle_count += 1

				if n_step >= env.MAX_EPS:
					exit = "MAX"
					break
				elif len(completedOrders) >= env.NUM_ORDERS:
					exit = "COMPLETED"
					break
				elif (len(completedOrders) + len(missedOrders)) >= env.NUM_ORDERS:
					exit = "MISSED"
					break
				elif (idle_count >= env.IDLE_THRESHOLD):
					exit = "IDLING"
					break
				else:

					t_step_start = time.time()
					tic = time.perf_counter()  # Used for timing purposes
					chance = random.randint(0, 100)
					reached = []

					# If the order is generated
					if len(createdOrders) < env.NUM_ORDERS:
						# Generates random order
						params = generateParam(env.episode_step, env.SIZE, 3)
						order = Order(id=-100 - env.episode_step, x_start=params[0], y_start=params[1], x_end=params[2],
									  y_end=params[3], deadline=params[4], iteration=step, ability_required=params[5])
						order.empty = True
						new_states = None
						orderA = deepcopy(order)
						orderA.empty = False

						task_created += 1
						# Adds order to the queue
						orderQueue.append(orderA)
						createdOrders.append(orderA)

					# If the queue is not empty
					if not (len(orderQueue) == 0):
						if SHOW_PREVIEW:
							print("From order queue")
							print(
								f"Order up for bid is {orderA.id} with Level {orderA.ability_required} ability required")

						actions_ahead = []
						for i in range(len(env.robotList)):
							actions_ahead.append([])
							for j in range(ORDERS_AHEAD):
								if j < len(orderQueue):
									orderW = orderQueue[j]
									if orderW.ability_required > env.robotList[i].ability:
										actions_ahead[i].append(0)
									elif env.robotList[i].currentOrder is not None and (
											len(env.robotList[i].nextOrders) >= env.robotList[i].queueLength):
										actions_ahead[i].append(0)
									else:
										if TRAIN:
											if np.random.random() > epsilon:
												actions_ahead[i].append(
													np.argmax(agentList[i].get_qs(current_states[i])))
											else:
												actions_ahead[i].append(np.random.randint(0, env.ACTION_SPACE_SIZE))
										else:
											actions_ahead[i].append(np.argmax(agentList[i].get_qs(current_states[i])))
						if SHOW_PREVIEW:
							print(f"actions_ahead is {actions_ahead}")

						maxonly_actions_ahead = []
						for i in range(len(actions_ahead)):
							maxonly_actions_ahead.append([])
							maxVal = max(actions_ahead[i])
							for j in range(len(actions_ahead[i])):
								if actions_ahead[i][j] < maxVal:
									maxonly_actions_ahead[i].append(0)
								else:
									maxonly_actions_ahead[i].append(maxVal)

						if SHOW_PREVIEW:
							print(f"maxonly_actions_ahead is {actions_ahead}")

						for i in range(ORDERS_AHEAD):
							if SHOW_PREVIEW:
								print(f"Length of queue is {len(orderQueue)}")
								print(f"i is {i}")

							if i < len(orderQueue):

								orderW = orderQueue[i]

								bidList = []
								actions = []
								dones = []

								for j in range(len(env.robotList)):
									actions.append(maxonly_actions_ahead[j][i])

								# Step taken to determine new states and rewards for each agent
								new_states, rewards, dones = env.step(actions)

								# Each robot then places its bid on the order
								for robot in env.robotList:
									bid = robot.bidAmount
									bidList.append(bid)
									bidListExport = deepcopy(bidList)

								if SHOW_PREVIEW:
									print(f"For order {orderW.id}, bids are {bidList}")

								# If all the bids are 0 then no suitable agents are available for the task to be completed.
								if checkBids(bidList):
									break
								else:
									# Otherwise, the order is assigned to the highest bidder and the order is removed from the queue
									maxIndex = argMax(bidList)

									orderW.agentID = env.robotList[maxIndex].id
									orderW.agentPos = [env.robotList[maxIndex].task_x_end,
													   env.robotList[maxIndex].task_y_end]
									orderW.agentQ = len(env.robotList[maxIndex].nextOrders)

									env.robotList[maxIndex].assignOrder(orderW)
									env.orderList[maxIndex] = orderW

									assignedOrders.append(orderW)

									travel_norms[maxIndex] += abs(orderW.x_start - orderW.x_end) + abs(
										orderW.y_start - orderW.y_end)
									task_counts[maxIndex] += 1

									orderQueue.remove(orderW)

					# To catch the error if new states are none
					if new_states is None:
						actions = []
						for agent in agentList:
							actions.append(0)

						new_states, rewards, dones = env.step(actions)

					current_states = new_states

					# If any agent returns True in the "dones" list, then it means a task has been completed.
					if True in dones:
						num_tasks_complete = True

					# For each robot
					for i in range(len(env.robotList)):
						# To log the lead time for each order, check that the order has not started execution in one iteration and that it has in the following iteration
						check = env.robotList[i].hasMoved
						test = env.robotList[i].move(completedOrders, step)

						# Test variable returns whether an order has been completed after the agent has moved in that step
						if test:
							enough_tasks += 1
						reached.append(test)
						lTime = 0

						if check == False and env.robotList[i].hasMoved == True:
							leadTime = step - env.robotList[i].currentOrder.orderCreated
							env.robotList[i].currentOrder.orderStarted = step
							leadTimes.append(leadTime)
							lTime = leadTime

						missed_task = 0

						if env.robotList[i].currentOrder is not None:

							currentOrder_calc = env.robotList[i].currentOrder
							travel_time = (env.SIZE * 2) + abs(
								currentOrder_calc.x_start - currentOrder_calc.x_end) + abs(
								currentOrder_calc.y_start - currentOrder_calc.y_end)

							if env.robotList[i].currentOrder.orderStarted is not None:
								if (step - env.robotList[i].currentOrder.orderStarted) >= travel_time:
									missed_task = 20
									missedOrders.append(currentOrder_calc)
									env.robotList[i].currentOrder = None

						# Account for lead time in the reward
						rewards[i] = rewards[i] - env.LEAD_TIME_PENALTY_FACTOR * lTime - missed_task
						agentList[i].update_replay_memory(
							(current_states[i], actions[i], rewards[i], new_states[i], dones[i]))

						# Train the agent
						if TRAIN:
							agentList[i].train(dones[i], step)

						episode_rewards[i] += rewards[i]

					# If the render option is set to True then the environment will be displayed
					if SHOW_PREVIEW:
						if True not in reached:
							env.render()

					# Sum rewards for all agents to see global reward
					global_reward = np.sum(episode_rewards)

					step += 1

					# If the episode has exceeded 200 steps then the episode is terminated
					if n_step >= env.MAX_EPS:
						done = True
						break

					n_step = n_step + 1
					toc = time.perf_counter()
					t_step = time.time() - t_step_start
				# print(f"Step took {t_step:0.6f} seconds")
				# print(f"Step took {toc-tic:0.2f} seconds")

			# For logging and stats
			ep_rewards.append(global_reward)

			norms2 = 0
			for order in completedOrders:
				norms2 += abs(order.x_start - order.x_end) + abs(order.y_start - order.y_end)
			if norms2 == 0:
				norms2 = 1

			for d in range(len(ag_rewards)):
				ag_rewards[i].append(episode_rewards[i])

			if not episode % AGGREGATE_STATS_EVERY or episode == 1:
				average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
				min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
				max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

				if len(leadTimes[-AGGREGATE_STATS_EVERY:]) == 0:
					divv = 1
				else:
					divv = len(leadTimes[-AGGREGATE_STATS_EVERY:])

				lead_time_avg = sum(leadTimes[-AGGREGATE_STATS_EVERY:]) / divv

				steps = step

				if len(bidListExport) == 0:
					bidsA = -2
					bidsB = -2
					bidsC = -2
				else:
					bidsA = bidListExport[0]
					bidsB = bidListExport[1]
				# bidsC = bidListExport[2]

				total_delay = 0
				for order in completedOrders:
					total_delay += (order.orderCompleted - order.orderCreated)

				div_delay = len(completedOrders)
				if div_delay == 0:
					div_delay = 1

				avg_delay = total_delay / div_delay

				z_total_travel = 0
				for robot in env.robotList:
					z_total_travel += robot.travel

				norm_total = 0
				for norm in travel_norms:
					norm_total += norm
				if norm_total == 0:
					norm_total = 1

				u = 0
				for agent in agentList:
					if travel_norms[u] == 0:
						div = 1
					else:
						div = travel_norms[u]
					agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
												   reward_max=max_reward, epsilon=epsilon, num_steps=steps,
												   z_tasks=task_counts[u], tasks_completed=enough_tasks,
												   avg_lead_time=lead_time_avg, z_travel=env.robotList[u].travel,
												   z_travel_norm=env.robotList[u].travel / div,
												   z_total_travel=z_total_travel,
												   z_total_travel_norm=z_total_travel / norm_total,
												   z_total_travel_norm2=z_total_travel / norms2, avg_delay=avg_delay,
												   total_delay=total_delay)

					# Save model, but only when min reward is greater or equal a set value
					if min_reward >= MIN_REWARD:
						agent.model.save(
							f'models/{agent.string}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
					u += 1

			if LOG_TASKS:
				for order in completedOrders:
					sheet.write(m, 0, str(ep))
					sheet.write(m, 1, str(order.x_start) + ',' + str(order.y_start))
					sheet.write(m, 2, str(order.x_end) + ',' + str(order.y_end))
					sheet.write(m, 3, str(order.orderCreated))
					sheet.write(m, 4, str(order.deadline))
					sheet.write(m, 5, str(order.ability_required))
					sheet.write(m, 6, str(order.orderStarted))
					sheet.write(m, 7, str(order.orderCompleted))
					sheet.write(m, 8, str(order.agentID))
					sheet.write(m, 9, str(order.agentPos[0]) + "," + str(order.agentPos[1]))
					sheet.write(m, 10, str(order.agentQ))

					m += 1

				for order in assignedOrders:
					sheets.write(mm, 0, str(ep))
					sheets.write(mm, 1, str(order.x_start) + ',' + str(order.y_start))
					sheets.write(mm, 2, str(order.x_end) + ',' + str(order.y_end))
					sheets.write(mm, 3, str(order.orderCreated))
					sheets.write(mm, 4, str(order.deadline))
					sheets.write(mm, 5, str(order.orderStarted))
					sheets.write(mm, 6, str(order.orderCompleted))
					sheets.write(mm, 7, str(order.agentID))
					sheets.write(mm, 8, str(order.agentPos[0]) + "," + str(order.agentPos[1]))
					sheets.write(mm, 9, str(order.agentQ))

					mm += 1

				for order in createdOrders:
					sheetss.write(mmm, 0, str(ep))
					sheetss.write(mmm, 1, str(order.x_start) + ',' + str(order.y_start))
					sheetss.write(mmm, 2, str(order.x_end) + ',' + str(order.y_end))
					sheetss.write(mmm, 3, str(order.orderCreated))
					sheetss.write(mmm, 4, str(order.deadline))
					sheetss.write(mmm, 5, str(order.orderStarted))
					sheetss.write(mmm, 6, str(order.orderCompleted))
					sheetss.write(mmm, 7, str(order.agentID))
					sheetss.write(mmm, 8, str(order.agentPos[0]) + "," + str(order.agentPos[1]))
					sheetss.write(mmm, 9, str(order.agentQ))
					sheetss.write(mmm, 10, exit)

					mmm += 1

				for order in missedOrders:
					sheetsss.write(mmmm, 0, str(ep))
					sheetsss.write(mmmm, 1, str(order.x_start) + ',' + str(order.y_start))
					sheetsss.write(mmmm, 2, str(order.x_end) + ',' + str(order.y_end))
					sheetsss.write(mmmm, 3, str(order.orderCreated))
					sheetsss.write(mmmm, 4, str(order.deadline))
					sheetsss.write(mmmm, 5, str(order.orderStarted))
					sheetsss.write(mmmm, 6, str(order.orderCompleted))
					sheetsss.write(mmmm, 7, str(order.agentID))
					sheetsss.write(mmmm, 8, str(order.agentPos[0]) + "," + str(order.agentPos[1]))
					sheetsss.write(mmmm, 9, str(order.agentQ))
					sheetsss.write(mmmm, 10, exit)

					mmmm += 1

				name = 'Task Log_' + str(AGENTS) + "A_" + str(env.NUM_ORDERS) + "T_K" + str(k) + "N" + str(n) + ".xls"
				tasklog.save(name)

			# Decay epsilon
			if epsilon > MIN_EPSILON:
				epsilon *= EPSILON_DECAY
				epsilon = max(MIN_EPSILON, epsilon)

			tim = time.perf_counter()
			# print(f"Episode took {tim-tin:0.2f} seconds")

			t_ep = time.time() - t_ep_start
			print(f"episode took {t_ep:0.02f} seconds")
			ep += 1

		if LOG_TASKS:
			tasklog.save(name)

