from typing import Match
import numpy as np
import random

class Env:

	def __init__(self):
		self.SIZE = 10 # size of environment
		self.TASKS = 10 # number of tasks to be completed
		self.AGENTS = 4 # number of agents
		self.num_orders = self.TASKS # seems redundant 
		self.ACTION_SPACE_SIZE = 12 # number of possible actions
		self.MAX_EPS = self.num_orders*10 # max steps per episode
		self.observation_space_values = (self.SIZE, self.SIZE, 1)

		# Rewards
		self.TASK_COMPLETE_REWARD = 100
		self.LATE_PENALTY = -60
		self.MOVE_PENALTY = -1
		self.LEAD_TIME_PENALTY_FACTOR = 1.5
		self.MA_TASK_COMPLETE = 50
		self.MIN_REWARD = -200

		self.return_images = True # use image to observe environment
		
		self.rob_task = {"robot": 1, 
					"task_start":2, 
					"task_end":3, 
					"offered_start":4, 
					"offered_end": 5}

		self.count = 0 # unused
		self.robotList = [] # stores list of robots
		self.orderList = [] # stores list of orders
		self.episode_step = 0 # tracks episode step

	def reset(self, agentList):
		
		self.count = 0 # unused
		self.robotList = [] # stores list of robots
		self.orderList = [] # stores list of orders
		self.episode_step = 0 # tracks episode step

		id = 1

		# Initialise robot list and order list
		for agent in agentList:
			robot = Robot(self.SIZE, id, agent.ability)
			order = Order(id = id+10)
			self.robotList.append(robot)
			self.orderList.append(order)
			id += 1

		observations = [] # stores observations seen by each agent

		# Observation is communicated as an array with the robot and order locations as non-zeros
		for i in range(len(self.robotList)):
			observation = np.array(self.get_image(i))
			observations.append(observation)

		return observations

	def step(self, actions, offered_task=None):
		# In each step in the episode, this function is called to determine the rewards that should be assigned
		# returns new obersevations, rewards, and dones

		new_observations = [] # stores new observations seen by each agent

		for i in range(len(self.robotList)):
			new_observation = np.array(self.get_image(i, offered_task))
			new_observations.append(new_observation)

		self.episode_step += 1
		rewards = [] #stores reward obtained by each robot
		dones = [] # append true when episode step reaches max_eps, append false otherwise for each robot. 
		no_tasks_complete = 0

		for i in range(len(self.robotList)):
			robot = self.robotList[i]
			order = self.orderList[i]

			robot.action(actions[i]) # stores bid for action taken

			if robot.getPos() == order.getEnd(): # if task completed
				# EDIT HERE FOR REWARD
				no_tasks_complete += 1
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

			rewards.append(reward + (no_tasks_complete*self.MA_TASK_COMPLETE))
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
		#Is called to get the observation required for each step
		env = np.zeros((self.SIZE, self.SIZE, 1), dtype=np.uint8)  # starts an rbg of our size

		robot = self.robotList[index]
		order = self.orderList[index]

		env[robot.x][robot.y] = self.rob_task["robot"] # locate robot

		if robot.currentOrder is not None:
			env[robot.currentOrder.x_end][robot.currentOrder.y_end] = self.rob_task["task_start"] # if robot has current order then do...
		else:
			env[robot.x][robot.y] = self.rob_task["task_end"]

		if offered_task is not None:
			env[offered_task.x_start][offered_task.y_start] = self.rob_task[self.OFFERED_START] # start position of task
			env[offered_task.x_end][offered_task.y_end] = self.rob_task[self.OFFERED_END] # end position of task

		return env

class Robot:
	def __init__(self, size, id, n, ability = 0):
		self.id = id
		self.size = size # environment size

		# initialise robot on board
		self.x = np.random.randint(0, size-1)
		self.y = np.random.randint(0, size-1)

		# initialise task end location
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
		self.hasMoved = False # unused?
		self.complete = False
		self.pickedUp = False

		self.bidAmount = 0 # bid made for task
		self.travel = 0 #

	def __str__(self):
		return f"Robot ({self.x}, {self.y})"

	def __sub__(self, other):
		return (self.x-other.x, self.y-other.y)

	def __eq__(self, other):
		return self.x == other.x and self.y == other.y

	def action(self, choice):
		if self.currentOrder is not None and (len(self.nextOrders) >= self.queueLength):
			self.set_bid(0)
		elif choice == 0:
			self.set_bid(0)
		elif choice == 1:
			self.set_bid(0.1)
		elif choice == 2:
			self.set_bid(0.2)
		elif choice == 3:
			self.set_bid(0.3)
		elif choice == 4:
			self.set_bid(0.4)
		elif choice == 5:
			self.set_bid(0.5)
		elif choice == 6:
			self.set_bid(0.6)
		elif choice == 7:
			self.set_bid(0.7)
		elif choice == 8:
			self.set_bid(0.8)
		elif choice == 9:
			self.set_bid(0.9)
		elif choice == 10:
			self.set_bid(1)

	def set_bid(self, bidAmount):
		self.bidAmount = bidAmount

	def move(self, completedOrders, step):

		self.complete = False

		if self.currentOrder is not None: 

			self.hasMoved = True

			if self.pickedUp: # if order picked, set target as task end location
				target_x = self.task_x_end
				target_y = self.task_y_end
			else: # else pick up task by setting target as start location
				target_x = self.task_x_start
				target_y = self.task_y_start

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
				# if SHOW_PREVIEW:
				# 	print(f"Order {self.currentOrder.id} picked up!")
				self.pickedUp = True
			else: # robot location is equal to task end location, aka taka complete!
				# if SHOW_PREVIEW:
				# 	print(f"Order {self.currentOrder.id} complete!")
				self.currentOrder.orderCompleted = step
				completedOrders.append(self.currentOrder)
				self.complete = True
				self.pickedUp = False
				self.currentOrder = None
				self.hasMoved = False

				if len(self.nextOrders) > 0: # if there are orders in the queue, assign them
					self.assignOrder(self.nextOrders[0])
					self.nextOrders.remove(self.nextOrders[0])
					self.pickedUp = False
					self.hasMoved = False

				return True

			return False


	def assignOrder(self, order):

		if self.currentOrder is None: # set order as current if no order currently
			self.complete = False
			self.pickedUp = False
			self.currentOrder = order

			self.task_x_start = order.x_start
			self.task_y_start = order.y_start
			self.task_x_end = order.x_end
			self.task_y_end = order.y_end

		else: # else append order to list
			self.nextOrders.append(order)

	def getPos(self):
		return [self.x, self.y]

class Order:
	def __init__(self, id = None, size = None, iteration = None, ability_range = None):
		self.id = id
		self.x_start = random.randint(0, size-1)
		self.y_start = random.randint(0, size-1)
		self.x_end = random.randint(0, size-1)
		self.y_end = random.randint(0, size-1)
		self.deadline = iteration + (iteration * random.randint(35, 50))
		self.ability_required = random.randint(0, ability_range-1)
		self.orderCreated = iteration
		
		self.orderStarted = None
		self.orderCompleted = None
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

	def generateRandParameters(iteration, size, ability_range):
		x_start = random.randint(0, size-1)
		y_start = random.randint(0, size-1)
		x_end = random.randint(0, size-1)
		y_end = random.randint(0, size-1)
		deadline = iteration + (iteration * random.randint(35, 50))
		ability_required = random.randint(0, ability_range-1)

		return [x_start, y_start, x_end, y_end, deadline, ability_required]		