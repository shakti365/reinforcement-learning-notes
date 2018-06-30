import numpy as np

class GridWorld:
	"""
	Implementation of GridWorld environment with an agent, 
	non-termial / terminal states and transition probabilties.

	Parameters:
	-----------
	num_rows: int 
			number of rows in the grid
	
	num_columns: int
			number of columns in the grid
	"""
	def __init__(self, num_rows, num_columns):
		self.num_rows = num_rows
		self.num_columns = num_columns
		
		# Initialize state space.
		self.states = np.arange(num_rows*num_columns).reshape(num_rows, num_columns)

		self.state_space = self.states.flatten()

		self.num_states = len(self.state_space)

		self.terminal_states = [0, 15]

		# Initialize action space.
		self.actions = {
			'UP': self._up,
			'DOWN': self._down,
			'LEFT': self._left,
			'RIGHT': self._right
		}

		self.action_space = self.actions.keys()
		self.num_actions = len(self.action_space)

		# Initialize transition probabilities.
		self.transition_proba = {action:1.0 for action in self.actions.keys()}
		
		self.current_state, self.x, self.y = None, None, None
		if self.current_state == None:
			self.reset()


	def reset(self):
		"""Resets the environment to original states."""
		# Initialize current_state of the agent.
		self.current_state = 5
		self.x, self.y  = np.where(self.states==self.current_state)

	def _up(self, current_state):
		"""
		Takes UP action and returns next state given current state.
		"""
		x,y = np.where(self.states == current_state)

		if x != 0:
			x = x-1
		
		next_state = np.asscalar(self.states[x,y])
		return next_state

	def _down(self, current_state):
		"""
		Takes DOWN action and returns next state given current state.
		"""

		x,y = np.where(self.states == current_state)
		if x != self.num_rows-1:
			x = x+1
		
		next_state = np.asscalar(self.states[x,y])
		return next_state

	def _left(self, current_state):
		"""
		Takes LEFT action and returns next state given current state.
		"""

		x,y = np.where(self.states == current_state)
		if y != 0:
			y = y-1
		
		next_state = np.asscalar(self.states[x,y])
		return next_state

	def _right(self, current_state):
		"""
		Takes RIGHT action and returns next state given current state.
		"""

		x,y = np.where(self.states == current_state)
		if y != self.num_columns-1:
			y = y+1
		
		next_state = np.asscalar(self.states[x,y])
		return next_state

	def step(self, action, state=None):
		"""Takes an action from current state 
		to end up in next state generating a reward
		
		Parameters:
		-----------

		Returns:
		--------

		"""
		if action not in self.actions.keys():
			raise ValueError('{} is not a valid action. Choose from {}'.format(action, self.actions))

		reward = 0
		next_state = 0

		if state==None:
			state=self.current_state

		if state not in self.terminal_states:
			next_state = self.actions[action](state)
			reward = -1

		assert next_state in self.states, """{} -> {} Transtition is invalid. 
		Next State from this action is not a valid state. 
		Check implementation of actions to correct this""".format(self.current_state, next_state)

		self.current_state = next_state

		return reward, next_state