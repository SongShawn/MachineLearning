import random
import numpy as np
import math
import os
import copy

class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):
        
        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        # sxn
        self.nS = self.maze.height * self.maze.width

        self.Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self, step_times=None):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            # TODO 1. No random choice when testing
            self.epsilon = 1.0
        else:
            # TODO 2. Update parameters when learning
            if step_times != None:
                if step_times <= self.maze.height*self.maze.width:
                    self.epsilon = 1.0/math.sqrt(self.t)
                else:
                    self.epsilon = 1.0/math.log(self.t+1000,1000)
            else:
                if self.t < 10 * 1000:
                    self.epsilon = 1.0/math.log(self.t+1000,1000)
                else :
                    self.epsilon = 1.0/math.sqrt(self.t)

        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """

        # TODO 3. Return robot's current state
        r, c = self.maze.robot['loc']
        return r, c

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        # TODO 4. Create qtable with current state
        # Our qtable should be a two level dict,
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # not change it.
        if state not in self.Qtable:
            self.Qtable[state] = {a:0.0 for a in self.maze.valid_actions}
        pass

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():

            # TODO 5. Return whether do random choice
            # hint: generate a random number, and compare
            # it with epsilon
            return random.uniform(0,1) < self.epsilon

        action = None
        if self.learning:
            if is_random_exploration():
                # TODO 6. Return random choose aciton
                action = np.random.choice(self.maze.valid_actions)
            else:
                # TODO 7. Return action with highest q value
                action = max(self.Qtable[self.state], key=self.Qtable[self.state].get)
        elif self.testing:
            # TODO 7. choose action with highest q value
            action = max(self.Qtable[self.state], key=self.Qtable[self.state].get)
        else:
            # TODO 6. Return random choose aciton
            action = np.random.choice(self.maze.valid_actions,\
                p=get_epsilon_greedy_probs(self.state))
        return action

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """

        if self.learning:
            # TODO 8. When learning, update the q table according
            # to the given rules
            QTable_next_state = max(list(self.Qtable[next_state].values()))

            self.Qtable[self.state][action] = (1 - self.alpha) * self.Qtable[self.state][action] + \
                self.alpha * (r + self.gamma * QTable_next_state)
            
    def update(self, avg_step_times_last_10=None):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """

        self.t += 1

        self.state = self.sense_state() # Get the current state
        self.create_Qtable_line(self.state) # For the state, create q table line

        action = self.choose_action() # choose action for this state
        reward = self.maze.move_robot(action) # move robot for given action

        next_state = self.sense_state() # get next state
        self.create_Qtable_line(next_state) # create q table line for next state

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state) # update q table
            self.update_parameter(avg_step_times_last_10) # update parameters

        return action, reward

    def Qstate_to_file(self, file_name):
        f = open(file_name, 'w') 
        for r in range(self.maze.height):
            for c in range(self.maze.width):
                if (r,c) in self.Qtable:
                    f.writelines('({},{}): {}\n'.format(r, c, self.Qtable[(r,c)])) 
                else:
                    f.writelines('({},{}): {}\n'.format(r, c, 'NA')) 
        f.close()
