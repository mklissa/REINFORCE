import os
import gym
import numpy as np

from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
import mxnet as mx




class REINFORCE(object):
    def __init__(self, env, lr=1e-3, seed=666):
        """
        This ``REINFORCE`` implements the REINFORCE algorithm. It does so by implementing the neural network
        used to estimate the policy and also implements the training loop in `fit()`.

        Args:
            env (Gym environment) : the environment that we are training our reinforcement learning.
            lr (float) : the learning rate used for to update the neural network.
            seed (int) : the random seed used to generate data from the environment.

        """

        self.env = env
        self.lr = lr
        self.seed = seed
        self.env.seed(self.seed)
        print('Random seed: {} '.format(seed))

        self._build_net()

    def _build_net(self,hidden_size=24):
        """
        Build the neural network and set up the trainer.

        Args:
            hidden_size (int) : the size of the hidden layers in the neural network.

        """        
        self.policy_net = nn.Sequential()
        self.policy_net.add(nn.Dense(hidden_size, activation="relu"),
                nn.Dense(hidden_size, activation="relu"),
                nn.Dense(self.env.action_space.n))
        self.policy_net.initialize(init=init.Xavier())

        self.trainer = gluon.Trainer(self.policy_net.collect_params(), 'adam', {'learning_rate': self.lr})


    def update(self,lr_coeff=0.999):
        """
        Perform an update on a batch of data collected during an episode. It will also reduce the learning rate 
        after the update as a way to improve convergence.

        Args:
            lr_coeff (float) : the coefficient with which we multiply the current learning rate.

        """
        returns = self.get_returns()
        batch_size = len(self.actions)

        with autograd.record():
            all_actions = nd.softmax(self.policy_net(nd.array(self.states[:-1])))
            loss = - nd.log(all_actions[np.array(range(batch_size)), np.array(self.actions)]) * returns

        loss.backward()
        self.trainer.step(batch_size)
        self.trainer.set_learning_rate(self.trainer.learning_rate * lr_coeff)



    def predict(self,  state):
        """
        Output the probabilities for all actions and choose stochastically one of them.

        Args:
            state (array of floats) : the state for which we want to select an action.

        Returns:
            action (int) : the selected action given the state.

        """
        actions = nd.softmax(self.policy_net(nd.array([state]))).asnumpy()[0]
        return np.random.choice(len(actions),p=actions)


    def get_returns(self, discount_factor=0.99):
        """
        Calculate the return for every state. This is defined as the discounted 
        sum of rewards after visiting the state. 

        Args:
            discount_factor (float) : determines how much we care about distant 
                                        rewards (1.0) vs immediate rewards (0.).

        Returns:
            normalized_returns (array of float) : the returns, from which the mean is 
                                                 substracted to reduce the variance.
        """
        returns=[]
        curr_sum = 0.
        for r in reversed(self.rewards):
            curr_sum = r + discount_factor*curr_sum
            returns.append(curr_sum)
        returns.reverse()
        normalized_returns = nd.array(returns) - nd.mean(nd.array(returns))
        return normalized_returns

    def setup_saving(self):
        directory= os.getcwd() + '/res/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_file = "{}cartpole_seed{}.csv".format(directory,self.seed)
        return save_file, []     

    def initialize_episode(self):
        """
        Initialiazes the variables total_rewards, ewards, actions and states, and
        resets the environment.


        Returns:
            state (array of float) : the first state of the episode.
        """
        self.rewards,self.actions,self.states = [],[],[]
        self.total_rewards = 0.

        state = self.env.reset()
        self.states.append(state) 

        return state


    def add_to_trajectory(self, action, next_state, reward):
        """
        Stores in memory the action, next_state and reward. This will later be used for updates.

        Args:
            
            action (int) : the selected action in the current state.
            action (int) : the reward after selectin the action.
            next_state (array of floats) : the next state returned by the environment after selecting the action.

        Returns:
            next_state (array of float) : the next state returned by the environment after selecting the action.
        """
        self.total_rewards += reward
        self.rewards.append(reward)
        self.actions.append(action)   
        self.states.append(next_state)

        return next_state
             

    def fit(self, num_episodes=1000, save_every=5):
        """
        Implements the training loop. 

        Args:
            
            num_episodes (int) : the number of episodes we train the agent.
            save_every (int) : the rate at which we save the results, which will be used for visualization.

        """
        save_file, stats = self.setup_saving()

        for i_episode in range(num_episodes):
            if i_episode % save_every == 0 and i_episode != 0:
                np.savetxt(save_file,stats,delimiter=',') 

            state = self.initialize_episode()
            done=False
            t=0

            while not done:
                t+=1
                action = self.predict(state)
                next_state, reward, done, _ = self.env.step(action)
                state = self.add_to_trajectory(action, next_state, reward)
                if i_episode%50 ==0:self.env.render()

            print("\rEpisode {} Total Rewards {} ".format(i_episode, self.total_rewards) )
            stats.append(t)
            self.update()

            
env = gym.make("CartPole-v1")
REINFORCE(env).fit()
