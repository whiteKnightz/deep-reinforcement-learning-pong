""" Helper class to train an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """

import numpy as np


class Helper:

    def __init__(self, H, batch_size, learning_rate, gamma, decay_rate, model, epx):
        self.H = H
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.resume = False
        self.render = False
        self.D = 80 * 80
        self.model = model
        self.epx = epx

    def set_epx(self, epx):
        self.epx = epx

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]

    def prepro(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        return I.astype(np.float64).ravel()

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def policy_forward(self, x):
        h = np.dot(self.model['W1'], x)
        h[h < 0] = 0  # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = self.sigmoid(logp)
        return p, h  # return probability of taking action 2, and hidden state

    def policy_backward(self, eph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0  # backpro prelu
        dW1 = np.dot(dh.T, self.epx)
        return {'W1': dW1, 'W2': dW2}
