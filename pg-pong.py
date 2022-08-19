""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import datetime
import csv

import numpy as np
import pickle
import gym
import os, psutil

from helper import Helper


def run_algo(H, learning_rate, writer_obj, type_of_variable, break_number):
    batch_size = 10  # every how many episodes to do a param update?
    gamma = 0.99  # discount factor for reward
    decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
    resume = False  # resume from previous checkpoint?
    render = False
    number_of_wins = 0
    number_of_loss = 0
    start_date_time = datetime.datetime.now()
    running = True
    process = psutil.Process(os.getpid())
    # model initialization
    D = 80 * 80  # input dimensionality: 80x80 grid

    if resume:
        model = pickle.load(open('save.p', 'rb'))
    else:
        model = {}
        model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
        model['W2'] = np.random.randn(H) / np.sqrt(H)

    helper = Helper(H, batch_size, learning_rate, gamma, decay_rate, model, None)

    grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory
    env = gym.make("Pong-v0")
    observation = env.reset()
    prev_x = None  # used in computing the difference frame
    xs, hs, dlogps, drs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0

    # print("\n\nRunning algorithm with: \t\t neurons: %d \t\t& \t\t learning rate: %f" % (H, learning_rate))

    while running:
        if render: env.render()

        # preprocess the observation, set input to network to be difference image
        cur_x = helper.prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, h = helper.policy_forward(x)
        action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

        # record various intermediates (needed later for backprop)
        xs.append(x)  # observation
        hs.append(h)  # hidden state
        y = 1 if action == 2 else 0  # a "fake label"
        dlogps.append(
            y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        # if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        #     print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))

        if done:  # an episode finished
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            helper.set_epx(np.vstack(xs))
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs, hs, dlogps, drs = [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = helper.discount_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
            grad = helper.policy_backward(eph, epdlogp)
            for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

            # perform rmsprop parameter update every batch_size episodes
            if episode_number % batch_size == 0:
                for k, v in model.items():
                    g = grad_buffer[k]  # gradient
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward) )
            if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
            if running_reward > break_number:
                running = False
                date_time_now = datetime.datetime.now()
                print("Ran algorithm with: \t\t neurons: %d \t\t& \t\t learning rate: %f" % (H, learning_rate))
                writer_obj.writerow(
                    [type_of_variable, H, learning_rate, start_date_time, date_time_now, start_date_time.timestamp(),
                     date_time_now.timestamp(), (date_time_now - start_date_time).total_seconds(),
                     process.memory_info().rss / 1024 ** 2])
            reward_sum = 0
            observation = env.reset()  # reset env
            prev_x = None


f = open('collected_data/data.csv', 'a')
writer = csv.writer(f)
# header = ['Variable Type', 'Number of Neurons', 'Learning Rate', 'Start Date Time', 'Stop Date Time',
#           'Start Time (seconds)', 'Stop Time (seconds)', 'Time Taken (Stop-Start) in Seconds', 'RAM Used(in MB)']
# writer.writerow(header)

h_list = [200, 400, 600, 800, 1000]  # number of hidden layer neurons
learning_rate_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

for h in h_list:
    run_algo(h, 1e-4, writer, "NEURONS", -11)

for rate_of_learning in learning_rate_list:
    run_algo(200, rate_of_learning, writer, 'LEARNING', -15)
f.close()
