import numpy as np
from collections import deque
from sklearn.utils import check_random_state


def lstm_update(env, network, episode_buffer, gamma, random_state=None):
    random_state = check_random_state(random_state)
    episode_index = random_state.choice(len(episode_buffer))

    X = episode_buffer[episode_index]
    T = len(X)

    states = []
    prev_h_a = np.zeros(network.n_units[1])
    prev_h_s = np.zeros(network.n_units[1])
    for t in range(T):
        x = X[t]

        state = network.forward_pass(x, prev_h_a, prev_h_s)

        prev_h_a = state['activation_output']
        prev_h_s = state['activation_cell']

        states.append(state)

    states.append({'activation_forget_gate': np.zeros(network.n_units[1])})

    Y = np.zeros((T, env.n_actions))
    mask = np.zeros((T, env.n_actions))
    for t in range(T - 2):
        next_x = X[t + 1]

        next_action = np.argmax(next_x[0: env.n_actions])
        next_reward = next_x[env.n_actions]

        values = states[t + 1]['output_layer_activation']

        Y[t, next_action] = next_reward + gamma*np.max(values)
        mask[t, next_action] = 1

    # Episode is complete
    if (episode_index < len(episode_buffer) - 1):
        next_x = X[T - 1]

        next_action = np.argmax(next_x[0: env.n_actions])
        next_reward = next_x[env.n_actions]

        Y[T - 2, next_action] = next_reward
        mask[T - 2, next_action] = 1

    errors = network.backward_pass(X, Y, mask, states)
    network.update_parameters(states, errors, T)


def q_lstm(env, network, nepisodes, gamma=0.99, min_epsilon=0.1,
           max_epsilon=1.0, decay_epsilon=0.99, max_queue=512, verbose=0,
           random_state=None):
    random_state = check_random_state(random_state)
    episode_buffer = deque(maxlen=max_queue)

    if verbose > 0:
        episode_return = 0.0
        episode_gamma = 1.0

    epsilon = max(min_epsilon, max_epsilon)
    for episode in range(nepisodes):
        if verbose > 0:
            print('Episode {0}.'.format(episode + 1))

        action_reward = np.zeros(env.n_actions + 1)
        s = env.start()
        x = np.concatenate([action_reward, s])
        episode_buffer.append([x])

        prev_h_a = np.zeros(network.n_units[1])
        prev_h_s = np.zeros(network.n_units[1])

        if verbose > 1:
            step = 0
            print('Step {0}.'.format(step+1))
        if verbose > 2:
            print(env)

        while not env.ended():
            network_state = network.forward_pass(x, prev_h_a, prev_h_s)
            values = network_state['output_layer_activation']

            prev_h_a = network_state['activation_output']
            prev_h_s = network_state['activation_cell']

            if random_state.uniform(0, 1) < epsilon:
                a = random_state.choice(env.n_actions)
            else:
                a = np.argmax(values)

            next_s, next_r = env.next_state_reward(a)

            action_reward = np.zeros(env.n_actions + 1)
            action_reward[a] = 1
            action_reward[-1] = next_r

            x = np.concatenate([action_reward, next_s])
            episode_buffer[-1].append(x)

            lstm_update(env, network, episode_buffer, gamma, random_state)

            if verbose > 0:
                episode_return += episode_gamma*next_r
                episode_gamma *= gamma
            if verbose > 1:
                step += 1
                print('Step {0}.'.format(step+1))
            if verbose > 2:
                print(env)

        epsilon = max(min_epsilon, epsilon*decay_epsilon)

        if verbose > 0:
            print('Return: {0}.'.format(episode_return))
            episode_return = 0.0
            episode_gamma = 1.0


    return network