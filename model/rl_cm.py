from collections import deque

import numpy as np
from sklearn.utils import check_random_state


def ffnn_control_update(env, ffnn, lstm, episode_buffer, gamma,
                        control_updates_per_transition, verbose=0,
                        random_state=None):
    episodes = random_state.permutation(len(episode_buffer))
    episodes = [i for i in episodes if len(episode_buffer[i]) > 1]
    episodes = episodes[0: control_updates_per_transition]

    Xb, Yb, mask = [], [], []

    for i in episodes:
        X = np.array(episode_buffer[i][0: -1])
        T = len(X)

        states = []
        prev_h_a = np.zeros(lstm.n_units[1])
        prev_h_s = np.zeros(lstm.n_units[1])

        for t in range(T):
            state = lstm.forward_pass(X[t], prev_h_a, prev_h_s)

            prev_h_a = state['activation_output']
            prev_h_s = state['activation_cell']

            states.append(state)

        for t in range(T):
            r = X[t, 0]
            s = X[t, 1: env.d_states + 1]

            action_code = X[t, env.d_states + 1:]

            if t == 0:
                prev_h_a = np.zeros(lstm.n_units[1])
                prev_h_s = np.zeros(lstm.n_units[1])
            else:
                prev_h_a = states[t - 1]['activation_output']
                prev_h_s = states[t - 1]['activation_cell']

            x = np.concatenate([[r], s, prev_h_a, prev_h_s])

            next_rs = X[t + 1] if t < T - 1 else episode_buffer[i][-1]
            next_r = next_rs[0]
            next_s = next_rs[1: env.d_states + 1]

            next_action_code = next_rs[env.d_states + 1:]

            if np.allclose(next_action_code, 0):
                v = next_r
            else:
                curr_h_a = states[t]['activation_output']
                curr_h_s = states[t]['activation_cell']

                xprime = np.concatenate([[next_r], next_s, curr_h_a, curr_h_s])

                v = next_r + gamma*np.max(ffnn.predict(xprime))

            Xb.append(x)
            Yb.append(action_code*v)
            mask.append(action_code)

    Xb, Yb, mask = np.array(Xb), np.array(Yb), np.array(mask)

    ffnn.fit_batch(Xb, Yb, mask)

    if len(episodes) > 0 and verbose > 1:
        error = np.linalg.norm(Yb - ffnn.predict_batch(Xb)*mask)/len(episodes)
        print('Controller error: {0}.'.format(error))


def lstm_model_update(env, lstm, episode_buffer, model_updates_per_episode,
                      verbose=0, random_state=None):
    episodes = random_state.permutation(len(episode_buffer))
    episodes = episodes[0: model_updates_per_episode]

    if verbose > 0:
        error = 0.0

    for i in episodes:
        X = np.array(episode_buffer[i][0: -1])
        T = len(X)

        Y = np.zeros((T, 1 + env.d_states))
        for t in range(T - 1):
            Y[t] = X[t + 1, 0: 1 + env.d_states]
        Y[T - 1] = episode_buffer[i][-1][0: 1 + env.d_states]

        lstm.fit(X, Y, np.ones(Y.shape))

        if verbose > 0:
            error += np.linalg.norm(Y - lstm.predict(X))

    if len(episodes) > 0 and verbose > 0:
        print('Model error: {0}.'.format(error / len(episodes)))


def q_cm(env, ffnn, lstm, n_episodes=1024, gamma=0.98, min_epsilon=0.1,
         max_epsilon=0.5, decay_epsilon=0.99, max_queue=256,
         model_updates_per_episode=32, control_updates_per_transition=2,
         verbose=1, random_state=None):
    random_state = check_random_state(random_state)
    episode_buffer = deque(maxlen=max_queue)

    if verbose > 0:
        episode_return = 0.0
        episode_gamma = 1.0

    epsilon = max(min_epsilon, max_epsilon)
    for episode in range(n_episodes):
        if verbose > 0:
            print('Episode {0}.'.format(episode + 1))

        lstm_model_update(env, lstm, episode_buffer, model_updates_per_episode,
                          verbose=verbose, random_state=random_state)

        episode_buffer.append([])

        r = 0.0
        s = env.start()

        prev_h_a = np.zeros(lstm.n_units[1])
        prev_h_s = np.zeros(lstm.n_units[1])

        if verbose > 1:
            step = 0
            print('Step {0}.'.format(step + 1))
        if verbose > 2:
            print(env)

        while not env.ended():
            if random_state.uniform(0, 1) < epsilon:
                a = random_state.choice(env.n_actions)
            else:
                x = np.concatenate([[r], s, prev_h_a, prev_h_s])
                a = np.argmax(ffnn.predict(x))

            action_code = np.zeros(env.n_actions)
            action_code[a] = 1

            x = np.concatenate([[r], s, action_code])

            episode_buffer[-1].append(x)

            lstm_state = lstm.forward_pass(x, prev_h_a, prev_h_s)

            prev_h_a = lstm_state['activation_output']
            prev_h_s = lstm_state['activation_cell']

            s, r = env.next_state_reward(a)

            ffnn_control_update(env, ffnn, lstm, episode_buffer, gamma,
                                control_updates_per_transition,
                                verbose=verbose, random_state=random_state)

            if verbose > 0:
                episode_return += episode_gamma*r
                episode_gamma *= gamma
            if verbose > 1:
                step += 1
                print('Step {0}.'.format(step + 1))
            if verbose > 2:
                print(env)

        x = np.concatenate([[r], s, np.zeros(env.n_actions)])

        episode_buffer[-1].append(x)

        epsilon = max(min_epsilon, epsilon*decay_epsilon)

        if verbose > 0:
            print('Return: {0}.'.format(episode_return))
            episode_return = 0.0
            episode_gamma = 1.0


    return ffnn, lstm
