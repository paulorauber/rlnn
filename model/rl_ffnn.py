import numpy as np
from collections import deque
from sklearn.utils import check_random_state


def ffnn_batch_update(env, network, transition_buffer, gamma, batch_size,
                      random_state=None):
    random_state = check_random_state(random_state)
    # Batch is a random sample of transitions
    indices = random_state.choice(len(transition_buffer), size=batch_size)
    batch = [transition_buffer[i] for i in indices]

    # Setting up targets
    X = np.zeros((batch_size, env.d_states))
    Y = np.zeros((batch_size, env.n_actions))
    mask = np.zeros((batch_size, env.n_actions))

    for i, transition in enumerate(batch):
        (s, a, next_r, next_s) = transition

        X[i] = s
        mask[i, a] = 1

        if next_s is None:
            Y[i, a] = next_r
        else:
            Y[i, a] = next_r + gamma*np.max(network.predict(next_s))

    network.fit_batch(X, Y, mask)


def q_ffnn(env, network, nepisodes, gamma=0.99, min_epsilon=0.1,
           max_epsilon=1.0, decay_epsilon=0.99, max_queue=8192,
           batch_size=128, verbose=0, random_state=None):
    random_state = check_random_state(random_state)
    transition_buffer = deque(maxlen=max_queue)

    if verbose > 0:
        episode_return = 0.0
        episode_gamma = 1.0

    epsilon = max(min_epsilon, max_epsilon)
    for episode in range(nepisodes):
        if verbose > 0:
            print('Episode {0}.'.format(episode + 1))

        s = env.start()

        if verbose > 1:
            step = 0
            print('Step {0}.'.format(step+1))
        if verbose > 2:
            print(env)

        while not env.ended():
            if random_state.uniform(0, 1) < epsilon:
                a = random_state.choice(env.n_actions)
            else:
                a = np.argmax(network.predict(s))

            next_s, next_r = env.next_state_reward(a)

            if not env.ended():
                transition_buffer.append((s, a, next_r, next_s))
            else:
                transition_buffer.append((s, a, next_r, None))

            s = next_s

            if len(transition_buffer) > batch_size:
                ffnn_batch_update(env, network, transition_buffer, gamma,
                                  batch_size, random_state=random_state)

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