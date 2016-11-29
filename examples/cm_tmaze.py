import numpy as np
from sklearn.utils import check_random_state

from rlnn.examples.nm_tmaze import NonMarkovianTMaze

from rlnn.model.ffnn import FeedForwardNetwork
from rlnn.model.lstm import LongShortTermMemoryNetwork

from rlnn.model.rl_cm import q_cm


def play(maze, ffnn, lstm, episodes, epsilon=0.0, random_state=None):
    random_state = check_random_state(random_state)

    actions = ['u', 'd', 'l', 'r']

    for episode in range(episodes):
        print('Episode {0}.'.format(episode + 1))

        r = 0.0
        s = maze.start()

        print(maze)

        prev_h_a = np.zeros(lstm.n_units[1])
        prev_h_s = np.zeros(lstm.n_units[1])

        while not maze.ended():
            values = ffnn.predict(np.concatenate([[r], s, prev_h_a, prev_h_s]))

            best = np.argmax(values)
            if random_state.uniform(0, 1) < epsilon:
                a = random_state.choice(maze.n_actions)
            else:
                a = best

            for i in range(maze.n_actions):
                is_best = '*' if i == best else ''
                is_chosen = '$' if i == a else ''
                print('{0}: {1}{2}{3}'.format(actions[i], values[i], is_chosen,
                                              is_best))
            print('')

            action_code = np.zeros(maze.n_actions)
            action_code[a] = 1

            x = np.concatenate([[r], s, action_code])

            lstm_state = lstm.forward_pass(x, prev_h_a, prev_h_s)

            prev_h_a = lstm_state['activation_output']
            prev_h_s = lstm_state['activation_cell']

            s, r = maze.next_state_reward(a)

            print(maze)


def main():
    seed = 0

    maze = NonMarkovianTMaze(4, random_state=seed)

    d_input = 1 + maze.d_states + maze.n_actions
    d_output = 1 + maze.d_states
    lstm = LongShortTermMemoryNetwork(d_input, 32, d_output,
                                      learning_rate=0.001, mu=0.9,
                                      random_state=seed, verbose=0)

    d_input = 1 + maze.d_states + 2*lstm.n_units[1]
    d_output = maze.n_actions
    ffnn = FeedForwardNetwork([d_input, 64, d_output], learning_rate=0.01,
                              lmbda=0.0, random_state=seed)

    # These settings are overly sensitive
    q_cm(maze, ffnn, lstm, n_episodes=2048, gamma=0.9, min_epsilon=0.1,
         max_epsilon=0.5, decay_epsilon=0.99, max_queue=256,
         model_updates_per_episode=32, control_updates_per_transition=2,
         verbose=1, random_state=seed)

    play(maze, ffnn, lstm, episodes=10, epsilon=0.1, random_state=seed)


if __name__ == "__main__":
    main()
