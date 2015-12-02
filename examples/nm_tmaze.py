import numpy as np
from sklearn.utils import check_random_state

from rlnn.examples.tmaze import TMaze

from rlnn.model.rl_lstm import q_lstm
from rlnn.model.lstm import LongShortTermMemoryNetwork


class NonMarkovianTMaze(TMaze):
    def __init__(self, length=4, random_state=None):
        TMaze.__init__(self, length=length, random_state=random_state)
        self.d_states = 3

    def observation(self):
        if self.pos == 0:
            if self.go_up:
                obs = np.array([0, 1, 1])
            else:
                obs = np.array([1, 1, 0])
        elif self.pos == self.length:
            obs = np.array([1, 0, 1])
        else:
            obs = np.array([0, 1, 0])

        return obs


def play(maze, network, episodes, epsilon=0.0, random_state=None):
    random_state = check_random_state(random_state)

    actions = ['u', 'd', 'l', 'r']

    prev_action_reward = np.zeros(maze.n_actions + 1)

    for episode in range(episodes):
        print('Episode {0}.'.format(episode + 1))

        s = maze.start()
        print(maze)

        prev_action_reward.fill(0)
        prev_h_a = np.zeros(network.n_units[1])
        prev_h_s = np.zeros(network.n_units[1])

        while not maze.ended():
            x = np.concatenate([prev_action_reward, s])

            network_state = network.forward_pass(x, prev_h_a, prev_h_s)
            values = network_state['output_layer_activation']

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

            s, r = maze.next_state_reward(a)

            prev_h_a = network_state['activation_output']
            prev_h_s = network_state['activation_cell']

            prev_action_reward.fill(0)
            prev_action_reward[a] = 1
            prev_action_reward[-1] = r

            print(maze)


def main():
    seed = 0

    maze = NonMarkovianTMaze(8, random_state=seed)

    n_input = maze.n_actions + 1 + maze.d_states

    # These settings are overly sensitive
    network = LongShortTermMemoryNetwork(n_input, 64, maze.n_actions,
                                         learning_rate=0.001, mu=0.9,
                                         random_state=seed, verbose=0)

    q_lstm(maze, network, nepisodes=2048, gamma=0.9, min_epsilon=0.1,
           max_epsilon=0.5, decay_epsilon=0.99, max_queue=256, verbose=1,
           random_state=seed)

    play(maze, network, episodes=4, epsilon=0.1, random_state=seed)

if __name__ == "__main__":
    main()
