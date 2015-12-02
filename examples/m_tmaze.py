import numpy as np
from sklearn.utils import check_random_state

from rlnn.examples.tmaze import TMaze

from rlnn.model.rl_ffnn import q_ffnn
from rlnn.model.ffnn import FeedForwardNetwork


class MarkovianTMaze(TMaze):
    def __init__(self, length=4, random_state=None):
        TMaze.__init__(self, length=length, random_state=random_state)
        self.d_states = self.length + 4

    def observation(self):
        obs = np.zeros(self.d_states, dtype=np.float)
        obs[self.pos] = 1
        obs[-1] = self.go_up

        return obs


def play(maze, network, episodes, epsilon=0.0, random_state=None):
    random_state = check_random_state(random_state)

    actions = ['u', 'd', 'l', 'r']

    for episode in range(episodes):
        print('Episode {0}.'.format(episode + 1))

        s = maze.start()

        print(maze)

        while not maze.ended():
            values = network.predict(s)

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

            s, _ = maze.next_state_reward(a)
            print(maze)


def main():
    maze = MarkovianTMaze(32, random_state=0)

    network = FeedForwardNetwork([maze.d_states, 64, maze.n_actions],
                                 learning_rate=0.1, lmbda=0.0,
                                 random_state=0)

    q_ffnn(maze, network, nepisodes=512, gamma=0.98, min_epsilon=0.01,
           max_epsilon=1.0, decay_epsilon=0.98, max_queue=1024, batch_size=32,
           verbose=1, random_state=0)

    play(maze, network, episodes=10, epsilon=0.1, random_state=0)


if __name__ == "__main__":
    main()
