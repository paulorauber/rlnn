from sklearn.utils import check_random_state


class TMaze:
    def __init__(self, length=4, random_state=None):
        if length < 0:
            raise Exception('Invalid corridor length')

        self.length = length
        self.n_actions = 4
        self.random_state = check_random_state(random_state)

    def start(self):
        self.go_up = self.random_state.randint(2)
        self.pos = 0

        return self.observation()

    def move(self, a):
        """a: up, down, left, right"""
        if a < 0 or a > 3:
            raise Exception('Invalid action')

        if 0 <= self.pos <= self.length - 1:
            # On corridor
            if a == 2 and self.pos > 0:
                # Valid left
                self.pos -= 1
            elif a == 3:
                # Right
                self.pos += 1
        elif self.pos == self.length:
            # On intersection
            if a == 2 and self.pos > 0:
                # Valid left
                self.pos -= 1
            elif a == 0:
                # Up
                self.pos += 1
            elif a == 1:
                # Down
                self.pos += 2

    def next_state_reward(self, a):
        self.move(a)

        if self.won():
            return self.observation(), 100.

        return self.observation(), 0.0

    def ended(self):
        return self.pos > self.length

    def won(self):
        return self.ended() and (self.go_up == (self.pos == self.length + 1))

    def observation(self):
        return None

    def __repr__(self):
        down, up = 'G', 'T'
        if self.go_up:
            up, down = down, up

        l1 = ['#']*(self.length + 3)
        l1[-2] = up
        if self.pos == self.length + 1:
            l1[-2] = '@'

        l2 = ['#'] + ['.']*(self.length + 1) + ['#']
        if self.pos <= self.length:
            l2[self.pos + 1] = '@'

        l3 = ['#']*(self.length + 3)
        l3[-2] = down
        if self.pos == self.length + 2:
            l3[-2] = '@'

        return ''.join(l1) + '\n' + ''.join(l2) + '\n' + ''.join(l3) + '\n'


def play(maze):
    maze.start()

    udlr = ['w', 's', 'a', 'd']
    while not maze.ended():
        print(maze)

        c = raw_input('Move:')
        if c not in udlr:
            break

        maze.move(udlr.index(c))

    print('Won' if maze.won() else 'Lost')


def main():
    maze = TMaze(8)
    play(maze)


if __name__ == "__main__":
    main()
