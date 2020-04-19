import numpy as np
from colr import color
from gym import Env
from gym.spaces import Discrete, Box


class Env2048(Env):
    def __init__(self):
        self.metadata = {"render.modes": ["human", "ansi"]}
        self.reward_range = (0, 2 ** 16)
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=2 ** 16, shape=(4, 4), dtype=np.uint16)
        self.board = Board(4, 4)

    def step(self, action):
        is_move_valid, total_merged = self.board.move(action)
        reward = total_merged if is_move_valid else -2  # TODO: how much negative reward here?
        done = self.board.is_game_over()
        info = {"score": self.board.score}
        return self.board.grid, reward, done, info

    def reset(self):
        self.board.reset()
        return self.board.grid

    def render(self, mode="human"):
        print(self.board.draw())


class Board:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.score, self.grid = None, None
        self.DIRECTIONS = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.PALETTE = {
            0: ("000000", "000000"),
            2 ** 1: ("222222", "eee4da"),
            2 ** 2: ("222222", "ede0c8"),
            2 ** 3: ("222222", "f2b179"),
            2 ** 4: ("222222", "f59563"),
            2 ** 5: ("222222", "f67c5f"),
            2 ** 6: ("222222", "f65e3b"),
            2 ** 7: ("222222", "edcf72"),
            2 ** 8: ("222222", "edcc61"),
            2 ** 9: ("222222", "edc850"),
            2 ** 10: ("222222", "edc53f"),
            2 ** 11: ("222222", "edc22e"),
            2 ** 12: ("f9f6f2", "3c3a32"),
            2 ** 13: ("f9f6f2", "3c3a32"),
            2 ** 14: ("f9f6f2", "3c3a32"),
            2 ** 15: ("f9f6f2", "3c3a32"),
            2 ** 16: ("f9f6f2", "3c3a32")
        }
        self.reset()

    def _add_random_tile(self):
        zero_indices = np.argwhere(self.grid == 0)
        random_index = zero_indices[np.random.choice(zero_indices.shape[0], 1)[0]]
        self.grid[tuple(random_index)] = 4 if np.random.uniform() > 0.9 else 2

    def _move(self, action, current_grid):
        def mirror(grid):
            return np.flip(grid, axis=1)

        def transpose(grid):
            return np.transpose(grid)

        def compress(row):
            non_zero_values = np.compress(row != 0, row)
            return np.pad(non_zero_values, (0, row.size - non_zero_values.size), "constant")

        def merge(row):
            nonlocal total_merged
            row = np.pad(row, (0, 1), "constant")
            result = []
            for i in range(row.size - 1):
                if row[i] == row[i + 1]:
                    result.append(row[i] + row[i + 1])
                    total_merged += row[i] + row[i + 1]
                    row[i + 1] = 0
                else:
                    result.append(row[i])
            return np.array(result)

        moves = {
            "UP": lambda grid: transpose(moves["LEFT"](transpose(grid))),
            "DOWN": lambda grid: transpose(moves["RIGHT"](transpose(grid))),
            "LEFT": lambda grid: np.array([compress(merge(compress(row))) for row in grid]),
            "RIGHT": lambda grid: mirror(moves["LEFT"](mirror(grid))),
        }

        total_merged = 0
        move = moves.get(self.DIRECTIONS.get(action))
        return move(current_grid), total_merged

    def reset(self):
        self.score = 0
        self.grid = np.zeros((self.width, self.height), dtype=np.uint16)
        self._add_random_tile()
        self._add_random_tile()

    def move(self, action):
        new_grid, total_merged = self._move(action, self.grid)
        self.score += total_merged
        if np.array_equal(new_grid, self.grid):
            return False, total_merged
        self.grid = new_grid
        self._add_random_tile()
        return True, total_merged

    def draw(self):
        result = f"SCORE : {self.score}\n"
        result += "┌" + ("────────┬" * self.width)[:-1] + "┐\n"
        for i, row in enumerate(self.grid):
            result += "|" + "|".join([
                color(" " * 8, fore=self.PALETTE[cell][0], back=self.PALETTE[cell][1], style="bold")
                for cell in row
            ]) + "|\n"
            result += "|" + "|".join([
                color(str(cell if cell != 0 else "").center(8), fore=self.PALETTE[cell][0], back=self.PALETTE[cell][1],
                      style="bold")
                for cell in row
            ]) + "|\n"
            result += "|" + "|".join([
                color(" " * 8, fore=self.PALETTE[cell][0], back=self.PALETTE[cell][1], style="bold")
                for cell in row
            ]) + "|\n"
            if i + 1 < self.grid.shape[0]:
                result += "├" + ("────────┼" * self.width)[:-1] + "┤\n"
        result += "└" + ("────────┴" * self.width)[:-1] + "┘\n"
        return result

    def is_game_over(self):
        return all([np.array_equal(self._move(action, self.grid)[0], self.grid) for action in self.DIRECTIONS])
