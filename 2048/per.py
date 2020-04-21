import os
import pickle

import numpy as np
from tqdm import tqdm


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.empty(capacity, dtype=object)
        self.head = 0

    @property
    def total_priority(self):
        return self.tree[0]

    @property
    def max_priority(self):
        return np.max(self.tree[-self.capacity:])

    @property
    def min_priority(self):
        return np.min(self.tree[-self.capacity:])

    def _tree_to_data_index(self, i):
        return i - self.capacity + 1

    def _data_to_tree_index(self, i):
        return i + self.capacity - 1

    def add(self, priority, data):
        tree_index = self._data_to_tree_index(self.head)
        self.update_priority(tree_index, priority)
        self.data[self.head] = data
        self.head += 1
        if self.head >= self.capacity:
            self.head = 0

    def update_priority(self, tree_index, priority):
        delta = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += delta

    def get_leaf(self, value):
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = parent
                break
            else:
                if value <= self.tree[left]:
                    parent = left
                else:
                    value -= self.tree[left]
                    parent = right
        data_index = self._tree_to_data_index(leaf)
        return leaf, self.tree[leaf], self.data[data_index]


class PrioritizedExperienceReplay:
    def __init__(self, capacity, initial_size, epsilon, alpha, beta, beta_annealing_rate,
                 max_td_error, ckpt_dir):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.epsilon = epsilon
        self.initial_size = initial_size
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing_rate = beta_annealing_rate
        self.max_td_error = max_td_error
        self.ckpt_dir = ckpt_dir

    def add(self, transition):
        max_priority = self.tree.max_priority
        if max_priority == 0:
            max_priority = self.max_td_error
        self.tree.add(max_priority, transition)

    def sample(self, batch_size):
        self.beta = np.min([1., self.beta + self.beta_annealing_rate])
        priority_segment = self.tree.total_priority / batch_size
        min_probability = self.tree.min_priority / self.tree.total_priority
        max_weight = (min_probability * batch_size) ** (-self.beta)

        samples, sample_indices, importance_sampling_weights = [], [], []
        for i in range(batch_size):
            value = np.random.uniform(priority_segment * i, priority_segment * (i + 1))
            index, priority, transition = self.tree.get_leaf(value)
            sample_probability = priority / self.tree.total_priority
            importance_sampling_weights.append(((batch_size * sample_probability) ** -self.beta) / max_weight)
            sample_indices.append(index)
            samples.append(transition)
        return sample_indices, samples, importance_sampling_weights

    def update_priorities(self, tree_indices, td_errors):
        td_errors += self.epsilon
        clipped_errors = np.minimum(td_errors, self.max_td_error)
        priorities = clipped_errors ** self.alpha
        for tree_index, priority in zip(tree_indices, priorities):
            self.tree.update_priority(tree_index, priority)

    def load_or_instantiate(self, env):
        if os.path.exists(os.path.join(self.ckpt_dir, "memory.pkl")):
            self.load()
            return
        state = env.reset()
        for _ in tqdm(range(self.initial_size), desc="Initializing replay memory", unit="transition"):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            transition = (state, action, reward, next_state, done)
            self.add(transition)
            state = next_state
            if done:
                state = env.reset()

    def load(self):
        with open(os.path.join(self.ckpt_dir, "memory.pkl"), "rb") as f:
            self.tree = pickle.load(f)

    def save(self):
        with open(os.path.join(self.ckpt_dir, "memory.pkl"), "wb") as f:
            pickle.dump(self.tree, f)
