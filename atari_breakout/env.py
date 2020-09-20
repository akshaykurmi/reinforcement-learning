import gym
import numpy as np


class BreakoutEnv:
    def __init__(self, num_stacked_frames=4):
        self.env = gym.make("BreakoutDeterministic-v4")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.state_shape = (80, 80)
        self.lives = self.env.ale.lives()
        self.state = None
        self.num_stacked_frames = num_stacked_frames

    def reset(self):
        frame = self.env.reset()
        self.lives = self.env.ale.lives()
        self.state = np.repeat(self.preprocess_frame(frame), self.num_stacked_frames, axis=2)
        return self.state

    def step(self, action):
        frame, reward, done, _ = self.env.step(action)
        life_lost = done or (self.env.ale.lives() < self.lives)
        self.lives = self.env.ale.lives()
        self.state = np.append(self.state[:, :, 1:], self.preprocess_frame(frame), axis=2)
        return self.state, reward, done, life_lost

    def close(self):
        self.env.close()

    @staticmethod
    def preprocess_frame(frame):
        frame = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
        frame = frame[34:194, :]
        frame = frame[::2, ::2]
        return frame
