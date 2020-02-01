import argparse
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Agent:
    def __init__(self):
        self.env = gym.make("Taxi-v3")
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def train(self, episodes, max_steps_per_episode, learning_rate, discount_factor,
              epsilon_max, epsilon_min, epsilon_decay, ckpt_fp):
        episode_rewards = []
        epsilon = epsilon_max
        for episode in tqdm(range(episodes)):
            state = self.env.reset()
            episode_reward = 0
            for step in range(max_steps_per_episode):
                action = np.argmax(
                    self.Q[state, :]) if np.random.uniform() > epsilon else self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                self.Q[state, action] += learning_rate * (
                            reward + discount_factor * np.max(self.Q[next_state, :]) - self.Q[state, action])
                state = next_state
                episode_reward += reward
                if done:
                    episode_rewards.append(episode_reward)
                    break
            epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-epsilon_decay * episode)
        print(len(episode_rewards))
        np.savez(ckpt_fp, Q=self.Q)
        self._plot_rewards(episode_rewards)

    @staticmethod
    def _plot_rewards(episode_rewards):
        plt.plot(episode_rewards)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.grid()
        plt.show()

    def test(self, ckpt_fp):
        self.Q = np.load(ckpt_fp)["Q"]
        state = self.env.reset()
        done = False
        while not done:
            self.env.render()
            action = np.argmax(self.Q[state, :])
            state, _, done, _ = self.env.step(action)
            if done:
                self.env.render()
                break
        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=["train", "test"], required=True, help="Train or test the agent?")
    parser.add_argument('--ckpt_dir', required=True, help="Name of checkpoint directory")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(__file__)
    CKPT_FP = os.path.join(BASE_DIR, args.ckpt_dir, "Q_values.npz")
    os.makedirs(os.path.dirname(CKPT_FP), exist_ok=True)

    agent = Agent()

    if args.mode == "train":
        agent.train(episodes=2000, max_steps_per_episode=100, learning_rate=0.7, discount_factor=0.95,
                    epsilon_max=1.0, epsilon_min=0.01, epsilon_decay=0.01, ckpt_fp=CKPT_FP)

    if args.mode == "test":
        agent.test(CKPT_FP)
