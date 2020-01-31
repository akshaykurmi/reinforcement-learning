import argparse
import os
from time import sleep

import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class PolicyNetwork(tf.keras.Model):
    def __init__(self, output_size):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=20, activation="relu", input_shape=(4,))
        self.dense2 = tf.keras.layers.Dense(units=10, activation="relu")
        self.dense3 = tf.keras.layers.Dense(units=output_size, activation="softmax")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)


class Agent:
    def __init__(self, learning_rate, discount_factor):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.env = gym.make("CartPole-v0")

    def _init(self, ckpt_dir):
        model = PolicyNetwork(self.env.action_space.n)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        step = tf.Variable(0, dtype=tf.int64)
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, step=step)
        ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        return model, optimizer, ckpt, ckpt_manager

    def test(self, ckpt_dir):
        model, optimizer, ckpt, ckpt_manager = self._init(ckpt_dir)
        state = self.env.reset()
        self.env.render()
        done = False
        while not done:
            sleep(0.005)
            action_probs = model(state.reshape(1, -1)).numpy().flatten()
            state, reward, done, info = self.env.step(np.argmax(action_probs))
            self.env.render()
        self.env.close()

    def train(self, episodes, ckpt_dir, log_dir):
        model, optimizer, ckpt, ckpt_manager = self._init(ckpt_dir)
        summary_writer = tf.summary.create_file_writer(log_dir)
        with tqdm(total=episodes, desc="Episode", unit="episode") as pbar:
            pbar.update(ckpt.step.numpy())
            for _ in range(episodes - ckpt.step.numpy()):
                states, actions, discounted_rewards, episode_reward = self._sample_episode(model)
                loss = self._train_step(states, actions, discounted_rewards, model, optimizer)
                ckpt_manager.save()
                with summary_writer.as_default():
                    tf.summary.scalar("reward", episode_reward, step=ckpt.step)
                    tf.summary.scalar("loss", loss, step=ckpt.step)
                ckpt.step.assign_add(1)
                pbar.update(1)

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, states, actions, discounted_rewards, model, optimizer):
        with tf.GradientTape() as tape:
            action_probs = model(states)
            negative_log_likelihood = tf.math.negative(tf.math.log(tf.gather_nd(action_probs, actions)))
            weighted_negative_log_likelihood = tf.multiply(negative_log_likelihood, discounted_rewards)
            loss = tf.reduce_sum(weighted_negative_log_likelihood)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def _sample_episode(self, model):
        states, actions, rewards = [], [], []
        state = self.env.reset()
        done = False
        episode_reward = 0
        while not done:
            action_probs = model(state.reshape(1, -1)).numpy().flatten()
            sampled_action = np.random.choice(self.env.action_space.n, p=action_probs)
            states.append(state)
            state, reward, done, _ = self.env.step(sampled_action)
            actions.append(sampled_action)
            rewards.append(reward)
            episode_reward += reward
        self.env.close()
        return np.array(states, dtype=np.float32), \
               np.array(list(enumerate(actions)), dtype=np.int32), \
               np.array(self._discount_rewards(rewards)).astype(dtype=np.float32), episode_reward

    def _discount_rewards(self, rewards):
        discounted = []
        for t in range(len(rewards)):
            expected_return, exponent = 0, 0
            for r in rewards[t:]:
                expected_return += r * (self.discount_factor ** exponent)
                exponent += 1
            discounted.append(expected_return)
        discounted -= np.mean(discounted)
        discounted /= np.std(discounted)
        return discounted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=["train", "test"], required=True, help="Train or test the agent?")
    parser.add_argument('--ckpt_dir', required=True, help="Name of checkpoint directory")
    parser.add_argument('--log_dir', required=True, help="Name of log directory")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(__file__)
    CKPT_DIR = os.path.join(BASE_DIR, args.ckpt_dir)
    LOG_DIR = os.path.join(BASE_DIR, args.log_dir)

    agent = Agent(learning_rate=3e-4, discount_factor=0.9)

    if args.mode == "train":
        agent.train(5000, CKPT_DIR, LOG_DIR)

    if args.mode == "test":
        agent.test(CKPT_DIR)
