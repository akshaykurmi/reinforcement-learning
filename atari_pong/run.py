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
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(5, 5), activation="relu",
                                            input_shape=(160, 160, 4))
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu",
                                            input_shape=(160, 160, 4))
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(units=output_size, activation="softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


class PongAgent:
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.env = gym.make("Pong-v0")

    def _init(self, ckpt_dir):
        model = PolicyNetwork(self.env.action_space.n)
        optimizer = tf.keras.optimizers.Adam()
        lossfn = tf.keras.losses.CategoricalCrossentropy()
        step = tf.Variable(0, dtype=tf.int64)
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, step=step)
        ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1, keep_checkpoint_every_n_hours=1)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        return model, optimizer, lossfn, ckpt, ckpt_manager

    def test(self, ckpt_dir):
        model, _, _, _, _ = self._init(ckpt_dir)
        observation = self.env.reset()
        self.env.render()
        done = False
        state = np.zeros((160, 160, 4), dtype=np.float32)
        while not done:
            sleep(0.005)
            state = np.dstack((state, self._preprocess_frame(observation)))
            state = state[:, :, 1:]
            predictions = model.predict_on_batch(state.reshape((1, 160, 160, 4)))[0, :].numpy()
            predictions /= np.sum(predictions)
            action = np.random.choice(self.env.action_space.n, p=predictions)
            observation, reward, done, info = self.env.step(action)
            self.env.render()
        self.env.close()

    def train(self, episodes, ckpt_dir, logs_dir):
        model, optimizer, lossfn, ckpt, ckpt_manager = self._init(ckpt_dir)
        summary_writer = tf.summary.create_file_writer(logs_dir)
        with tqdm(total=episodes, desc="Episode", unit="episode") as pbar:
            pbar.update(ckpt.step.numpy())
            for _ in range(episodes - ckpt.step.numpy()):
                X, y, episode_reward = self._sample_episode(model)
                loss = self._train_step(X, y, model, optimizer, lossfn)
                ckpt_manager.save()
                with summary_writer.as_default():
                    tf.summary.scalar("reward", episode_reward, step=ckpt.step)
                    tf.summary.scalar("loss", loss, step=ckpt.step)
                ckpt.step.assign_add(1)
                pbar.update(1)

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, X, y, model, optimizer, lossfn):
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = lossfn(y, y_pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def _sample_episode(self, model):
        states, rewards, action_probs, action_prob_grads = [], [], [], []
        observation = self.env.reset()
        done = False
        episode_reward = 0
        state = np.zeros((160, 160, 4), dtype=np.float32)
        while not done:
            state = np.dstack((state, self._preprocess_frame(observation)))
            state = state[:, :, 1:]
            predictions = model.predict_on_batch(state.reshape((1, 160, 160, 4)))[0, :].numpy()
            predictions /= np.sum(predictions)
            sampled_action = np.random.choice(self.env.action_space.n, p=predictions)
            action = np.zeros(self.env.action_space.n)
            action[sampled_action] = 1
            observation, reward, done, info = self.env.step(sampled_action)
            states.append(state)
            rewards.append(reward)
            action_probs.append(predictions)
            action_prob_grads.append(action - predictions)
            episode_reward += reward
        self.env.close()
        y = np.vstack(action_probs) + self.alpha * (
                np.vstack(self._discount_rewards(rewards)) * np.vstack(action_prob_grads))
        return np.array(states), y, episode_reward

    @staticmethod
    def _preprocess_frame(frame):
        frame = frame[34:194]
        frame = frame[:, :, 0]
        frame[frame == 144] = 0
        frame[frame != 0] = 1
        return frame.astype(np.float32)

    def _discount_rewards(self, rewards):
        discounted = []
        exponent = 0
        current_reward = 0
        for r in reversed(rewards):
            if r != 0:
                current_reward = r
                exponent = 0
            discounted.append(current_reward * (self.gamma ** exponent))
            exponent += 1
        discounted = list(reversed(discounted))
        discounted -= np.mean(discounted)
        discounted /= np.std(discounted)
        return discounted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=["train", "test"], required=True)
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(__file__)
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")

    agent = PongAgent(0.1, 0.99)

    if args.mode == "train":
        agent.train(10000, CKPT_DIR, LOGS_DIR)

    if args.mode == "test":
        agent.test(CKPT_DIR)
