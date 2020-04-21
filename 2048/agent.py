import numpy as np
import tensorflow as tf
from env import Env2048
from model import DQN
from per import PrioritizedExperienceReplay
from policy import EpsilonGreedyPolicy
from tqdm import tqdm


class Agent:
    def __init__(self, args):
        self.env = Env2048()
        self.args = args

    def _init(self):
        dqn_online = DQN(input_shape=self.env.observation_space.shape, num_actions=self.env.action_space.n)
        dqn_target = DQN(input_shape=self.env.observation_space.shape, num_actions=self.env.action_space.n)
        dqn_target.set_weights(dqn_online.get_weights())
        action_step = tf.Variable(0, dtype=tf.int64, trainable=False)
        episode_step = tf.Variable(0, dtype=tf.int64, trainable=False)
        optimizer = tf.keras.optimizers.Adam(self.args.learning_rate)
        ckpt = tf.train.Checkpoint(dqn_online=dqn_online, dqn_target=dqn_target, optimizer=optimizer,
                                   action_step=action_step, episode_step=episode_step)
        ckpt_manager = tf.train.CheckpointManager(ckpt, self.args.ckpt_dir, max_to_keep=1)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        memory = PrioritizedExperienceReplay(self.args.per_capacity, self.args.per_initial_size, self.args.per_epsilon,
                                             self.args.per_alpha, self.args.per_beta, self.args.per_beta_annealing_rate,
                                             self.args.per_max_td_error, self.args.ckpt_dir)
        memory.load_or_instantiate(self.env)
        return dqn_online, dqn_target, optimizer, memory, ckpt, ckpt_manager

    def test(self):
        pass

    def train(self):
        dqn_online, dqn_target, optimizer, memory, ckpt, ckpt_manager = self._init()
        policy = EpsilonGreedyPolicy(self.env, dqn_online, self.args.egp_epsilon_max, self.args.egp_epsilon_min,
                                     self.args.egp_epsilon_decay)
        summary_writer = tf.summary.create_file_writer(self.args.log_dir)

        with tqdm(total=self.args.max_action_steps, desc="Training", unit="action") as pbar:
            pbar.update(ckpt.action_step.numpy())
            observation = self.env.reset()
            episode_reward, episode_score = 0, 0
            for i in range(ckpt.action_step.numpy(), self.args.max_action_steps + 1):
                state = self.env.preprocess(observation)
                action, explore_probability = policy.take_action(state, ckpt.action_step)
                observation, reward, done, info = self.env.step(action.numpy())
                next_state = self.env.preprocess(observation)
                transition = (state, action, reward, next_state, done)
                memory.add(transition)
                episode_reward += reward
                episode_score += info["score"]

                if done:
                    observation = self.env.reset()
                    with summary_writer.as_default(), tf.name_scope("episode stats"):
                        tf.summary.scalar("reward", episode_reward, step=ckpt.episode_step)
                        tf.summary.scalar("score", episode_score, step=ckpt.episode_step)
                        tf.summary.scalar("max_tile", info["max_tile"], step=ckpt.episode_step)
                    episode_reward, episode_score = 0, 0
                    ckpt.episode_step.assign_add(1)

                if i % self.args.train_steps == 0:
                    sample_indices, samples, importance_sampling_weights = memory.sample(self.args.batch_size)

                    states = tf.convert_to_tensor([s[0] for s in samples], dtype=tf.float32)
                    actions = tf.convert_to_tensor([s[1] for s in samples], dtype=tf.int32)
                    rewards = tf.convert_to_tensor([s[2] for s in samples], dtype=tf.float32)
                    next_states = tf.convert_to_tensor([s[3] for s in samples], dtype=tf.float32)
                    dones = tf.convert_to_tensor([float(s[4]) for s in samples], dtype=tf.float32)
                    importance_sampling_weights = tf.convert_to_tensor(importance_sampling_weights, dtype=tf.float32)

                    loss, td_errors = self._train_step(states, actions, rewards, next_states, dones,
                                                       importance_sampling_weights,
                                                       dqn_online, dqn_target, optimizer, self.args.gamma)
                    memory.update_priorities(sample_indices, td_errors)
                    with summary_writer.as_default(), tf.name_scope("training stats"):
                        tf.summary.scalar("loss", np.mean(loss), step=ckpt.action_step)
                        tf.summary.scalar("td_errors", np.mean(td_errors), step=ckpt.action_step)
                        tf.summary.scalar("explore_probability", explore_probability, step=ckpt.action_step)

                if i % self.args.update_dqn_target_steps == 0:
                    dqn_target.set_weights(dqn_online.get_weights())

                if i % self.args.save_steps == 0:
                    ckpt_manager.save()
                    memory.save()

                ckpt.action_step.assign_add(1)
                pbar.update(1)

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones, importance_sampling_weights,
                    dqn_online, dqn_target, optimizer, gamma):
        states = tf.expand_dims(states, axis=3)
        next_states = tf.expand_dims(next_states, axis=3)

        batch_size = states.shape[0]

        q_online_next_states = dqn_online(next_states)
        q_target_next_states = dqn_target(next_states)

        q_targets = rewards
        q_targets += gamma * dones * tf.gather_nd(
            q_target_next_states,
            tf.stack([
                tf.range(batch_size, dtype=tf.int32),
                tf.argmax(q_online_next_states, axis=1, output_type=tf.int32)
            ], axis=1)
        )
        actions = tf.one_hot(actions, depth=self.env.action_space.n, dtype=tf.float32)

        with tf.GradientTape() as tape:
            q_preds = dqn_online(states)
            q_preds = tf.reduce_sum(q_preds * actions, axis=1)
            td_errors = tf.math.abs(q_targets - q_preds)
            loss = importance_sampling_weights * tf.math.squared_difference(q_targets, q_preds)
        gradients = tape.gradient(loss, dqn_online.trainable_variables)
        optimizer.apply_gradients(zip(gradients, dqn_online.trainable_variables))
        return loss, td_errors
