import tensorflow as tf


class EpsilonGreedyPolicy:
    def __init__(self, env, dqn, epsilon_max, epsilon_min, epsilon_decay):
        self.env = env
        self.dqn = dqn
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def take_action(self, state, step):
        explore_probability = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * tf.math.exp(
            -self.epsilon_decay * tf.cast(step, tf.float32))
        if explore_probability > tf.random.uniform(shape=()):
            return tf.constant(self.env.action_space.sample(), dtype=tf.int32), explore_probability
        state = tf.reshape(state, (1, *state.shape, -1))
        q_preds = self.dqn(state)[0]
        return tf.argmax(q_preds, output_type=tf.int32), explore_probability
