import tensorflow as tf


class EpsilonGreedyPolicy:
    def __init__(self, env, dqn):
        self.env = env
        self.dqn = dqn
        self.max = 1.0
        self.min = 0.01
        self.decay = 0.00005

    @tf.function
    def take_action(self, state, step):
        explore_probability = self.min + (self.max - self.min) * tf.math.exp(-self.decay * tf.cast(step, tf.float32))
        if explore_probability > tf.random.uniform(shape=()):
            return self.env.action_space.sample()
        state = tf.reshape(state, (1, *state.shape, -1))
        q_preds = self.dqn(state)[0]
        return tf.argmax(q_preds, output_type=tf.int32)
