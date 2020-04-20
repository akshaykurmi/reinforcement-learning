import tensorflow as tf


class DQN(tf.keras.Model):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), activation="relu", input_shape=input_shape)
        self.flatten = tf.keras.layers.Flatten()
        self.q_values = tf.keras.layers.Dense(num_actions, activation="linear")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        return self.q_values(x)
