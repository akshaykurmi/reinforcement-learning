import tensorflow as tf


class DQN(tf.keras.Model):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.inp = tf.keras.layers.InputLayer(input_shape=input_shape)
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), activation="relu")
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation="relu")
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.value_fc1 = tf.keras.layers.Dense(1024, activation="relu")
        self.value_fc2 = tf.keras.layers.Dense(1, activation="linear")
        self.advantage_fc1 = tf.keras.layers.Dense(1024, activation="relu")
        self.advantage_fc2 = tf.keras.layers.Dense(num_actions, activation="linear")

    def call(self, inputs, **kwargs):
        x = self.inp(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        value = self.value_fc1(x)
        value = self.value_fc2(value)
        advantage = self.advantage_fc1(x)
        advantage = self.advantage_fc2(advantage)
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        return q_values
