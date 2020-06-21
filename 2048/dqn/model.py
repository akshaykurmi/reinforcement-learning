import tensorflow as tf


class MixtureOfConvolutions(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv21 = tf.keras.layers.Conv2D(filters=512, kernel_size=(2, 1), activation="relu", padding="same")
        self.conv12 = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 2), activation="relu", padding="same")
        self.conv31 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 1), activation="relu", padding="same")
        self.conv13 = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 3), activation="relu", padding="same")
        self.conv41 = tf.keras.layers.Conv2D(filters=512, kernel_size=(4, 1), activation="relu", padding="same")
        self.conv14 = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 4), activation="relu", padding="same")
        self.conv32 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 2), activation="relu", padding="same")
        self.conv23 = tf.keras.layers.Conv2D(filters=512, kernel_size=(2, 3), activation="relu", padding="same")
        self.conv22 = tf.keras.layers.Conv2D(filters=512, kernel_size=(2, 2), activation="relu", padding="same")
        self.conv33 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same")

    def call(self, inputs):
        x21 = self.conv21(inputs)
        x12 = self.conv12(inputs)
        x31 = self.conv31(inputs)
        x13 = self.conv13(inputs)
        x41 = self.conv41(inputs)
        x14 = self.conv14(inputs)
        x32 = self.conv32(inputs)
        x23 = self.conv23(inputs)
        x22 = self.conv22(inputs)
        x33 = self.conv33(inputs)
        x = tf.concat([x21, x12, x31, x13, x41, x14, x32, x23, x22, x33], axis=-1)
        return x


class DQN(tf.keras.Model):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.inp = tf.keras.layers.InputLayer(input_shape=input_shape)
        self.convmix1 = MixtureOfConvolutions()
        self.convmix2 = MixtureOfConvolutions()
        self.flatten = tf.keras.layers.Flatten()
        self.value_fc1 = tf.keras.layers.Dense(1024, activation="relu")
        self.value_fc2 = tf.keras.layers.Dense(1, activation="linear")
        self.advantage_fc1 = tf.keras.layers.Dense(1024, activation="relu")
        self.advantage_fc2 = tf.keras.layers.Dense(num_actions, activation="linear")

    def call(self, inputs):
        x = self.inp(inputs)
        x = self.convmix1(x)
        x = self.convmix2(x)
        x = self.flatten(x)
        value = self.value_fc1(x)
        value = self.value_fc2(value)
        advantage = self.advantage_fc1(x)
        advantage = self.advantage_fc2(advantage)
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        return q_values
