import tensorflow as tf

class NaiveModel(tf.keras.Model):
    def __init__(self, predicted_window_size):
        super().__init__()
        self.predicted_window_size = predicted_window_size

    def call(self, inputs):
        return tf.repeat(inputs[:, -1:, 0], self.predicted_window_size, axis=1)


class RandomModel(tf.keras.Model):
    def __init__(self, predicted_window_size):
        super().__init__()
        self.predicted_window_size = predicted_window_size

    def call(self, inputs):
        output_shape = (tf.shape(inputs)[0], self.predicted_window_size)
        return tf.random.uniform(shape=output_shape)