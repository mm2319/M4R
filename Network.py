import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K
from keras.optimizers import SGD, Adam

 
# define the neural network
class FeedForwardNetwork(tf.keras.Model):
    def __init__(self, output_size,name=None):
        super().__init__(name=name)
        # this neural network contains 8 layers
        self.dense_1 = tf.keras.layers.Dense(1, activation="selu")
        self.dense_2 = tf.keras.layers.Dense(64, activation="selu")
        self.dense_3 = tf.keras.layers.Dense(128, activation="selu")
        self.dense_4 = tf.keras.layers.Dense(256, activation="selu")
        self.dense_5 = tf.keras.layers.Dense(1024, activation="selu")
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense_6 = tf.keras.layers.Dense(256, activation="selu")
        self.dense_7 = tf.keras.layers.Dense(128, activation="selu")
        self.dense_8 = tf.keras.layers.Dense(output_size, activation="selu")

    def __call__(self, x):
        x = self.dense_1(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.dense_5(x)
        x = self.dense_6(x)
        x = self.dropout(x)
        x = self.dense_7(x)
        x = self.dropout(x)
        x = self.dense_8(x)
        return x