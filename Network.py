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
        # this neural network contains 5 layers
        self.dense_1 = tf.keras.layers.Dense(1024, activation="selu")
        self.dropout_1 = tf.keras.layers.Dropout(0.5)
        self.dense_2 = tf.keras.layers.Dense(2048, activation="selu")
        self.dropout_2 = tf.keras.layers.Dropout(0.5)
        self.dense_3 = tf.keras.layers.Dense(1024, activation="selu")
        self.dropout_3 = tf.keras.layers.Dropout(0.5)
        self.dense_4 = tf.keras.layers.Dense(output_size, activation="selu")

    def __call__(self, x):
        x = self.dense_1(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        x = self.dense_3(x)
        x = self.dropout_3(x)
        x = self.dense_4(x)
        return x