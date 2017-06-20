# coding: utf8
import tensorflow as tf
import numpy as np

def initializeVariable(shape, inputDimension):
    """
        Initialize a tensor with a gaussian distribution of variance 0.01.
    """
    return tf.Variable(tf.random_normal(shape, stddev=0.01), tf.float32, name="weights")

def initializeBias(shape):
    """
        Initialize biases.
    """
    return tf.Variable(tf.zeros(shape), tf.float32, name="biases")

def lrelu(x, leak):
    """
        Computes the leaky RELU function on a tensor : max(alpha*input, input) (formula valid for alpha<=1 !).
        Implementation obtained from : https://github.com/bamos/dcgan-completion.tensorflow
        args:
            input : tensor on which the non linearity is applied.
            alpha : value of the leaky RELU parameter.
    """
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)
         
def scaleToRange(tensor):
    """
        Puts tensor in the range [-1, 1].
        args:
            tensor
        returns:
            Tensor in range [-1, 1].
    """
    tensor = tensor/255.0
    tensor = tensor*2.0
    tensor = tensor-1.0
    return tensor

def generateLatentCode(batchSize, latentCodeSize):
    """
        Generate a random vector of given shape in interval [-1;1] (uniform distribution).
    """
    return 2.0*np.random.rand(batchSize,latentCodeSize)-1.0
