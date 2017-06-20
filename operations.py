# coding: utf8
import tensorflow as tf
import numpy as np

from utility import *


def projection(input, weights, biases):
    """
        Calculates the matrix operation input*W+bias.
        Then reshapes the vector to (?, shapeOutput).
        args :
            input : placeholder that contains the input variable.
            weights : weight matrix W in the projection Wx+b
            biases : biases
		returns :
			The results of input*weights+bias.
    """
    
    outputProjection = tf.matmul(input, weights)+biases
    
    return outputProjection

    
def projectionAndReshapeDCGAN_batchNorm(input, weights, biases, bnScale, bnOffset, outputShape):
    """
        Calculates the matrix operation input*W+bias and reshape the result into a tensor of shape : outputShape.
        Then adds a batch norm layer after.
        args :
            input : placeholder that contains the input variable.
            weights : weight matrix W in the projection Wx+b
            biases : biases b
            bnScale : scale tensor for the batch normalization.
            bnOffset : offset tensor used for the batch normalization.
            outputShape :  shape of the output (necessary for the reshape)
        returns :
            The results of input*weights+bias + reshape + batch normalization.
    """
        
    
    outputProjection = tf.reshape(tf.matmul(input, weights)+biases, outputShape)
    
    #Batch normalization
    batchMean, batchVar = tf.nn.moments(outputProjection,[0])
    
    epsilon = 1e-3
    batchNorm_ouput = tf.nn.batch_normalization(outputProjection, batchMean, batchVar, bnOffset, bnScale, epsilon)

    return batchNorm_ouput

def convolution2D(input, weights, biases, strides, padding):
    """
        Performs a 2D convolution with biases.
        args :
            input : placeholder that contains the input space.
            weights : filters used for the convolutions.
            biases : biases used for the convolution.
            strides : strides used for the convolution.
            padding : padding used for the convolution.
        returns :
            Result of the convolution.
    """
    
    #Apply the convolution
    convLayer = tf.nn.conv2d(input, weights, strides=strides, padding=padding)
    
    #Add the biases
    convLayer_withBias = tf.nn.bias_add(convLayer, biases)
    
    return convLayer_withBias

def convolution2D_batchNorm(input, weights, biases, bnScale, bnOffset, strides, padding):
    """
        Performs a 2D convolution with biases and adds a batch normalization layer after.
        args :
            input : placeholder that contains the input space.
            weights : filters used for the convolutions.
            biases : biases used for the convolution.
            bnScale : scale tensor for the batch normalization.
            bnOffset : offset tensor used for the batch normalization.
            strides : strides used for the convolution.
            padding : padding used for the convolution.
        returns :
            Result of the convolution + batch norm layer.
    """
    
    #Apply the convolution
    convLayer = tf.nn.conv2d(input, weights, strides=strides, padding=padding)
    
    #Add the biases
    convLayer_withBias = tf.nn.bias_add(convLayer, biases)
    
    #Batch normalization
    batchMean, batchVar = tf.nn.moments(convLayer_withBias,[0])
    
    epsilon = 1e-3
    batchNorm_ouput = tf.nn.batch_normalization(convLayer_withBias, batchMean, batchVar, bnOffset, bnScale, epsilon)
    
    return batchNorm_ouput

def upConvolution2D(input, weights, biases, outputShape, strides, padding):
    """
        Performs a 2D fractionally strided convolution with biases and adds a batch normalization layer after.
        args :
            input : placeholder that contains the input space.
            weights : filters used for the convolutions.
            biases : biases used for the convolution.  
            outputShape : shape of the tensor after the fractionally strided convolution.
            strides : strides used for the convolution.
            padding : padding used for the convolution.
        returns :
            Result of the convolution + batch norm layer.
    """


    #Data format is NHWC = (batch, height, width, color channels)
    upConvLayer = tf.nn.conv2d_transpose(input, weights, outputShape, strides=strides, padding=padding, data_format='NHWC')

    #Add bias
    upConvLayer_withBias = tf.nn.bias_add(upConvLayer, biases)
    
    return upConvLayer_withBias

def upConvolution2D_batchNorm(input, weights, biases, bnScale, bnOffset, outputShape, strides, padding):
    """
        Performs a 2D fractionally strided convolution with biases and adds a batch normalization layer after.
        args :
            input : placeholder that contains the input space.
            weights : filters used for the convolutions.
            biases : biases used for the convolution.
            bnScale : scale tensor for the batch normalization.
            bnOffset : offset tensor used for the batch normalization.
            outputShape : shape of the tensor after the fractionally strided convolution.
            strides : strides used for the convolution.
            padding : padding used for the convolution.
        returns :
            Result of the fractionally strided convolution + batch norm layer.
    """
    
    #Data format is NHWC = (batch, height, width, color channels)
    upConvLayer = tf.nn.conv2d_transpose(input, weights, outputShape, strides, padding=padding, data_format='NHWC')

    #Add bias
    upConvLayer_withBias = tf.nn.bias_add(upConvLayer, biases)
    
    #Batch normalization
    batchMean, batchVar = tf.nn.moments(upConvLayer_withBias,[0])
    
    epsilon = 1e-3
    batchNorm_ouput = tf.nn.batch_normalization(upConvLayer_withBias, batchMean, batchVar, bnOffset, bnScale, epsilon)
    
    return batchNorm_ouput

def convolution3D(input, weights, biases, strides, padding):
    """
        Performs a 3D convolution with biases.
        args :
            input : placeholder that contains the input space.
            weights : filters used for the convolutions.
            biases : biases used for the convolution.
            strides : strides used for the convolution.
            padding : padding used for the convolution.
        returns :
            Result of the convolution.
    """
    
    #Apply the convolution
    #Conv3d must have strides[0] = strides[4] = 1
    convLayer = tf.nn.conv3d(input, weights, strides=strides, padding=padding)
    
    #Add the biases
    convLayer_withBias = tf.nn.bias_add(convLayer, biases)
    
    return convLayer_withBias

def convolution3D_batchNorm(input, weights, biases, bnScale, bnOffset, strides, padding):
    """
        Performs a 3D convolution with biases and adds a batch normalization layer after.
        args :
            input : placeholder that contains the input space.
            weights : filters used for the convolutions.
            biases : biases used for the convolution.
            strides : strides used for the convolution.
            padding : padding used for the convolution.
        returns :
            Result of the convolution + batch norm layer.
    """
    
    #Apply the convolution
    #Conv3d must have strides[0] = strides[4] = 1
    convLayer = tf.nn.conv3d(input, weights, strides=strides, padding=padding)
    
    #Add the biases
    convLayer_withBias = tf.nn.bias_add(convLayer, biases)

    #Batch normalization
    batchMean, batchVar = tf.nn.moments(convLayer_withBias,[0])
    
    epsilon = 1e-3
    batchNorm_ouput = tf.nn.batch_normalization(convLayer_withBias, batchMean, batchVar, bnOffset, bnScale, epsilon)
    
    return batchNorm_ouput

def upConvolution3D(input, weights, biases, outputShape, strides, padding):
    """
        Performs a 3D fractionally strided convolution with biases and adds a batch normalization layer after.
        args :
        input : placeholder that contains the input space.
        weights : filters used for the convolutions.
        biases : biases used for the convolution.
        outputShape : shape of the tensor after the fractionally strided convolution.
        strides : strides used for the convolution.
        padding : padding used for the convolution.
        returns :
        Result of the fractionally strided convolution + batch norm layer.
    """
    
    
    #Apply the convolution
    #Conv3d must have strides[0] = strides[4] = 1
    convLayer = tf.nn.conv3d_transpose(input, weights, outputShape, strides=strides, padding=padding)
    
    #Add the biases
    convLayer_withBias = tf.nn.bias_add(convLayer, biases)
    
    return convLayer_withBias

def upConvolution3D_batchNorm(input, weights, biases, bnScale, bnOffset, outputShape, strides, padding):
    """
        Performs a 3D fractionally strided convolution with biases and adds a batch normalization layer after.
        args :
            input : placeholder that contains the input space.
            weights : filters used for the convolutions.
            biases : biases used for the convolution.
            bnScale : scale tensor for the batch normalization.
            bnOffset : offset tensor used for the batch normalization.
            outputShape : shape of the tensor after the fractionally strided convolution.
            strides : strides used for the convolution.
            padding : padding used for the convolution.
        returns :
            Result of the fractionally strided convolution + batch norm layer.
    """
    
    
    #Apply the convolution
    #Conv3d must have strides[0] = strides[4] = 1
    convLayer = tf.nn.conv3d_transpose(input, weights, outputShape, strides=strides, padding=padding)
            
    #Add the biases
    convLayer_withBias = tf.nn.bias_add(convLayer, biases)
    
    #Batch normalization
    batchMean, batchVar = tf.nn.moments(convLayer_withBias,[0])
    
    epsilon = 1e-3
    batchNorm_ouput = tf.nn.batch_normalization(convLayer_withBias, batchMean, batchVar, bnOffset, bnScale, epsilon)

    return batchNorm_ouput