# coding: utf8
import tensorflow as tf
import numpy as np

from utility import *
from operations import *

class Discriminator:
    """
        Class to define the neural network that acts as the discriminator.
    """
    def __init__(self, discriminatorVariables, discriminatorPadding, discriminatorStrides):
        """
            Constructor of the discriminator class.
            args :
                discriminatorVariables : dictionnary that contains the variables of each layer.
                discriminatorPadding : dictionnary that contains the padding for each layer.
                discriminatorStrides : dictionnary that contains the strides of the convolutions for each layer.
        """
        self.variables = discriminatorVariables
        self.padding = discriminatorPadding
        self.strides = discriminatorStrides

    def model(self, input):
        """
            Defines the model for the discriminator.
            The model is the reverse of the of the generator.
            Input a latent code of shape (batch, latentCode)
            The filters are 3D : width x height x colorChannels.
            The output is a probability of wether the data belongs to the dataset.
            [Radford et al. 2016] discriminator should use leaky RELU for all layers.
        """
            
        """
                First layer of 2D convolution with batch norm.
                Input size : (?, 32, 32, 1)
                Output size : (?, 16, 16, 32)
                Convolution : 2D filter 4x4, padding 1, stride 2 to down sample
        """
        convLayer1_withBias = convolution2D(input, self.variables['wCL1'], self.variables['bCL1'], self.strides['ConvLayer1'], self.padding['ConvLayer1'])
            
        outputConvLayer1 = lrelu(convLayer1_withBias,  0.2)
        
        """
                Second layer of 2D convolution with batch norm.
                Input size : (?, 16, 16, 32)
                Output size : (?, 8, 8, 64)
                Convolution : 2D filter 4x4, padding 1, stride 2 to down sample
        """
        convLayer2_BNwithBias = convolution2D_batchNorm(outputConvLayer1, self.variables['wCL2'], self.variables['bCL2'], self.variables['bnCL2_scale'], self.variables['bnCL2_offset'], self.strides['ConvLayer2'], self.padding['ConvLayer2'])
        
        outputConvLayer2 = lrelu(convLayer2_BNwithBias,  0.2)
    
        """
                Third layer of 2D convolution with batch norm.
                Input size : (?, 8, 8, 64)
                Output size : (?, 4, 4, 128)
                Convolution : 2D filter 4x4, padding 1, stride 2 to down sample
        """
        convLayer3_BNwithBias = convolution2D_batchNorm(outputConvLayer2, self.variables['wCL3'], self.variables['bCL3'], self.variables['bnCL3_scale'], self.variables['bnCL3_offset'], self.strides['ConvLayer3'], self.padding['ConvLayer3'])
            
        outputConvLayer3 = lrelu(convLayer3_BNwithBias,  0.2)
        
        """
                Fourth layer of 2D convolution with batch norm.
                Input size : (?, 4, 4, 128)
                Output size : (?, 2, 2, 256)
                Convolution : 2D filter 4x4, padding 1, stride 2 to down sample
        """
        convLayer4_BNwithBias = convolution2D_batchNorm(outputConvLayer3, self.variables['wCL4'], self.variables['bCL4'], self.variables['bnCL4_scale'], self.variables['bnCL4_offset'], self.strides['ConvLayer4'], self.padding['ConvLayer4'])
                    
        outputConvLayer4 =lrelu(convLayer4_BNwithBias, 0.2)
        
        """
                Fifth layer : Linearize the space and project it onto a single number.
                Use sigmoid output to map it to a probability.
                Input size : (?, 2, 2, 256)
                Output size : (?, 1)
                Convolution : 2D filter 4x4, padding 1, stride 2 to down sample
        """
        outputConvLayer4Reshaped = tf.reshape(outputConvLayer4, [-1,2*2*256])
     
        output = projection(outputConvLayer4Reshaped, self.variables['wProjection'], self.variables['bProjection'])
        outputSigmoid = tf.nn.sigmoid(output)
        
        return outputSigmoid, output

