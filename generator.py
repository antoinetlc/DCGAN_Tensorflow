# coding: utf8
import tensorflow as tf
import numpy as np

from utility import *
from operations import *

class Generator:
    """
    Class to define the neural network that acts as the generator.
    """
    def __init__(self, generatorVariables, upConvOutputShapes, generatorPadding, generatorStrides, batchSize):
        """
            Constructor of the discriminator class.
            args :
            generatorVariables : dictionnary that contains the variables of each layer.
            upConvOutputShapes : dictionnary that contains the shapes of each layer.
            generatorPadding : dictionnary that contains the padding for each layer.
            generatorStrides : dictionnary that contains the strides of the convolutions for each layer.
        """
        self.variables = generatorVariables
        self.padding = generatorPadding
        self.strides = generatorStrides
        self.upConvOutputShapes = upConvOutputShapes
        self.batchSize = batchSize

    def model(self, latentCode):
        """
            Defines the model for the generator.
            Input is a latentCode shape (?, 200)
            The output is an image of size (?,32,32,1)
        """
        """
                First layer : Projection and reshape
                Input size : (?, 200)
                Output size : (?, 1, 1, 512)
        """
        outputShape = self.upConvOutputShapes['Projection']
           
        projection_batchNorm = projectionAndReshapeDCGAN_batchNorm(latentCode, self.variables['wProj'], self.variables['bProj'], self.variables['bnProj_scale'], self.variables['bnProj_scale'], outputShape)
            
        outputProjection = tf.nn.relu(projection_batchNorm)
        
        """
                Second layer : 2d fractionally strided convolution with batch norm
                Input size : (?, 1, 1, 512)
                Output size : (?, 2, 2, 256)
        """
        outputShape = self.upConvOutputShapes['ConvLayer1']
        outputShape[0] = self.batchSize  #The transpose convolution needs to know the exact shape

        upConvLayer1_BNwithBias = upConvolution2D_batchNorm(outputProjection, self.variables['wCL1'], self.variables['bCL1'], self.variables['bnCL1_scale'], self.variables['bnCL1_offset'], outputShape, self.strides['ConvLayer1'], self.padding['ConvLayer1'])
            
        outputConvLayer1 = tf.nn.relu(upConvLayer1_BNwithBias)
        
        """
                Third layer : 2d fractionally strided convolution with batch norm
                Input size : (?, 2, 2, 256)
                Output size : (?, 4, 4, 128)
        """
        outputShape = self.upConvOutputShapes['ConvLayer2']
        outputShape[0] = self.batchSize  #The transpose convolution needs to know the exact shape
            
        upConvLayer2_BNwithBias = upConvolution2D_batchNorm(outputConvLayer1, self.variables['wCL2'], self.variables['bCL2'], self.variables['bnCL2_scale'], self.variables['bnCL2_offset'], outputShape, self.strides['ConvLayer2'], self.padding['ConvLayer2'])
          
        outputConvLayer2 = tf.nn.relu(upConvLayer2_BNwithBias)
     
        """
                Third layer : 2d fractionally strided convolution with batch norm
                Input size : (?, 4, 4, 128)
                Output size : (?, 8, 8, 64)
        """
        outputShape = self.upConvOutputShapes['ConvLayer3']
        outputShape[0] = self.batchSize  #The transpose convolution needs to know the exact shape
            
        upConvLayer3_BNwithBias = upConvolution2D_batchNorm(outputConvLayer2, self.variables['wCL3'], self.variables['bCL3'], self.variables['bnCL3_scale'], self.variables['bnCL3_offset'], outputShape, self.strides['ConvLayer3'], self.padding['ConvLayer3'])
            
        outputConvLayer3 = tf.nn.relu(upConvLayer3_BNwithBias)

        """
                Fourth layer : 2d fractionally strided convolution with batch norm
                Input size : (?, 8, 8, 64)
                Output size : (?, 16, 16, 32)
        """
        outputShape = self.upConvOutputShapes['ConvLayer4']
        outputShape[0] = self.batchSize  #The transpose convolution needs to know the exact shape
            
        upConvLayer4_BNwithBias = upConvolution2D_batchNorm(outputConvLayer3, self.variables['wCL4'], self.variables['bCL4'], self.variables['bnCL4_scale'], self.variables['bnCL4_offset'], outputShape, self.strides['ConvLayer4'], self.padding['ConvLayer4'])
            
        outputConvLayer4 = tf.nn.relu(upConvLayer4_BNwithBias)

        """
                Fourth layer : 2d fractionally strided convolution. NO batch norm
                Input size : (?, 16, 16, 32)
                Output size : (?, 32, 32, 1)
                Finish with tanh to map the output to range [-1;1] (same interval as latent code)
        """
        outputShape = self.upConvOutputShapes['ConvLayer5']
        outputShape[0] = self.batchSize  #The transpose convolution needs to know the exact shape
            
        upConvLayer5_BNwithBias = upConvolution2D(outputConvLayer4, self.variables['wCL5'], self.variables['bCL5'], outputShape, self.strides['ConvLayer5'], self.padding['ConvLayer5'])
        
        outputConvLayer5 = tf.tanh(upConvLayer5_BNwithBias)

        return outputConvLayer5

