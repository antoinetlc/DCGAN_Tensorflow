# coding: utf8
import os
import sys
import gc
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import imageio
from random import shuffle

from discriminator import *
from generator import *
from operations import *
from utility import *

def lossFunctions(latentCode, imagesBatch, generator, discriminator):
    """
        Computes the loss functions for the generator and the discriminator.
        args:
            latentCode. Input images to generate the videos dim [batch, width, height, colorChannels]
            imagesBatch : batch of images taken from the dataset.
            generator : generator to generate images from the latent codes.
            discriminator : discriminator.
    """
    lossDiscriminator = 0.0
    lossGenerator = 0.0
    
    # Discriminator : data coming from dataset. Discriminator should predict that it is real.
    probabilityReal, logits_real = discriminator.model(imagesBatch) #[batch, 2]
    logitsReal_reshaped = tf.reshape(logits_real,[-1,1])
    lossDiscriminator = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logitsReal_reshaped, labels=tf.ones_like(logitsReal_reshaped)))
    
    # Discriminator : data coming from generator. Discriminator should predict that it is fake->zeros
    generatedImages = generator.model(latentCode) #[batch, time, height, width, colorchannels]
    probabilityFake, logits_fake = discriminator.model(generatedImages) #[batch, 2]
    logitsFake_reshaped = tf.reshape(logits_fake,[-1,1])

    lossDiscriminator += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logitsFake_reshaped, labels=tf.zeros_like(logitsFake_reshaped)))


    # Generator : data coming from generator. Generator wants to fool discriminator and let it predict that it is real->ones (although it is fake data)
    lossGenerator += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logitsFake_reshaped, labels=tf.ones_like(logitsFake_reshaped)))
    
    return lossDiscriminator, lossGenerator, generatedImages, probabilityReal, probabilityFake

def setupCheckpoint(checkpointDir, session, globalStep, saver):
	"""
		Setup the checkpoints system to save progress during training.
		args:
			session : variable to the current session.
		returns:
			startStep : last step at which the training stopped.
	"""
	if not os.path.exists(checkpointDir):
		os.makedirs(checkpointDir)
	
	#Start training from the value saved in global step
	checkpoint = tf.train.get_checkpoint_state(checkpointDir)
		
	#If the checkpoint exist, restores from that
	if checkpoint and checkpoint.model_checkpoint_path:
		print(checkpoint.model_checkpoint_path)
		saver.restore(session, checkpoint.model_checkpoint_path)

	startStep = globalStep.eval()

	return startStep

def main():
    
    LATENT_CODE_SIZE = 200
    WIDTH = 32
    HEIGHT = 32
    NUM_COLOR_CHANNEL = 1
    BATCH_SIZE = 128
    NUM_TRAINING_STEPS = 500000
    
    #load mnist and reshape
    TRAIN_DIR = "MNIST_DATA"
    mnist = input_data.read_data_sets(TRAIN_DIR, one_hot = True)
    
    #Only one GPU is visible
    os.environ["CUDA_VISIBLE_DEVICES"]="0"    

    """
        Discriminator : size of the variables for each layer.
		An image is a 3D space : width x height x colorChannels, hence we need 3d filters
		For a 2D convolutional layer (e.g ConvLayer1) the list contains the filter information
		[width, height, input_channels, ouput_channels]
		For the batch norm layer the list contains the size of the variables scales and offset (same size as the ouput of the previous layer)
        
        Last layer of the discriminator is a projection to map the features to a probability.
        No batch norm in the first layer of the discriminator.
    """
    discriminatorWeights = {
    'ConvLayer1' : [4,4, NUM_COLOR_CHANNEL, 32],
    'ConvLayer2' : [4,4, 32, 64],
    'ConvLayer2BN' : [8, 8, 64],
    'ConvLayer3' : [4,4, 64, 128],
    'ConvLayer3BN' : [4, 4, 128],
    'ConvLayer4' : [4,4, 128, 256],
    'ConvLayer4BN' : [2, 2, 256],
    'Projection' : [2*2*256, 1], #time,height,width,inputChannels, outputChannels
    }

	#Each layer has the same number of biases as the number of filters
    discriminatorBiases = {
    'ConvLayer1' : [32],
    'ConvLayer2' : [64],
    'ConvLayer3' : [128],
    'ConvLayer4' : [256],
    'Projection' : [1]
    }
    
    #Padding for each conv layer
    discriminatorPadding = {
    'ConvLayer1' : 'SAME',
    'ConvLayer2' : 'SAME',
    'ConvLayer3' : 'SAME',
    'ConvLayer4' : 'SAME'
    }
    
    #Stride for each conv layer
    discriminatorStrides = {
    'ConvLayer1' : [1, 2, 2, 1],
    'ConvLayer2' : [1, 2, 2, 1],
    'ConvLayer3' : [1, 2, 2, 1],
    'ConvLayer4' : [1, 2, 2, 1]
    }
    
    """
        It is important to define the variables in advance to use the same variables when evaluating
        the discriminator with the real and the fake example (do not create different variables for fake and real
        example).
    """
    discriminatorVariables = {
    'wCL1': tf.Variable(tf.random_normal(discriminatorWeights['ConvLayer1'], stddev=0.02), tf.float32, name="D_weightsCL1"),
    'bCL1': tf.Variable(tf.zeros(discriminatorBiases['ConvLayer1']), tf.float32, name="D_biasesCL1"),
    'wCL2': tf.Variable(tf.random_normal(discriminatorWeights['ConvLayer2'], stddev=0.02), tf.float32, name="D_weightsCL2"),
    'bCL2': tf.Variable(tf.zeros(discriminatorBiases['ConvLayer2']), tf.float32, name="D_biasesCL2"),
    'bnCL2_scale': tf.Variable(tf.ones(discriminatorWeights['ConvLayer2BN']), tf.float32, name="D_bnScaleCL2"),
    'bnCL2_offset': tf.Variable(tf.zeros(discriminatorWeights['ConvLayer2BN']), tf.float32, name="D_bnOffsetCL2"),
    'wCL3': tf.Variable(tf.random_normal(discriminatorWeights['ConvLayer3'], stddev=0.02), tf.float32, name="D_weightsCL3"),
    'bCL3': tf.Variable(tf.zeros(discriminatorBiases['ConvLayer3']), tf.float32, name="D_biasesCL3"),
    'bnCL3_scale': tf.Variable(tf.ones(discriminatorWeights['ConvLayer3BN']), tf.float32, name="D_bnScaleCL3"),
    'bnCL3_offset': tf.Variable(tf.zeros(discriminatorWeights['ConvLayer3BN']), tf.float32, name="D_bnOffsetCL3"),
    'wCL4': tf.Variable(tf.random_normal(discriminatorWeights['ConvLayer4'], stddev=0.02), tf.float32, name="D_weightsCL4"),
    'bCL4': tf.Variable(tf.zeros(discriminatorBiases['ConvLayer4']), tf.float32, name="D_biasesCL4"),
    'bnCL4_scale': tf.Variable(tf.ones(discriminatorWeights['ConvLayer4BN']), tf.float32, name="D_bnScaleCL4"),
    'bnCL4_offset': tf.Variable(tf.zeros(discriminatorWeights['ConvLayer4BN']), tf.float32, name="D_bnOffsetCL4"),
    'wProjection': tf.Variable(tf.random_normal(discriminatorWeights['Projection'], stddev=0.02), tf.float32, name="D_weightsProjection"),
    'bProjection': tf.Variable(tf.zeros(discriminatorBiases['Projection']), tf.float32, name="D_biasesProjection")
    }
    
    
    """
        Generator : size of the variables for each layer.
        An image is a 3D space : width x height x colorChannels, hence we need 3d filters
        For a 2D convolutional layer (e.g ConvLayer1) the list contains the filter information
        [width, height, input_channels, ouput_channels]
        For the batch norm layer the list contains the size of the variables scales and offset (same size as the ouput of the previous layer)
        
        No batch norm in the last layer of the generator.
    """
    generatorWeights = {
    'Projection': [200, 1*1*512],
    'ProjectionBN' : [1,1,512],
    'ConvLayer1' : [4, 4, 256, 512], #The filter go through the entire depth
    'ConvLayer1BN' : [2,2, 256],
    'ConvLayer2' : [4, 4, 128, 256],
    'ConvLayer2BN' : [4, 4, 128],
    'ConvLayer3' : [4, 4, 64, 128],
    'ConvLayer3BN' : [8, 8, 64],
    'ConvLayer4' : [4, 4, 32, 64],
    'ConvLayer4BN' : [16, 16, 32],
    'ConvLayer5' : [4, 4, 1, 32]
    }
			
    """
		Output shapes after the upconvolution.
		[batch, time, height, width, filterNumber]
		-1 means keep the size as it is along that dimension.
    """
    generatorUpConvOutputShapes = {
    'Projection': [-1, 1, 1, 512],
    'ConvLayer1' : [-1, 2, 2, 256],
    'ConvLayer2' : [-1, 4, 4, 128],
    'ConvLayer3' : [-1, 8, 8, 64],
    'ConvLayer4' : [-1, 16, 16, 32],
    'ConvLayer5' : [-1, 32, 32, 1]
    }
			
	#Each layer has the same number of biases as filters
    generatorBiases = {
    'Projection' : [1*1*512],
    'ConvLayer1' : [256],
    'ConvLayer2' : [128],
    'ConvLayer3' : [64],
    'ConvLayer4' : [32],
    'ConvLayer5' : [1]
    }
    
    #Padding for each conv layer
    generatorPadding = {
    'ConvLayer1' : 'SAME',
    'ConvLayer2' : 'SAME',
    'ConvLayer3' : 'SAME',
    'ConvLayer4' : 'SAME',
    'ConvLayer5' : 'SAME'
    }
    
    #Stride for each conv layer
    generatorStrides = {
    'ConvLayer1' : [1, 2, 2, 1],
    'ConvLayer2' : [1, 2, 2, 1],
    'ConvLayer3' : [1, 2, 2, 1],
    'ConvLayer4' : [1, 2, 2, 1],
    'ConvLayer5' : [1, 2, 2, 1]
    }
    
    #Generator variables
    generatorVariables = {
    'wProj': tf.Variable(tf.random_normal(generatorWeights['Projection'], stddev=0.02), tf.float32, name="G_wProjection"),
    'bProj': tf.Variable(tf.zeros(generatorBiases['Projection']), tf.float32, name="G_bProjection"),
    'bnProj_scale': tf.Variable(tf.ones(generatorWeights['ProjectionBN']), tf.float32, name="G_bnScaleProj"),
    'bnProj_offset': tf.Variable(tf.zeros(generatorWeights['ProjectionBN']), tf.float32, name="G_bnOffsetProj"),
    'wCL1': tf.Variable(tf.random_normal(generatorWeights['ConvLayer1'], stddev=0.02), tf.float32, name="G_weightsCL1"),
    'bCL1': tf.Variable(tf.zeros(generatorBiases['ConvLayer1']), tf.float32, name="G_biasesCL1"),
    'bnCL1_scale': tf.Variable(tf.ones(generatorWeights['ConvLayer1BN']), tf.float32, name="G_bnScaleCL1"),
    'bnCL1_offset': tf.Variable(tf.zeros(generatorWeights['ConvLayer1BN']), tf.float32, name="G_bnOffsetCL1"),
    'wCL2': tf.Variable(tf.random_normal(generatorWeights['ConvLayer2'], stddev=0.02), tf.float32, name="G_weightsCL2"),
    'bCL2': tf.Variable(tf.zeros(generatorBiases['ConvLayer2']), tf.float32, name="G_biasesCL2"),
    'bnCL2_scale': tf.Variable(tf.ones(generatorWeights['ConvLayer2BN']), tf.float32, name="G_bnScaleCL2"),
    'bnCL2_offset': tf.Variable(tf.zeros(generatorWeights['ConvLayer2BN']), tf.float32, name="G_bnOffsetCL2"),
    'wCL3': tf.Variable(tf.random_normal(generatorWeights['ConvLayer3'], stddev=0.02), tf.float32, name="G_weightsCL3"),
    'bCL3': tf.Variable(tf.zeros(generatorBiases['ConvLayer3']), tf.float32, name="G_biasesCL3"),
    'bnCL3_scale': tf.Variable(tf.ones(generatorWeights['ConvLayer3BN']), tf.float32, name="G_bnScaleCL3"),
    'bnCL3_offset': tf.Variable(tf.zeros(generatorWeights['ConvLayer3BN']), tf.float32, name="G_bnOffsetCL3"),
    'wCL4': tf.Variable(tf.random_normal(generatorWeights['ConvLayer4'], stddev=0.02), tf.float32, name="G_weightsCL4"),
    'bCL4': tf.Variable(tf.zeros(generatorBiases['ConvLayer4']), tf.float32, name="G_biasesCL4"),
    'bnCL4_scale': tf.Variable(tf.ones(generatorWeights['ConvLayer4BN']), tf.float32, name="G_bnScaleCL4"),
    'bnCL4_offset': tf.Variable(tf.zeros(generatorWeights['ConvLayer4BN']), tf.float32, name="G_bnOffsetCL4"),
    'wCL5': tf.Variable(tf.random_normal(generatorWeights['ConvLayer5'], stddev=0.02), tf.float32, name="G_weightsCL5"),
    'bCL5': tf.Variable(tf.zeros(generatorBiases['ConvLayer5']), tf.float32, name="G_biasesCL5"),
    }

    #Checkpoint directory
    checkpointDir = "./checkpoints"
   
    #Placeholders
    latentCode = tf.placeholder(tf.float32, [None, LATENT_CODE_SIZE], name='latentCode')
    imagesBatch = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, NUM_COLOR_CHANNEL], name='imageBatch')
    
    #Initialize generator and discriminator
    generator = Generator(generatorVariables, generatorUpConvOutputShapes, generatorPadding, generatorStrides, BATCH_SIZE)
    
    discriminator = Discriminator(discriminatorVariables, discriminatorPadding, discriminatorStrides)
    
    #Loss functions
    discriminatorLoss, generatorLoss, generatedImages, probabilityReal, probabilityFake = lossFunctions(latentCode, imagesBatch, generator, discriminator)

    #Tensorboard summaries
    tf.summary.scalar("Generator Loss", generatorLoss)
    tf.summary.scalar("Discriminator Loss", discriminatorLoss)
    
	#Optimizers
	#Both discriminator and generator try to minimize the loss function
	#When optimizing keep the variables of the other constant
    startlearningRate = 0.0002
    momentumTerm = 0.5
   
    #Apply a decay of the learning rate with a global step
    globalStep = tf.Variable(0, name="globalStep", trainable=False)
    
    #Graph to increase global step
    incrementGlobalStepOp = tf.assign(globalStep, globalStep+1)
    learningRate = tf.train.exponential_decay(startlearningRate, globalStep, 100000, 0.96, staircase=True)

    #For batch normalization
    #.values() converts the dictionnary values to a list of values
    #Do not attach global step to the minimize operations otherwise one step (i.e train D and G and G) is counted 3 times
    generatorOptimizer = tf.train.AdamOptimizer(learningRate, beta1=momentumTerm).minimize(generatorLoss, var_list=generatorVariables.values())
    discriminatorOptimizer = tf.train.AdamOptimizer(learningRate, beta1=momentumTerm).minimize(discriminatorLoss, var_list=discriminatorVariables.values())
   
    tf.Graph().finalize()
   
    with tf.Session() as sess:
        #Save and store tutorial : https://github.com/nlintz/TensorFlow-Tutorials/blob/master/10_save_restore_net.ipynb
        #Create a global step variable and a saver
        #!! Saver MUST be declared after all variables otherwise the variable will not be saved
        saver = tf.train.Saver()
        
        #Initialize the variables
        sess.run(tf.global_variables_initializer())

        #Write logs
        #Tutorial https://github.com/nlintz/TensorFlow-Tutorials/blob/master/09_tensorboard.ipynb
        writer = tf.summary.FileWriter("./logs/logs", sess.graph)
        merged = tf.summary.merge_all()

        #Setup checkpoints and load the last training step
        startStep = setupCheckpoint(checkpointDir, sess, globalStep, saver)
        
        
        #Saves a generated result to binary
        currentGenImages = sess.run(generatedImages, {latentCode:generateLatentCode(BATCH_SIZE, LATENT_CODE_SIZE)})
        
        #Put in range [0;1]
        currentGenImages += 1.0
        currentGenImages /= 2.0
        currentGenImages *= 255.0
        
        for i in range(0, BATCH_SIZE):
            fileName = "genImage_" + str(i) + ".jpg"
            #Save image
            imageio.imwrite(fileName, np.array(np.floor(currentGenImages[i,:,:,:]), dtype=np.uint8)) #Save the first frame too
    
        sess.close()

    
if __name__ == "__main__":
	main()

