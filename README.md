# DCGAN_Tensorflow
Implementation of Deep Convolutional Generative Adversarial Networks on MNIST data in Tensorflow from the paper :

*Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. Alec Radford, Luke Metz, Soumith Chintala. 2015*

# Tensorflow implementations notes
* Do not use tf.reshape or tf.assign in the main training loop. These create new nodes to the graph and slow down the training over time. The graph should be built before the training starts and not be modified afterwards. If reshape is needed in the training loop, it is better to use numpy.reshape.

