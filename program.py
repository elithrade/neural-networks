import neuralnetwork as nn
import numpy as np

# Ensure file is closed
with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']

layer_sizes = [28*28, 5, 10]
net = nn.NeuralNetwork(layer_sizes)
print(net.accuracy(training_images, training_labels))
