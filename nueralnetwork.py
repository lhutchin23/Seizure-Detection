import random
import normalize

import matplotlib as plt
import numpy as np
import pandas as pd
import dataloader
from scipy import signal

datas = dataloader.preprocess()
training_data = dataloader.batch_generator(datas[0], batch_size=128)


#80, 10, 10 train, val, test split

X_train = np.array([], dtype=np.float64)
X_val = np.array([], dtype=np.float64)
X_test = np.array([], dtype=np.float64)
Y_train = np.array([], dtype=np.int64)
Y_val = np.array([], dtype=np.int64)
Y_test = np.array([], dtype=np.int64)

X_train, Y_train, X_val, Y_val, X_test, Y_test = dataloader.preprocess()



class convolutionalLayer(training_data):
    def __init__(self, input_shape, kernal_size, depth):
        input_height, input_width, input_depth = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.output_shape = (depth, input_height - kernal_size + 1, input_width - kernal_size + 1)
        self.kernal_size = (depth, input_depth, kernal_size, kernal_size)
        self.kernals = np.random.randn(*self.kernal_size) 
        self.bias = np.random.randn(*self.output_shape)

        def forward(self, input):
            self.input = input
            self.output_shape = np.copy(self.bias)
            for i in range(self.depth):
                for j in range(self.input_shape[2]):
                    self.output_shape[i] += self.convolve2d(input[j], self.kernals[i][j])

            return self.output
        

        def backward(self, output_gradient, learning_rate):
            input_gradient = np.zeros(self.input_shape)
            kernal_gradient = np.zeros(self.kernal_size)

            for i in range(self.depth):
                for j in range(self.input_shape[2]):
                    kernal_gradient[i][j] += self.convolve2d(self.input[j], output_gradient[i])
                    input_gradient[j] += self.convolve2d(output_gradient[i], self.kernals[i][j], mode='full')

            self.kernals -= learning_rate * kernal_gradient
            self.bias -= learning_rate * output_gradient

            return input_gradient
        #basic convolution layer operation


class maxPoolingLayer(training_data):
    def __init__(self, input_shape, pool_size, stride=2):
        # use stride of 2 for smaller data
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.stride = stride
        self.output_shape = (input_shape[0], 
                             (input_shape[1] - pool_size)//stride + 1,
                             (input_shape[2] - pool_size)//stride + 1)

    def forward(self, input):
        self.input = input
        output = np.zeros(self.output_shape)
        for d in range(self.input_shape[0]):
            for i in range(0, self.input_shape[1] - self.pool_size + 1, self.stride):
                for j in range(0, self.input_shape[2] - self.pool_size + 1, self.stride):
                    region = input[d, i:i+self.pool_size, j:j+self.pool_size]
                    output[d, i//self.stride, j//self.stride] = np.max(region)
        return output
    

class fullyConnectedLayer(training_data):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient.mean(axis=0, keepdims=True)

        return input_gradient
    

          
