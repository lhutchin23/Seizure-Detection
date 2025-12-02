import random

import matplotlib as plt
import numpy as np
import pandas as pd
from scipy import signal

import dataloader


class conv2d:
    def __init__(self, input_shape, kernal_size, depth, stride):
        self.input_depth, self.input_height, self.input_width = input_shape
        self.depth = depth
        self.stride = stride
        self.input_shape = input_shape
        self.output_shape = (
            depth,
            (self.input_height - kernal_size) // stride + 1,
            (self.input_width - kernal_size) // stride + 1,
        )
        self.kernal_size = (depth, self.input_depth, kernal_size, kernal_size)
        self.kernals = np.random.randn(*self.kernal_size)
        self.bias = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.bias)
        for i in range(self.depth):
            for j in range(self.input_shape[2]):
                conv_res = signal.convolve2d(
                        input[j], self.kernals[i][j], mode='valid'
                )
                if self.stride > 1:
                    conv_res = conv_res[::self.stride, ::self.stride]
                    self.output[i] += conv_res

        return self.output

    def backward(self, output_gradient, learning_rate = np.float32(0.02)):

        if self.stride > 1:
            h_pre_stride = (self.input_height - self.kernal_size)+1 
            w_pre_stride = (self.input_width - self.kernal_size)+1

            dilated_grad = np.zeros((self.depth, h_pre_stride, w_pre_stride))
            dilated_grad[:, ::self.stride, ::self.stride] = output_gradient
            output_gradient = dilated_grad
    
        input_gradient = np.zeros(self.input_shape)
        kernal_gradient = np.zeros(self.kernal_size)

        for i in range(self.depth):
            for j in range(self.input_shape[2]):
                kernal_gradient[i][j] += signal.convolve2d(
                    self.input[j], output_gradient[i], mode='valid'
                )
                input_gradient[j] += signal.convolve2d(
                    output_gradient[i], self.kernals[i][j], mode="full"
                )

        self.kernals -= learning_rate * kernal_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient

        # basic convolution layer operation


class maxPoolingLayer:
    def __init__(self, input_shape, pool_size, stride=2):
        # use stride of 2 for smaller data
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.stride = stride
        self.output_shape = (
            input_shape[0],
            (input_shape[1] - pool_size) // stride + 1,
            (input_shape[2] - pool_size) // stride + 1,
        )

    def forward(self, input):
        self.input = input
        output = np.zeros(self.output_shape)
        for d in range(self.input_shape[0]):
            for i in range(0, self.input_shape[1] - self.pool_size + 1, self.stride):
                for j in range(
                    0, self.input_shape[2] - self.pool_size + 1, self.stride
                ):
                    region = input[d, i : i + self.pool_size, j : j + self.pool_size]
                    output[d, i // self.stride, j // self.stride] = np.max(region)
        return output


    def backward(self, output_gradient):
        input_gradient = np.zeros(self.input_shape)
        for d in range(self.input_shape[0]):
            for i in range(0, self.input_shape[1] - self.pool_size + 1, self.stride):
                for j in range(
                    0, self.input_shape[2] - self.pool_size + 1, self.stride
                ):
                    region = self.input[d, i : i + self.pool_size, j : j + self.pool_size]
                    max_index = np.argmax(region)
                    max_i, max_j = np.unravel_index(max_index, region.shape)
                    input_gradient[d, i + max_i, j + max_j] += output_gradient[d, i // self.stride, j // self.stride]
        return input_gradient


class fullyConnectedLayer:
    def __init__(self, input_shape, output):
        self.input_shape = input_shape
        self.output = output
        self.input_size = np.prod(input_shape)
        self.weights = np.random.randn(self.input_size, output)
        self.bias = np.random.randn(output)
        

    def forward(self, input):
        self.input = input.flatten()
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.outer(self.input, output_gradient)

        self.weights -= learning_rate * weights_gradient

        self.bias -= learning_rate * output_gradient

        return input_gradient


class relu:
    def __init__(self, input):
        self.input = input

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient):
        input_gradient = output_gradient.copy()
        input_gradient[self.input <= 0] = 0
        return input_gradient


class dropout:
    def __init__(self, dropout_rate=0.2, training=True):
        self.dropout_rate = dropout_rate
        self.training = training
        self.mask = None

    def forward(self, input):
        if self.training:
            self.mask = np.random.rand(*input.shape) > self.dropout_rate
            output = np.multiply(input, self.mask)
            output = output / (1.0 - self.dropout_rate)
        else:
            output = input
        return output

    def backward(self, output_gradient, learning_rate):
        if self.training:
            input_gradient = output_gradient * self.mask
            input_gradient = input_gradient / (1.0 - self.dropout_rate)
            return input_gradient
        else:
            return output_gradient


class convlayer(): 

    def __init__(self, input_shape, kernal_size, depth, stride):
        self.conv = conv2d(input_shape, kernal_size, depth, stride)
        conv_output_shape = self.conv.output_shape
        self.relu = relu(conv_output_shape)
        self.pool = maxPoolingLayer(conv_output_shape, pool_size=2, stride=2)
        self.output_shape = self.pool.output_shape

    def forward(self, input):
        conv_out = self.conv.forward(input)
        relu_out = self.relu.forward(conv_out)
        convlayeroutput = self.pool.forward(relu_out)
        return convlayeroutput

    def backward(self, output_gradient, learning_rate=0.02):
        pool_grad = self.pool.backward(output_gradient)
        relu_grad = self.relu.backward(pool_grad)
        convlayergrad = self.conv.backward(relu_grad, learning_rate)
        return convlayergrad

class FCLayer:
    def __init__(self, input_shape, output):
        self.fullyConnected = fullyConnectedLayer(input_shape, output)
        self.relu = relu(self.fullyConnected.output)
        
    def forward(self, input):
        fullyConnected_out = self.fullyConnected.forward(input)
        final_output = self.relu.forward(fullyConnected_out)
        return final_output
    def backward(self, output_gradient, learning_rate=0.02):
        relu_grad = self.relu.backward(output_gradient)
        final_grad = self.fullyConnected.backward(relu_grad, learning_rate)
        return final_grad





    
        







