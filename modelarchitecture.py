import random

import matplotlib as plt
import numpy as np
import pandas as pd
from scipy import signal

import dataloader


class convolutionalLayer:
    def __init__(self, input_shape, kernal_size, depth, stride):
        input_height, input_width, input_depth = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.output_shape = (
            depth,
            (input_height - kernal_size) / stride + 1,
            (input_width - kernal_size) / stride + 1,
        )
        self.kernal_size = (depth, input_depth, kernal_size, kernal_size)
        self.kernals = np.random.randn(*self.kernal_size)
        self.bias = np.random.randn(*self.output_shape)

        def forward(self, input):
            self.input = input
            self.output_shape = np.copy(self.bias)
            for i in range(self.depth):
                for j in range(self.input_shape[2]):
                    self.output_shape[i] += signal.convolve2d(
                        input[j], self.kernals[i][j]
                    )

            return self.output

        def backward(self, output_gradient, learning_rate):
            input_gradient = np.zeros(self.input_shape)
            kernal_gradient = np.zeros(self.kernal_size)

            for i in range(self.depth):
                for j in range(self.input_shape[2]):
                    kernal_gradient[i][j] += signal.convolve2d(
                        self.input[j], output_gradient[i]
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


class fullyConnectedLayer:
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


class relu:
    def __init__(self, input):
        self.input = input

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient, learning_rate):
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
