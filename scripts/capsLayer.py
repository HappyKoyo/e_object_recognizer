#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import tensorflow as tf

epsilon = 1e-9

def getWeights(shape, stddev=0.01):
    var = tf.get_variable(
        'weights',
        shape,
        initializer=tf.truncated_normal_initializer(stddev=0.01))
    return var

def squash(vector):
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return vec_squashed

def routing(input,b_IJ):
    W = tf.get_variable('Weight', shape=(1, 1152, 10, 8, 16), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=0.01))
    input = tf.tile(input, [1, 1, 10, 1, 1])
    W = tf.tile(W, [1, 1, 1, 1, 1])
    assert input.get_shape() == [1, 1152, 10, 8, 1]
    u_hat = tf.matmul(W, input, transpose_a=True)
    #u_hat = tf.matmul(W,input,transpose_a=True)
    assert u_hat.get_shape() == [1, 1152, 10, 16, 1]
    
    u_hat_stopped = tf.stop_gradient(u_hat,name='stop_gradient')
    for r_iter in range(3):#default=3
        with tf.variable_scope('iter'+str(r_iter)):
            c_IJ = tf.nn.softmax(b_IJ,dim=2)
            if r_iter < 3-1:
                s_J = tf.multiply(c_IJ,u_hat_stopped)
                s_J = tf.reduce_sum(s_J,axis=1,keep_dims=True)
                v_J = squash(s_J)
                v_J_tiled = tf.tile(v_J,[1,1152,1,1,1])
                u_produce_v = tf.matmul(u_hat_stopped,v_J_tiled,transpose_a=True)
                b_IJ += u_produce_v
            elif r_iter == 3-1:
                s_J = tf.multiply(c_IJ,u_hat)
                s_J = tf.reduce_sum(s_J,axis=1,keep_dims=True)
                v_J = squash(s_J)
                assert v_J.get_shape() == [1,1,10,16,1]
    return (v_J)
    
class CapsLayer():
    def __call__(self,input,layer_type):
        if layer_type == 'primary':
            assert input.get_shape() == [1,20,20,256]
            capsules = tf.contrib.layers.conv2d(input, 32 * 8,
                                                9, 2, padding="VALID",
                                                activation_fn=tf.nn.relu)
            capsules = tf.reshape(capsules,(1,-1,8,1))
            capsules = squash(capsules)
            return capsules
        elif layer_type == 'digit':
            input = tf.reshape(input, shape=(1, -1, 1, input.shape[-2].value, 1))
            with tf.variable_scope('routing'):
                b_IJ = tf.constant(np.zeros([1, input.shape[1].value, 10,  1, 1], dtype=np.float32))
                capsules = routing(input,b_IJ)
                capsules = tf.squeeze(capsules,axis=1)
                return capsules
