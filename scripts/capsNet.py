#!/usr/bin/env python
# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np
from capsLayer import CapsLayer
from PIL import Image


def getWeights(shape, stddev=0.01):
    var = tf.get_variable(
        'weights',
        shape,
        initializer=tf.truncated_normal_initializer(stddev=stddev))
    return var
    
class CapsNet(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.image_placeholder = tf.placeholder(tf.float32,
                                   shape=[28,28,3],
                                   name='input_image')
            self.img = tf.expand_dims(self.image_placeholder,0)
            self.buildArch()
            self.pred = tf.reduce_sum(tf.square(self.caps2), -2, keep_dims=True)
            
    def buildArch(self):
        with tf.variable_scope('Conv1_layer'):# as scope:
            print self.img
            conv1 = tf.contrib.layers.conv2d(self.img,num_outputs=256,
                                         kernel_size=9,stride=1,
                                         padding='VALID')
            assert conv1.get_shape() == [1,20,20,256]
            print conv1
        with tf.variable_scope('PrimaryCaps_layer'):# as scope:
            primary_caps = CapsLayer()
            caps1 = primary_caps(conv1,'primary')
            assert caps1.get_shape() == [1,1152,8,1]
        with tf.variable_scope('DigitCaps_layer'):# as scope:
            digit_caps = CapsLayer()
            self.caps2 = digit_caps(caps1,'digit')
        
    def inference(self):
        return self.pred
