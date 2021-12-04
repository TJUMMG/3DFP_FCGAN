import sys
import numpy as np
import tensorflow as tf

if sys.version_info.major == 3:
    xrange = range


def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)


def conv2d(inputs, name, out_channels, bn = False, is_training = False ,activation=False, ksize=3, stride = 1):
    with tf.variable_scope(name):
        in_channels = inputs.get_shape()[-1]
        filter = tf.get_variable('weight', shape=[ksize, ksize, in_channels, out_channels],
                                 initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(inputs, filter, strides=[1, stride, stride, 1], padding='SAME')
        bias = tf.get_variable('bias', shape=[out_channels], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, bias)

        if activation:
            conv = tf.nn.relu(conv)

        tf.add_to_collection('weights', filter)

        return conv

def deconv2d(inputs, name, out_channels, bn = False, is_training = False ,activation=False, ksize = 4, stride = 2):
    with tf.variable_scope(name):
        input_shape = inputs.get_shape()
        in_channels = input_shape[-1]
        input_shape = tf.shape(inputs)
        filter = tf.get_variable('weight', shape=[ksize, ksize, out_channels, in_channels],
                                     initializer=tf.contrib.layers.xavier_initializer())
        output_shape = [input_shape[0], input_shape[1] * stride, input_shape[2] * stride, out_channels]
        deconv = tf.nn.conv2d_transpose(inputs, filter, output_shape, [1, stride, stride, 1])
        bias = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, bias)

        if activation:
            deconv = tf.nn.relu(deconv)

        tf.add_to_collection('weights', filter)
        return deconv



def resblock(inputs,out_channels,is_training,name):
    with tf.variable_scope(name):
        in_channels = inputs.get_shape().as_list()[-1]

        x = conv2d(inputs,'conv1',out_channels,bn=False,is_training=is_training,activation=True)
        x = conv2d(x,'conv2',out_channels,bn=False,is_training=is_training,activation=False)

        if in_channels!=out_channels:
            inputs = conv2d(inputs,'conv_fuse',out_channels,bn=False,is_training=is_training,activation=True)
        x = inputs + x
        return x


def conv_block(x,output_num,is_training,name):
    with tf.variable_scope(name):
        inp = x
        x1 = tf.nn.relu(inp)
        x1 = conv2d(x1,name='conv1',out_channels=output_num//2)

        x2 = tf.nn.relu(x1)
        x2 = conv2d(x2,name='conv2',out_channels=output_num//4)


        x3 = tf.nn.relu(x2)
        x3 = conv2d(x3,name='conv3',out_channels=output_num//4)

        x3 = tf.concat([x1,x2,x3],axis=-1)

        input_num = inp.get_shape()[-1]
        if input_num != output_num:
            inp = tf.nn.relu(inp)
            inp = conv2d(inp, name='downsample', out_channels=output_num, ksize=1)
        return x3 + inp


def hour_glass(x, output_num, depth, is_training, name):
    with tf.variable_scope(name + '_%d' % depth):
        if depth <= 0:
            return x
        up1 = x
        up1 = conv_block(up1, output_num, is_training, 'cb1')

        low1 = tf.layers.average_pooling2d(x, 2, strides=2)
        low1 = conv_block(low1, output_num, is_training, 'cb2')

        if depth > 1:
            low2 = hour_glass(low1, output_num, depth-1, is_training, name)
        else:
            low2 = low1
            low2 = conv_block(low2, output_num, is_training, 'cb3')

        low3 = low2
        low3 = conv_block(low3, output_num, is_training, 'cb4')

        up2 = tf.image.resize_bilinear(low3, tf.shape(low3)[1:3] * 2)

        return up1 + up2


