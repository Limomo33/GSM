from __future__ import print_function
import tensorflow
from keras import activations, initializers, constraints, regularizers
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
from keras.layers import Dropout, LeakyReLU
import tensorflow as tf
import numpy as np

from tensorflow.python.keras import backend


class GraphAttention(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self,
                 F_,
                 atten_heads,
                 dropout_rate=0.3,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphAttention, self).__init__(**kwargs)
        self.F_ = F_
        self.atten_heads=atten_heads
        self.dropout_rate = dropout_rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        self.kernel = []  # Layer kernels for attention heads
        self.biases = []  # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads




    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes

        output_shape = (features_shape[0], features_shape[1], self.F_)
        return [output_shape,(features_shape[0],features_shape[1],features_shape[1])]  # (batch_size, output_dim)

    def build(self, input_shapes):
        #print('input shapes is:' + input_shapes)
        #assert isinstance(input_shapes, list)
        #features_shape = input_shapes[0]
        # assert len(features_shape) == 3
        F = input_shapes[2]
        for head in range(self.atten_heads):

            kernel = self.add_weight(shape=(F, self.F_),
                                          initializer=self.kernel_initializer,
                                          name='kernel_{}'.format(head),
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            self.kernel.append(kernel)
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_,),
                                            initializer=self.bias_initializer,
                                            name='bias_{}'.format(head),
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
                self.biases.append(bias)

            # else:
            #     self.bias = None

            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                                    initializer=self.attn_kernel_initializer,
                                                    regularizer=self.attn_kernel_regularizer,
                                                    constraint=self.attn_kernel_constraint,
                                                    name='attn_kernel_self_{}'.format(head))
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                      initializer=self.attn_kernel_initializer,
                                                      regularizer=self.attn_kernel_regularizer,
                                                      constraint=self.attn_kernel_constraint,
                                                      name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self,attn_kernel_neighs])

            self.built = True

    def call(self, inputs, mask=None):
        X = inputs
        outputs=[]
        for head in range(self.atten_heads):
            kernel=self.kernel[head]
            attn_kernel=self.attn_kernels[head]
            features = K.dot(X, kernel)  # (b,n,f')
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attn_kernel[0])  # (b,n,1)
            attn_for_neighs = K.dot(features, attn_kernel[1])  # (b,n,1)
            # Attention (Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + tf.transpose(attn_for_neighs, [0, 2, 1])  # (b,n,n)
            dense = LeakyReLU(alpha=0.2)(dense)
            # mask = -10e9 * (1.0 - A)
            # dense += mask
            dense = K.softmax(dense)  # (b,n,n)
            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(self.dropout_rate)(dense)  # (b,n,n)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (b,n,f')
            # Linear combination with neighbors' features
            dropout_feat = tf.transpose(dropout_feat, [0, 2, 1])  # (b,f',n)
            # Add output of attention head to final output
            node_features = K.batch_dot(dropout_attn, dropout_feat, axes=(1, 2))  # (b,n,f')


            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            outputs.append(node_features)
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        return [self.activation(output), dense]

    def get_config(self):
        config = {'F_': self.F_,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'atten_heads': self.atten_heads,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'attn_kernel_initializer': initializers.serialize(
                      self.attn_kernel_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'attn_kernel_regularizer':regularizers.serialize(
                      self.activity_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(
                      self.bias_constraint),
                  'attn_kernel_constraint': constraints.serialize(
                      self.attn_kernel_constraint)
        }

        base_config = super(GraphAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class GraphAttention2(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self,
                 F_,
                 atten_heads,
                 dropout_rate=0.3,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphAttention2, self).__init__(**kwargs)
        self.F_ = F_
        self.atten_heads=atten_heads
        self.dropout_rate = dropout_rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        self.kernel = []  # Layer kernels for attention heads
        self.biases = []  # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads




    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], features_shape[1], self.F_)
        return [output_shape,(features_shape[0],features_shape[1],features_shape[2])]  # (batch_size, output_dim)

    def build(self, input_shapes):
        #features_shape = input_shapes[0]
        print(input_shapes)
        #assert isinstance(input_shapes, list)
        # assert len(features_shape) == 3
        F = input_shapes[0][2]
        for head in range(self.atten_heads):
            kernel = self.add_weight(shape=(F, self.F_),
                                          initializer=self.kernel_initializer,
                                          name='kernel_{}'.format(head),
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            self.kernel.append(kernel)
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_,),
                                            initializer=self.bias_initializer,
                                            name='bias_{}'.format(head),
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
                self.biases.append(bias)

            # else:
            #     self.bias = None

            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                                    initializer=self.attn_kernel_initializer,
                                                    regularizer=self.attn_kernel_regularizer,
                                                    constraint=self.attn_kernel_constraint,
                                                    name='attn_kernel_self_{}'.format(head))
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                      initializer=self.attn_kernel_initializer,
                                                      regularizer=self.attn_kernel_regularizer,
                                                      constraint=self.attn_kernel_constraint,
                                                      name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self,attn_kernel_neighs])

            self.built = True

    def call(self, inputs, mask=None):
        X = inputs[0]
        A=inputs[1]
        outputs=[]
        for head in range(self.atten_heads):
            kernel=self.kernel[head]
            attn_kernel=self.attn_kernels[head]
            features = K.dot(X, kernel)  # (b,n,f')
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attn_kernel[0])  # (b,n,1)
            attn_for_neighs = K.dot(features, attn_kernel[1])  # (b,n,1)
            # Attention (Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + tf.transpose(attn_for_neighs, [0, 2, 1])  # (b,n,n)
            dense = LeakyReLU(alpha=0.2)(dense)
            mask = -10e9 * (0.05 - A)
            dense += mask
            dense = K.softmax(dense)  # (b,n,n)
            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(self.dropout_rate)(dense)  # (b,n,n)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (b,n,f')
            # Linear combination with neighbors' features
            dropout_feat = tf.transpose(dropout_feat, [0, 2, 1])  # (b,f',n)
            # Add output of attention head to final output
            node_features = K.batch_dot(dropout_attn, dropout_feat, axes=(1, 2))  # (b,n,f')


            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            outputs.append(node_features)
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        return [self.activation(output), dense]

    def get_config(self):
        config = {'F_': self.F_,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'atten_heads': self.atten_heads,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'attn_kernel_initializer': initializers.serialize(
                      self.attn_kernel_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'attn_kernel_regularizer':regularizers.serialize(
                      self.activity_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(
                      self.bias_constraint),
                  'attn_kernel_constraint': constraints.serialize(
                      self.attn_kernel_constraint)
        }

        base_config = super(GraphAttention2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))