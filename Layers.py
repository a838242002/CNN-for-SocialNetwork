from datetime import datetime
import math
import time
import tensorflow as tf

# input_op > tensor, name > layer name, kh > kernel height, 
# kw > kernel > kernel width, n_out > kernel channels, dh > stride height, dw > stride width
def kernal_variable(scope, shape):
	kernal = tf.get_variable(scope + "w",
			shape=shape, dtype=tf.float32,
			initializer=tf.glorot_uniform_initializer(seed=100))
	return kernal

def bias_variable(scope, shape):
	
	initial=tf.constant(0.1, shape=shape, name='b')
	return tf.Variable(initial)

def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
	n_in = input_op.get_shape()[-1].value

	with tf.name_scope(name) as scope:
		kernel = kernal_variable(scope, [kh, kw, n_in, n_out])

		conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1),
							padding='SAME')
		biases = bias_variable(scope, [n_out])
		
		z = tf.nn.bias_add(conv, biases)
		activation = tf.nn.relu(z, name = scope)
		p += [kernel, biases]

		return activation

def fc_op(input_op, name, n_out, p, activation=True, keep_prob=1):
	n_in = input_op.get_shape()[-1].value

	with tf.name_scope(name) as scope:
		kernel = kernal_variable(scope, [n_in, n_out])

		biases = bias_variable(scope, [n_out])

		if activation == True:
			activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
			activation = tf.nn.dropout(activation, keep_prob, name= "drop")
		else:
			activation = tf.matmul(input_op, kernel) + biases
			activation = tf.nn.dropout(activation, keep_prob, name= "drop")
		p += [kernel, biases]

		

		return activation

def mpool_op(input_op, name, kh, kw, dh, dw):
	return tf.nn.max_pool(input_op,
						  ksize=[1, kh, kw, 1],
						  strides=[1, dh, dw, 1],
						  padding='SAME',
						  name=name)

