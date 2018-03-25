from VGG import inference_op, loss
import tensorflow as tf
import pickle
import numpy as np

def weight_variable(shape):
	initial = tf.truncated_normal(shape=shape, stddev=0.1)
	return tf.Variable(initial, name='w')

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name='b')

def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
	n_in = input_op.get_shape()[-1].value
	print('>>>>>>>>>>>', n_in)

	with tf.name_scope(name) as scope:
		# kernel = tf.get_variable(scope + "w",
		# 	shape=[kh, kw, n_in, n_out], dtype=tf.float32,
		# 	initializer=tf.glorot_uniform_initializer(seed=1))
		kernel = weight_variable([kh, kw, n_in, n_out])

		conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1),
							padding='SAME')
		biases = tf.Variable(tf.constant(0.1, shape = [n_out],
										 dtype=tf.float32), name='b')
		z = tf.nn.bias_add(conv, biases)
		activation = tf.nn.relu(z, name = scope)
		# p += [kernel, biases]

		return activation

def openPickle(filename):
	with open(filename, 'rb') as file:
		pkl = pickle.load(file)

	return pkl

if __name__ == '__main__':

	with tf.name_scope('inputs'):
		x = tf.placeholder(tf.float32, [None, 10, 333, 1], name='x_input')
		y = tf.placeholder(tf.int64, [None], name='y_input')
		keep_prob = tf.placeholder(tf.float32, name='keep_prob')
	predictions = inference_op(x, keep_prob)

	loss = loss(predictions, y)
	train_op = tf.train.AdamOptimizer(0.1).minimize(loss)
	
	data = openPickle('trainning_333_1.pickle')
	
	train_x = np.reshape(data[0:1000, :3330], (-1, 10, 333, 1))
	train_y = np.reshape(data[0:1000, 3330:], (-1))
	print(train_x.shape)
	print(train_y.shape)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(50):
			# print(sess.run(test, feed_dict={x: train_x, p: []}))
			# print(sess.run)
			_, loss_value, predict = sess.run([train_op, loss, predictions], feed_dict={x: train_x, y: train_y,
			 keep_prob:1.0})
			print(loss_value, predict[1,:])
