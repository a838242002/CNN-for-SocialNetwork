import tensorflow as tf
import pickle
import numpy as np
import AlexNet, VGG, Layers
import TFRecord

def openPickle(filename):
	with open(filename, 'rb') as file:
		pkl = pickle.load(file)

	return pkl

if __name__ == '__main__':

	IMAGE_HEIGHT = 300
	IMAGE_WIDTH = 333
	IMAGE_CHANNEL = 1

	with tf.name_scope('inputs'):
		x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], name='x_input')
		y = tf.placeholder(tf.int64, [None], name='y_input')
		keep_prob = tf.placeholder(tf.float32, name='keep_prob')
	predictions = AlexNet.inference_op(x, keep_prob)

	loss = AlexNet.loss(predictions, y)
	train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
	
	# data = openPickle('trainning_333_1.pickle')
	logits = AlexNet.get_logits(predictions)
	# # test = Layers.kernal_variable('conv1', [3, 3, 1, 1])
	# train_x = np.reshape(data[0:10, :3330], (-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
	# train_y = np.reshape(data[0:10,3330:], (-1))
	# print(train_x.shape)
	# print(train_y.shape)

	# with tf.Session() as sess:
	# 	init = tf.global_variables_initializer()
	# 	sess.run(init)

	# 	for i in range(100):
	# 		# print(sess.run(y, feed_dict={})
	# 		# print(sess.run(p, feed_dict={x: train_x, keep_prob:1.0}))
	# 		# print(sess.run(logits, feed_dict={x: train_x, y: train_y,
	# 		#  keep_prob:1.0}))
	# 		_, loss_value = sess.run([train_op, loss], feed_dict={x: train_x, y: train_y,
	# 		 keep_prob:1.0})
	# 		print(loss_value)

	filename_queue = tf.train.string_input_producer(
		['test2.tfrecords'], num_epochs=10)
	images, labels = TFRecord.read_and_decode(filename_queue, 300, 333)

	
	
	with tf.Session() as sess:
		init_op = tf.group(tf.global_variables_initializer(),
						tf.local_variables_initializer())
		
		sess.run(init_op)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		for i in range(1000):
			img, lab = sess.run([images, labels])

			_, loss_value = sess.run([train_op, loss], feed_dict={x: img, y: lab, keep_prob:1.0})
			# print(sess.run(logits, feed_dict={x: img, y: lab, keep_prob:1.0}))

			print(loss_value)

		coord.request_stop()
		coord.join(threads)
