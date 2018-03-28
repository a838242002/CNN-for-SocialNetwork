import tensorflow as tf
import pickle
import numpy as np
import AlexNet, VGG, Layers
import TFRecord

def openPickle(filename):
	with open(filename, 'rb') as file:
		pkl = pickle.load(file)

	return pkl

def input_img_lab(height, width, channel):
	with tf.name_scope('inputs'):
		x = tf.placeholder(tf.float32, [None, height, width, channel], name='x_input')
		y = tf.placeholder(tf.int64, [None], name='y_input')
		keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	return x, y, keep_prob


if __name__ == '__main__':

	IMAGE_HEIGHT = 300
	IMAGE_WIDTH = 333
	IMAGE_CHANNEL = 1
	MODEL = AlexNet

	x, y, keep_prob = input_img_lab(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)
	predictions = MODEL.inference_op(x, keep_prob)

	loss = MODEL.loss(predictions, y)
	train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
	
	logits = MODEL.get_logits(predictions)
	accuracy = MODEL.predict(logits, y)

	filename_queue = tf.train.string_input_producer(
		['test2.tfrecords'], num_epochs=10)
	images, labels = TFRecord.read_and_decode(filename_queue, IMAGE_HEIGHT, IMAGE_WIDTH)

	
	
	with tf.Session() as sess:
		init_op = tf.group(tf.global_variables_initializer(),
						tf.local_variables_initializer())
		
		sess.run(init_op)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		for i in range(5000):
			img, lab = sess.run([images, labels])
			# print(sum(lab))
			
			_, loss_value = sess.run([train_op, loss], feed_dict={x: img, y: lab, keep_prob:0.7})
			# print(sess.run(logits, feed_dict={x: img, y: lab, keep_prob:1.0}))
			if i % 10 == 0:
				pred = sess.run(accuracy, feed_dict={x: img, y:lab, keep_prob:0.7})
				print('loss >> {0}, pred >> {1}'.format(loss_value, pred))

		coord.request_stop()
		coord.join(threads)
