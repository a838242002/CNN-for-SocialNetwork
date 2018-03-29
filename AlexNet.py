from Layers import conv_op, mpool_op, fc_op
import tensorflow as tf

def inference_op(input_op, keep_prob):
	p = []

	conv1 = conv_op(input_op, name="conv1", kh=5, kw=5, n_out=64, dh=2, dw=2, p=p)
	pool1 = mpool_op(conv1, name="pool1", kh=1, kw=2, dw=2, dh=2)

	conv2 = conv_op(pool1, name="conv2", kh=5, kw=5, n_out=192, dh=1, dw=1, p=p)
	pool2 = mpool_op(conv2, name="pool2", kh=3, kw=3, dw=2, dh=2)

	conv3 = conv_op(pool2, name="conv3", kh=3, kw=3, n_out=256, dh=2, dw=2, p=p)

	conv4 = conv_op(pool3, name="conv4", kh=3, kw=3, n_out=256, dw=2, dh=2, p=p)

	conv5 = conv_op(pool4, name="conv5", kh=3, kw=3, n_out=256, dw=2, dh=2, p=p)

	shape = pool5.get_shape()
	flattened_shape = shape[1].value * shape[2].value * shape[3].value
	reshape1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

	fc6 = fc_op(reshape1, "fc6", 128, p)
	fc7 = fc_op(fc6, "fc7", 64, p)
	fc8 = fc_op(fc7, "fc8", 2, p, False)

	return fc8

def get_logits(logits):
	return logits

def loss(logits, labels):
	with tf.name_scope('loss_function') as scope:
		labels = tf.one_hot(tf.cast(labels, tf.int64), 2)
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
			logits=logits, labels=labels, name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
		return tf.add_n(tf.get_collection('losses'), name='total_loss')

def predict(logits, labels):
	with tf.name_scope('accuracy') as scope:
		pred = tf.nn.softmax(logits)
		correct_prediction = tf.equal(tf.argmax(pred, 1), labels, name='accuracy_pre_example')
		evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_rate')
		return evaluation_step

