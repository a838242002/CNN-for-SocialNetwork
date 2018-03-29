from Layers import conv_op, mpool_op, fc_op
import tensorflow as tf

def inference_op(input_op, keep_prob):
	p = []

	conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=16, dh=1, dw=1, p=p)
	conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=16, dh=1, dw=1, p=p)
	pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

	conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=32, dh=1, dw=1, p=p)
	conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=32, dh=1, dw=1, p=p)
	pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dw=2, dh=2)

	conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
	conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
	conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
	pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dw=2, dh=2)

	conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
	conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
	conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
	pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dw=2, dh=2)

	conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
	conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
	conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
	pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)

	shape = pool5.get_shape()
	flattened_shape = shape[1].value * shape[2].value * shape[3].value
	reshape1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

	fc6 = fc_op(reshape1, "fc6", 512, p)
	fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

	fc7 = fc_op(fc6_drop, name="fc7", n_out=256, p=p)
	fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

	fc8 = fc_op(fc7_drop, name="fc8", n_out=2, p=p, activation=False)

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
	with tf.name_scope('Accuracy') as scope:
		pred = tf.nn.softmax(logits)
		correct_prediction = tf.equal(tf.argmax(pred, 1), labels, name='accuracy_pre_example')
		evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_rate')
		return evaluation_step



