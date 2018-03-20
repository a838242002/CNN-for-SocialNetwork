from Layers import conv_op, mpool_op, fc_op

def inference_op(input_op, keep_prob):
	p = []

	conv1_1 = conv_op(input_op, name="conv1_1", kh= , kw= , n_out= , dh= , dw= , p=p)
	conv1_2 = conv_op(conv1_1, name="conv1_2", kh= , kw= , n_out= , dh= , dw= , p=p)
	pool1 = mpool_op(conv1_2, name="pool1", kh= , kw= , dw= , dh= )

	conv2_1 = conv_op(pool1, name="conv2_1", kh= , kw= , n_out= , dh= , dw= , p=p)
	conv2_2 = conv_op(conv2_1, name="conv2_2", kh= , kw= , n_out= , dh= , dw= , p=p)
	pool2 = mpool_op(conv2_2, name="pool2", kh= , kw= , dw= , dh= )

	conv3_1 = conv_op(pool2, name="conv3_1", kh= , kw= , n_out= , dh= , dw= , p=p)
	conv3_2 = conv_op(conv3_1, name="conv3_2", kh= , kw= , n_out= , dh= , dw= , p=p)
	conv3_3 = conv_op(conv3_2, name="conv3_3", kh= , kw= , n_out= , dh= , dw= , p=p)
	pool3 = mpool_op(conv3_3, name="pool3", kh= , kw= , dw= , dh= )

	conv4_1 = conv_op(pool3, name="conv4_1", kh= , kw= , n_out= , dh= , dw= , p=p)
	conv4_2 = conv_op(conv4_1, name="conv4_2", kh= , kw= , n_out= , dh= , dw= , p=p)
	conv4_3 = conv_op(conv4_2, name="conv4_3", kh= , kw= , n_out= , dh= , dw= , p=p)
	pool4 = mpool_op(conv4_3, name="pool4", kh= , kw= , dw= , dh= )

	conv5_1 = conv_op(pool4, name="conv5_1", kh= , kw= , n_out= , dh= , dw= , p=p)
	conv5_2 = conv_op(conv5_1, name="conv5_2", kh= , kw= , n_out= , dh= , dw= , p=p)
	conv5_3 = conv_op(conv5_2, name="conv5_3", kh= , kw= , n_out= , dh= , dw= , p=p)
	pool5 = mpool_op(conv5_3, name="pool5", kh= , kw= , dw= , dh= )

	shape = pool5.get_shape()
	flattened_shape = shape[1].value * shape[2].value * shape[3].value
	reshape1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

	fc6 = fc_op(reshape1, name="fc6", n_out= , p=p)
	fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

	fc7 = fc_op(fc6_drop, name="fc7", n_out= , p=p)
	fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

	fc8 = fc_op(fc7_drop, name="fc8", n_out= , p=p)
	softmax = tf.nn.softmax(fc8)
	predictions = tf.argmax(softmax, 1)
	return predictions, softmax, fc8, p