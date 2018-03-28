import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle



def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def write_tfrecord(filename, data, labels):
	compression = tf.python_io.TFRecordCompressionType.GZIP
	tfrecords_filename = filename

	writer = tf.python_io.TFRecordWriter(tfrecords_filename, options=tf.python_io.TFRecordOptions(compression))

	count = 0
	

	for record in data:
		

		height = data.shape[1]
		width = data.shape[2]

		label = labels[count, 0]
		image_string = record.tostring()

		# print('height >> {0}, width >> {1}'.format(height, width))
		example = tf.train.Example(features=tf.train.Features(feature={
			'height': _int64_feature(height),
			'width': _int64_feature(width),
			'image_string': _bytes_feature(image_string),
			'label': _float_feature([label])
			}))

		writer.write(example.SerializeToString())
		print(count)
		count += 1

	writer.close()

def read_tfrecord(filename):
	compression = tf.python_io.TFRecordCompressionType.GZIP
	record_iterator = tf.python_io.tf_record_iterator(path=filename, options=tf.python_io.TFRecordOptions(compression))

	for string_record in record_iterator:
		example = tf.train.Example()

		example.ParseFromString(string_record)

		height = int(example.features.feature['height'].int64_list.value[0])
		width = int(example.features.feature['width'].int64_list.value[0])
		image_string = (example.features.feature['image_string'].bytes_list.value[0])
		label = (example.features.feature['label'].float_list.value[0])

		image_1d = np.fromstring(image_string, dtype=np.float32)
		
		image = np.reshape(image_1d, (height, width))
		
		plt.imshow(image)
		plt.show()

def read_and_decode(filename_queue, IMAGE_HEIGHT, IMAGE_WIDTH):
	compression = tf.python_io.TFRecordCompressionType.GZIP
	reader = tf.TFRecordReader(options=tf.python_io.TFRecordOptions(compression))

	_, serialized_example = reader.read(filename_queue)

	features = tf.parse_single_example(
		    serialized_example,
		    features={
		      'height': tf.FixedLenFeature([], tf.int64),
		      'width': tf.FixedLenFeature([], tf.int64),
		      'image_string': tf.FixedLenFeature([], tf.string),
		      'label': tf.FixedLenFeature([], tf.float32)
		      })

	image = tf.decode_raw(features['image_string'], tf.float32)

	label = tf.cast(features['label'], tf.float32)

	height = tf.cast(features['height'], tf.int32)
	width = tf.cast(features['width'], tf.int32)

	image = tf.reshape(image, [height, width, 1])

	resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
		target_height=IMAGE_HEIGHT,
		target_width=IMAGE_WIDTH)

	images, labels = tf.train.shuffle_batch(
		[resized_image, label],
		batch_size=500,
		capacity=1000,
		num_threads=1,
		min_after_dequeue=500)

	return images, labels


if __name__ == '__main__':
	# with open('trainning_333_1.pickle', 'rb') as file:
	# 	data = pickle.load(file)
	# test_data = np.repeat(np.reshape(data[:, :3330], (-1, 10, 333)), 30, axis=1).astype(np.float32)
	
	

	# labels = data[:, 3330:]
	# print(labels[1,:])
	# write_tfrecord('test2.tfrecords', test_data, labels)
	# read_tfrecord('test2.tfrecords')

	filename_queue = tf.train.string_input_producer(
		['test2.tfrecords'], num_epochs=10)
	images, labels = read_and_decode(filename_queue, 300, 333)

	init_op = tf.group(tf.global_variables_initializer(),
						tf.local_variables_initializer())
	
	with tf.Session() as sess:
		
		
		sess.run(init_op)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		for i in range(3):

			img, lab = sess.run([images, labels])

			print(img.shape)

		coord.request_stop()
		coord.join(threads)

	