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
	

	for record in data:
		

		height = data.shape[1]
		width = data.shape[2]

		label = labels[count, 0]
		image_string = record.tostring()

		feature = {}
		
		feature['height']= tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
		feature['width']= tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
		feature['image_string']= tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
		feature['label']= tf.train.Feature(float_list = tf.train.FloatList(value=[label]))
			
		

		# print('height >> {0}, width >> {1}'.format(height, width))
		features=tf.train.Features(feature=feature)
		example = tf.train.Example(features=features)

		writer.write(example.SerializeToString())

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

def read_and_decode(filename_queue, IMAGE_HEIGHT, IMAGE_WIDTH, comp=True, b_size=32):

	compression = tf.python_io.TFRecordCompressionType.GZIP
	if comp == True:
		reader = tf.TFRecordReader(options=tf.python_io.TFRecordOptions(compression))
	else:
		reader = tf.TFRecordReader()

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
		batch_size=32,
		capacity=1000,
		num_threads=1,
		min_after_dequeue=500)

	return images, labels

def parse_function(proto):

	features = tf.parse_single_example(
		    proto,
		    features={
		      'height': tf.FixedLenFeature((), tf.int64),
		      'width': tf.FixedLenFeature((), tf.int64),
		      'image_string': tf.FixedLenFeature((), tf.string),
		      'label': tf.FixedLenFeature((), tf.float32)
		      })

	image = tf.decode_raw(features['image_string'], tf.float32)

	label = tf.cast(features['label'], tf.float32)

	height = tf.cast(features['height'], tf.int32)
	width = tf.cast(features['width'], tf.int32)

	image = tf.reshape(image, [height, width, 1])

	resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
		target_height=300,
		target_width=333)

	return resized_image, label

if __name__ == '__main__':

	while(True):
		c1 = "1: Convert to TFRecord with gzip comprassion"
		c2 = "2: Read TFRecord"
		c3 = "3: Exit"
		choose = int(input(c1 + "\n" + c2 + "\n" + c3 + "\n***>>>===>>>***"))
		if(choose == 1):
			with open('trainning_333_1.pickle', 'rb') as file:
				data = pickle.load(file)
			# 	# test_data = np.reshape(data[:, :3330], (-1, 10, 333)).astype(np.float32)
			# training_data = np.repeat(np.reshape(data[0:7119, :3330], (-1, 10, 333)), 30, axis=1).astype(np.float32)
			# training_labels = data[0:7119, 3330:]
			# print(training_data.shape)
			# write_tfrecord('training7119.tfrecords', training_data, training_labels)

			# validation_data = np.repeat(np.reshape(data[7119:8136, :3330], (-1, 10, 333)), 30, axis=1).astype(np.float32)
			# validation_labels = data[7119:8136, 3330:]
			# print(validation_data.shape)
			# write_tfrecord('validation1017.tfrecord', validation_data, validation_labels)

			# test_data = np.repeat(np.reshape(data[8136:, :3330], (-1, 10, 333)), 30, axis=1).astype(np.float32)
			# test_labels = data[8136:, 3330:]
			# print(test_data.shape)
			# write_tfrecord('test_data2034.tfrecords', test_data, test_labels)
			
			# print(labels[1,:])
		elif(choose == 2):
			read_tfrecord('training7119.tfrecords')
		elif(choose == 3):
			break
		else:
			print("Choose again.")

	