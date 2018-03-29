import tensorflow as tf
# 为数据操作
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import TFRecord

# # 精度3位
# np.set_printoptions(precision=3)
# # 用于显示数据
# def display(alist, show = True):
#     print('type:%s\nshape: %s' %(alist[0].dtype,alist[0].shape))
#     if show:
#         for i in range(3):
#             print('样本%s\n%s' %(i,alist[i]))

# scalars = np.array([1,2,3], dtype=np.int64)
# print('\n标量')
# display(scalars)

# vectors = np.array([[0.1,0.1,0.1],
#                    [0.2,0.2,0.2],
#                    [0.3,0.3,0.3]],dtype=np.float32)
# print('\n向量')
# display(vectors)

# matrices = np.array([np.array((vectors[0],vectors[0])),
#                     np.array((vectors[1],vectors[1])),
#                     np.array((vectors[2],vectors[2]))],dtype=np.float32)
# print('\n矩阵')
# display(matrices)

# # shape of image：(806,806,3)
# img=mpimg.imread('YJango.jpg') # 我的头像
# tensors = np.array([img,img,img])
# # show image
# print('\n张量')

# writer = tf.python_io.TFRecordWriter('%s.tfrecord' %'test')
# # 这里我们将会写3个样本，每个样本里有4个feature：标量，向量，矩阵，张量
# for i in range(3):
#     # 创建字典
#     features={}
#     # 写入标量，类型Int64，由于是标量，所以"value=[scalars[i]]" 变成list
#     features['scalar'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[scalars[i]]))
    
#     # 写入向量，类型float，本身就是list，所以"value=vectors[i]"没有中括号
#     features['vector'] = tf.train.Feature(float_list = tf.train.FloatList(value=vectors[i]))
    
#     # 写入张量，类型float，本身是三维张量，另一种方法是转变成字符类型存储，随后再转回原类型
#     features['tensor']         = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensors[i].tostring()]))
#     # 存储丢失的形状信息(806,806,3)
#     features['tensor_shape'] = tf.train.Feature(int64_list = tf.train.Int64List(value=tensors[i].shape))

#     # 将存有所有feature的字典送入tf.train.Features中
#     tf_features = tf.train.Features(feature= features)
#     # 再将其变成一个样本example
#     tf_example = tf.train.Example(features = tf_features)
#     # 序列化该样本
#     tf_serialized = tf_example.SerializeToString()
#     # 写入一个序列化的样本
#     writer.write(tf_serialized)
#     # 由于上面有循环3次，所以到此我们已经写了3个样本
#     # 关闭文件    
# writer.close()

# 从多个tfrecord文件中导入数据到Dataset类 （这里用两个一样）
filenames = ["training7119.tfrecords"]
dataset = tf.data.TFRecordDataset(filenames,compression_type='GZIP')
new_dataset = dataset.map(TFRecord.parse_function)
# 创建获取数据集中样本的迭代器
iterator = new_dataset.make_one_shot_iterator()
# 获得下一个样本
next_element = iterator.get_next()
# 创建Session
sess = tf.InteractiveSession()

# 获取
i = 1
while True:
    # 不断的获得下一个样本

    label, img = sess.run(next_element)
    print(img)
    i+=1