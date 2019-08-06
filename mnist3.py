"""
神经网络：两层卷积+密集连接层+dropout
预测类别：tf.nn.softmax()
损失函数：交叉熵
梯度下降：ADAM优化器来做梯度最速下降
准确度：
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets("MNIST.data", one_hot=True)

sess = tf.InteractiveSession()

# x,y是占位符，有了shape参数，Tensorflow能够自动捕捉因数据维度不一致导致的错误
# x是一个2维的浮点数张量，None表示其值大小不定，作为第一维度，代表batch的大小，784指一张展平的MNIST图片（28*28）
# y是一个2维的浮点数张量，每一行为一个10维的one-hot向量，代表图片类别（数字0-9）
X = tf.placeholder("float", shape=[None, 784])
Y = tf.placeholder("float", shape=[None, 10])


# 权重初始化，加入少量的噪声来打破对称性以及避免0梯度
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置初始化，由于使用的是ReLU神经元，因此较好的做法是用一个较小的正数来初始化偏置，以避免神经元节点输出恒为0的问题
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积使用1步长，0边距的模版，保证输出和输入是同一个大小
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 池化用简单传统的2x2大小的模版做max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#           第一层卷积：由一个卷积接一个max pooling完成
# 卷积在每个5x5的patch中算出32个特征，1表示输入的通道数目，32表示输出的通道数目
W_conv1 = weight_variable([5, 5, 1, 32])
# 每一个输出通道都有一个对应的偏置量
b_conv1 = bias_variable([32])

# 为了用第一层卷积，要把x编程一个4d向量， 第2，3维对应图片的宽高，第4维代表图片的颜色通道数（灰度图维1，rgb彩色图为3）
x_image = tf.reshape(X, [-1, 28, 28, 1])

# 将x_image和权重向量进行卷积，加上偏置项， 应用ReLU激活函数，进行max pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#           第二层卷积：5x5的patch得到64个特征
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#           密集连接层：图片尺寸减小到7x7
# 加入一个有1024个神经元的全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减少过拟合，在输出层之前加入dropout
# 用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#           输出层：softmax regression
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 损失函数：目标类别与预测类别之间的交叉熵
cross_entropy = -tf.reduce_sum(Y * tf.log(y_conv))

# 使用ADAM优化器来做梯度最速下降
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction), "float")

sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch = mnist_data.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={X: batch[0], Y: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={X: mnist_data.test.images,
                                                    Y: mnist_data.test.labels,
                                                    keep_prob: 1.0}))
