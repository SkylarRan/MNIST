"""
神经网络：无隐藏层
预测类别：tf.nn.softmax()
损失函数：交叉熵 -tf.reduce_sum(y * tf.log(prediction))
梯度下降：tf.train.GradientDescentOptimizer(0.01).minimize(loss)
准确度：0.91
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST.data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 预测值
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 交叉熵: 计算所有图片的交叉熵的总和
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 梯度下降
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    # 使用一小部分的随机数据来进行训练， 随机梯度下降训练
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(i, sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))
