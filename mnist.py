# -*-coding:utf-8-*
"""
神经网络：一个隐藏层
预测类别：tf.nn.softmax()
损失函数：平方差 tf.reduce_mean(tf.square(y - prediction))
梯度下降：tf.train.GradientDescentOptimizer(0.2).minimize(loss)
准确度：
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets("MNIST.data", one_hot=True)
batch_size = 100
batch_num = mnist_data.train.num_examples//batch_size

#训练数据 图片大小：28*28
x = tf.placeholder(tf.float32, [None, 784])
# 标签， 数字0-9，共10个类别
y = tf.placeholder(tf.float32, [None, 10])

weights = {
    'hidden_1': tf.Variable(tf.random_normal([784, 256])),
    'out': tf.Variable(tf.random_normal([256, 10]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([256])),
    'out': tf.Variable(tf.random_normal([10]))
}


def neural_network(x):
    hidden_layer_1 = tf.add(tf.matmul(x, weights['hidden_1']), biases['b1'])
    out_layer = tf.matmul(hidden_layer_1, weights['out']) + biases['out']
    return out_layer


result = neural_network(x)
# 预测类别
prediction = tf.nn.softmax(result)
# 平方差损失函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 梯度下降法（学习率设置为0.2）： 对loss进行优化
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# argmax函数返回最大值所在的位置
# 使用equal函数与正确的类别进行比较，返回一个bool数组
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# cast将bool转化为float，True为1， False为0
# 对所有预测结果求平均值，算出最后的准确率
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


step_num = 400
with tf.Session() as sess:
    sess.run(init)
    for step in range(step_num+1):
        for batch in range(batch_num):
            batch_x, batch_y = mnist_data.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: mnist_data.test.images, y: mnist_data.test.labels})
        print("Step" + str(step) + ",Training Accuracy " + "{:.3f}" + str(acc))
    print("Finished!")

