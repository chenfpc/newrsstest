import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MINST_data/',one_hot=True)
#
# # #超参数
# learning_rate = 0.5
# epochs = 10
# batch_size = 100
#
# tf.reset_default_graph()
# #placeholder
# x = tf.placeholder(tf.float32,[None,784])
# y = tf.placeholder(tf.float32,[None,10])
#
# w1 = tf.Variable(tf.random_normal([784,300],stddev=0.03),name="W1")
# b1 = tf.Variable(tf.random_normal([300]),name="b1")
#
# w2 = tf.Variable(tf.random_normal([300,10],stddev=0.03),name="W2")
# b2 = tf.Variable(tf.random_normal([10]),name="b2")
#
# hidden_out = tf.add(tf.matmul(x,w1), b1)
# hidden_out = tf.nn.relu(hidden_out)
# y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, w2), b2))
#
# y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
# cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
#
# # init operator
# init_op = tf.global_variables_initializer()
#
# # 创建准确率节点
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# # 创建session
# with tf.Session() as sess:
#     # 变量初始化
#     sess.run(init_op)
#     total_batch = int(len(mnist.train.labels) / batch_size)
#     for epoch in range(epochs):
#         avg_cost = 0
#         for i in range(total_batch):
#             batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
#             _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
#             avg_cost += c / total_batch
#         print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))
#     print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
BASE_URL = r'C:\Users\chenf\Desktop\\'
# 训练数据 478
trainingAll = BASE_URL+"all.txt"
trainingAll = np.loadtxt(trainingAll)
training2_4g = BASE_URL + 'all2_4.txt'
training2_4g = np.loadtxt(training2_4g)
training5g = BASE_URL + 'all5.txt'
training5g = np.loadtxt(training5g)

# 测试数据  74
testingAll = BASE_URL + 'test_all.txt'
testingAll = np.loadtxt(testingAll)
testing2_4g = BASE_URL + 'test_2.4g.txt'
testing2_4g = np.loadtxt(testing2_4g)
testing5g = BASE_URL + 'test_5g.txt'
testing5g = np.loadtxt(testing5g)

# 坐标数据
cordinaryAll = BASE_URL + 'position.txt'
cordinaryAll = np.loadtxt(cordinaryAll)
cordinaryTest = BASE_URL + 'position_test.txt'
cordinaryTest = np.loadtxt(cordinaryTest)

input1 = tf.placeholder(tf.float32,shape=[None,12],name="24g")
input2 = tf.placeholder(tf.float32,shape=[None,12],name="5g")

#占位符
y1 = tf.placeholder(tf.float32,shape=[1],name='y1')
y2 = tf.placeholder(tf.float32,shape=[1],name='y2')
x1 = tf.placeholder(tf.float32,shape=[1],name='x1')
x2 = tf.placeholder(tf.float32,shape=[1],name='x2')


w1 = tf.Variable(tf.random_normal([12,5],stddev=0.03),name="W1")
b1 = tf.Variable(tf.random_normal([5]),name="b1")

w2 = tf.Variable(tf.random_normal([5,1],stddev=0.03),name="W2")
b2 = tf.Variable(tf.random_normal([1]),name="b2")

hidden_out = tf.add(tf.matmul(x,w1), b1)
hidden_out = tf.nn.relu(hidden_out)
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, w2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# init operator
init_op = tf.global_variables_initializer()

# 创建准确率节点
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 创建session
with tf.Session() as sess:
    # 变量初始化
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))