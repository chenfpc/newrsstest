import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
BASE_URL = r'C:\Users\chenf\Desktop\\'
# 训练数据 478
training2_4g = BASE_URL + 'all2_4.txt'
training2_4g = np.loadtxt(training2_4g)

training5g = BASE_URL + 'all5.txt'
training5g = np.loadtxt(training5g)

testing2_4g = BASE_URL + 'test_2.4g.txt'
testing2_4g = np.loadtxt(testing2_4g)

testing5g = BASE_URL + 'test_5g.txt'
testing5g = np.loadtxt(testing5g)

# 坐标数据
cordinaryAll = BASE_URL + 'position.txt'
cordinaryAll = np.loadtxt(cordinaryAll)

cordinaryTest = BASE_URL + 'position_test.txt'
cordinaryTest = np.loadtxt(cordinaryTest)


# 将原始数据做归一化处理     （没有将归一化处理的数据传入到函数当中）
scaler = preprocessing.StandardScaler().fit(training2_4g)
training2_4g = scaler.transform(training2_4g)
testing2_4g = scaler.transform(testing2_4g)

print(np.any(np.isnan(training2_4g)))

print(np.any(np.isnan(testing2_4g)))

# 定义变量函数(权重和偏差)，stdev参数表示方差
def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return (weight)


def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return (bias)

# 创建一个全连接层函数
def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return (tf.nn.relu(layer))

#数据占位符
input1 = tf.placeholder(tf.float32, shape=[None, 12], name="24g")
target = tf.placeholder(tf.float32,shape=[None,2],name="zuobiao")

x = tf.placeholder(tf.float32, shape=[1], name='x')
y = tf.placeholder(tf.float32, shape=[1], name='y')

# --------Create the first layer (12 hidden nodes)--------
w11 = tf.Variable(tf.random_normal([12, 5], stddev=0.03), name="w11")
b11 = tf.Variable(tf.random_normal([5], stddev = 0.03),name="b11")
layer_1 = fully_connected(input1,w11,b11)

#------------- 第2层
w12 = tf.Variable(tf.random_normal([5, 2], stddev=0.1), name="w12")
b12 = tf.Variable(tf.random_normal([1, 2], stddev=0.1), name="b12")
layer_2 = fully_connected(layer_1,w12,b12)

# 定义loss表达式
loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(target - layer_2), reduction_indices=[1])))
train_step = tf.train.AdadeltaOptimizer(0.1 ).minimize(loss)

loss_vec = []
test_loss = []
# 激活会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(30000):
        sess.run(train_step,feed_dict={input1:training2_4g,target:cordinaryAll})
        temp_loss = sess.run(loss,feed_dict={input1:training2_4g,target:cordinaryAll})
        loss_vec.append(temp_loss)

        test_temp_loss = sess.run(loss,feed_dict={input1:testing2_4g, target:cordinaryTest})
        test_loss.append(test_temp_loss)
        if(i+1) % 20 == 0:
            print('Generation' + str(i+1) + '.LOSS = ' + str(test_temp_loss))

plt.plot(loss_vec,"k--",label="train loss")
plt.plot(test_loss,"r--",label="test loss")
plt.title("loss(mse) per generation")
plt.legend(loc="upper right")
plt.xlabel("generation")
plt.ylabel("loss")
plt.show()
