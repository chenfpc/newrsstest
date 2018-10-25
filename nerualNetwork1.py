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

# # 将原始数据做归一化处理     （没有将归一化处理的数据传入到函数当中）
# scaler = preprocessing.StandardScaler().fit(training5g)
# training5g = scaler.transform(training5g)
# testing5g = scaler.transform(testing5g)

ACTIVATION = tf.nn.relu
N_LAYERS = 1
N_HIDDEN_UNITS = 5


def build_net(xs, ys, norm):
    # 创建一个全连接层函数
    def add_layer(inputs, in_size, out_size, activation_function=None, norm=False):
        Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0, stddev=1))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

        Wx_plus_b = tf.matmul(inputs, Weights) + biases

        if norm:
            fc_mean, fc_var = tf.nn.moments(
                Wx_plus_b,
                axes=[0],
            )
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.001

            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            mean, var = mean_var_with_update()

            Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    if norm:
        # BN for the first input
        fc_mean, fc_var = tf.nn.moments(
            xs,
            axes=[0],
        )
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        epsilon = 0.001
        # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)

        mean, var = mean_var_with_update()
        xs = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)

    layers_inputs = [xs]

    for l_n in range(N_LAYERS):
        layer_input = layers_inputs[l_n]
        in_size = layers_inputs[l_n].get_shape()[1].value

        output = add_layer(
            layer_input,  # input
            in_size,  # input size
            N_HIDDEN_UNITS,  # output size
            ACTIVATION,  # activation function
            norm,  # normalize before activation
        )
        layers_inputs.append(output)  # add output for next run

    # build output layer
    prediction = add_layer(layers_inputs[-1], 5, 2, ACTIVATION)
    cost = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1])))
    train_op = tf.train.AdadeltaOptimizer(0.01).minimize(cost)
    return [train_op, cost, layers_inputs]


# 数据占位符
xs = tf.placeholder(tf.float32, shape=[None, 12], name="24g")
ys = tf.placeholder(tf.float32, shape=[None, 2], name="zuobiao")

train_op, cost, layers_inputs = build_net(xs, ys, norm=True)

train_losses = []
test_losses = []
# 激活会话
with tf.Session() as sess:
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(100000):
        sess.run(train_op, feed_dict={xs: training2_4g, ys: cordinaryAll})
        train_loss = sess.run(cost, feed_dict={xs: training2_4g, ys: cordinaryAll})
        train_losses.append(train_loss)

        test_loss = sess.run(cost, feed_dict={xs: testing2_4g, ys: cordinaryTest})
        test_losses.append(test_loss)
        if (i + 1) % 20 == 0:
            print('Generation' + str(i + 1) + '.LOSS = ' + str(test_loss))
# y_ticks = np.arange(0,25,0.5)
plt.plot(train_losses, "k--", label="train loss")
plt.plot(test_losses, "r--", label="test loss")
plt.title("loss(mse) per generation")
# plt.yticks(y_ticks)
plt.legend(loc="upper right")
plt.xlabel("generation")
plt.ylabel("loss")
plt.show()
