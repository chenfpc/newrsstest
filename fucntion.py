# -*- coding: utf-8 -*-
# 导入相应的包

from scipy.cluster.vq import vq, kmeans, whiten
import scipy
import scipy.cluster.hierarchy as sch
from sklearn.model_selection import train_test_split
import scipy.cluster.hierarchy as sch
# import Matlab as m
import sklearn as sk
import random
import numpy as np
import matplotlib.pylab as plt
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
import scipy
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pylab as plt

BASE_URL = r'C:\Users\chenf\Desktop\\'
def getCategoryForBayes():
    dataFileName = BASE_URL + 'all5.txt'
    originalData = np.loadtxt(dataFileName)

    result = np.zeros((646, 3))  # 待定

    x1 = np.array([1] * 91).reshape((91, 1))
    result[0:91, :] = np.concatenate((originalData[126:217, 2:4], x1), axis=1)

    x2 = np.array([-1] * 91).reshape((91, 1))
    result[91:182] = np.concatenate((originalData[126:217, 10:12], x2), axis=1)

    x3 = np.array([1] * 112).reshape(112, 1)
    result[182:294, :] = np.concatenate((originalData[355:467, 10:12], x3), axis=1)

    x4 = np.array([-1] * 112).reshape(112, 1)
    result[294:406, :] = np.concatenate((originalData[355:467, 2:4], x4), axis=1)

    x5 = np.array([1] * 120).reshape(120, 1)
    result[406:526, :] = np.concatenate((originalData[235:355, 6:8], x5), axis=1)

    x6 = np.array([-1] * 120).reshape(120, 1)
    result[526:646, :] = np.concatenate((originalData[235:355, 2:4], x6), axis=1)
    return result


"""
    testData: 表示测试点的数据（RSS和位置）
    position_test: 只是包含测试点的位置
    dataTag: kmeans分类的类别
    centroid: 每个类别的中心
"""


def getRgression(data2_4, data1_2_4, data2_2_4, distance, data5, data1_5, data2_5):  # 此函数没有被用到
    test1_2_4 = runKnnSimulate(data2_4, distance, data1_2_4[0:1800, :], distance[0:1800, :], 7, 4)
    test2_2_4 = runKnnSimulate(data2_4, distance, data2_2_4[1800:, :], distance[1800:, :], 7, 4)
    x1 = np.array(test1_2_4[0])  # predict 0-1800 2.4
    x2 = np.array(test2_2_4[0])  # predict 1800-3600 2.4
    y1 = np.array(test1_2_4[1])  # real 0-1800 2.4
    y2 = np.array(test2_2_4[1])  # real 1800-3600 2.4

    x_2 = np.concatenate((np.transpose(x1[:, 0]), np.transpose(x2[:, 0])), axis=0)  # predict x for 2.4
    y_2 = np.concatenate((np.transpose(x1[:, 1]), np.transpose(x2[:, 1])), axis=0)  # predict y for 2.4

    test1_5 = runKnnSimulate(data5, distance, data1_5[0:1800, :], distance[0:1800, :], 7, 4)
    test2_5 = runKnnSimulate(data5, distance, data2_5[1800:, :], distance[1800:, :], 7, 4)

    x1_5 = np.array(test1_5[0])
    x2_5 = np.array(test2_5[0])
    y1_5 = np.array(test1_5[1])
    y2_5 = np.array(test2_5[1])

    x_5 = np.concatenate((np.transpose(x1_5[:, 0]), np.transpose(x2_5[:, 0])), axis=0)
    y_5 = np.concatenate((np.transpose(x1_5[:, 1]), np.transpose(x2_5[:, 1])), axis=0)

    a = np.column_stack((x_2, x_5))  # 2.4g 和 5g 的 x (同一位置）
    b = np.column_stack((y_2, y_5))  # 2.4g和5g的y(同一位置）
    real_x = np.concatenate((np.transpose(y1[:, 0]), np.transpose(y2[:, 0])), axis=0)  # 真实位置的x
    real_y = np.concatenate((np.transpose(y1[:, 1]), np.transpose(y2[:, 1])), axis=0)  # 真实位置的y
    return a, b, real_x, real_y


def runGression(data2_4, data1_2_4, data2_2_4, distance, data5, data1_5, data2_5, test_24G, test_5G):  # 此函数没有被用到
    a, b, x, y = getRgression(data2_4, data1_2_4, data2_2_4, distance, data5, data1_5, data2_5)

    reg = sk.linear_model.LinearRegression()
    reg.fit(a, x)
    reg1 = sk.linear_model.LinearRegression()
    reg1.fit(b, y)

    test = runKnnSimulate(data2_4, distance, test_24G, distance, 7, 20)
    test2 = runKnnSimulate(data5, distance, test_5G, distance, 7, 20)

    dataOfTest = np.array(test[0])
    dataOfTest1 = np.array(test2[0])

    realDis = np.array(test[1])

    xOf24g = np.array(dataOfTest[:, 0])
    yOf24g = np.array(dataOfTest[:, 1])

    xOf5g = np.array(dataOfTest1[:, 0])
    yOf5g = np.array(dataOfTest1[:, 1])

    x_x = np.column_stack((xOf24g, xOf5g))
    y_y = np.column_stack((yOf24g, yOf5g))
    x_predict = reg.predict(x_x)
    y_predict = reg1.predict(y_y)

    result = np.column_stack((x_predict, y_predict))

    temp = result - realDis
    temp = temp ** 2
    temp = temp.sum(axis=1)
    temp = temp ** 0.5
    temp = temp.sum(axis=0)
    print("回归之后的误差为：", temp / len(result))


# index 代表是隔几个选一个

def selectTestSet(alldata, alldistance, index):  # 仿真
    testdata = []
    test_distance = []
    lenght = len(alldata)
    for i in range(lenght):
        if (i % index == 0):
            test_distance.append(alldistance[i])
            testdata.append(alldata[i])
    return testdata, test_distance


def runKnnSimulate(alldata, alldistance, alldata1, distance1, k, index):
    # 测试集划分
    # traindata, testdata, train_distance, test_distance = train_test_split(alldata1, distance1, train_size=0.9)
    testdata, test_distance = selectTestSet(alldata1, distance1, index)
    # training set 就是包含3600个点所有数据。注意，training set 的distance 和 testing set 的distance是同一个文件
    trainingSet_cordinary = np.column_stack((alldata, alldistance))  # 这里直接随机选区五分之一的数据点作为测试点，然后取knn = 7进行仿真。
    testingSet_cordinary = np.column_stack((testdata, test_distance))
    cordinaryTestSet = test_distance

    result = 0

    predict_cordinary = [None] * len(testdata)
    for i in range(len(testdata)):
        knnResult = calculateCordinary(k, trainingSet_cordinary, testingSet_cordinary[i], i, cordinaryTestSet)
        result += knnResult[0]
        predict_cordinary[i] = knnResult[1]
    print("平均误差为")
    print(result / len(testdata))
    return predict_cordinary, test_distance


def runKnnRealityKNN(trainingSet, testingSet, cordinaryAllSet, cordinaryTestSet, k, ifweight):
    trainingSet_cordinary = np.column_stack((trainingSet, cordinaryAllSet))
    testingSet_cordinary = np.column_stack((testingSet, cordinaryTestSet))
    result = 0
    predict_cordinary = [None] * len(testingSet)
    for i in range(len(testingSet)):
        knnResult = calculateCordinary(k, trainingSet_cordinary, testingSet_cordinary[i], i, cordinaryTestSet, ifweight)
        result += knnResult[0]
        predict_cordinary[i] = knnResult[1]
    print("平均误差为")
    print(result / len(testingSet))
    return predict_cordinary


def runClassfication_Simulate(trainingSet, cordinarySet):
    trainingSet = np.reshape(trainingSet, (60, 60, 12))
    cordinarySet = np.reshape(cordinarySet, (60, 60, 2))
    trainingSet1 = np.zeros((900, 12))
    cordinarySet1 = np.zeros((900, 2))
    trainingSet2 = np.zeros((900, 12))
    cordinarySet2 = np.zeros((900, 2))
    cordinarySet3 = np.zeros((900, 2))
    trainingSet3 = np.zeros((900, 12))
    trainingSet4 = np.zeros((900, 12))
    cordinarySet4 = np.zeros((900, 2))

    index = 0
    for i in range(30):
        for j in range(30):
            trainingSet1[index] = trainingSet[i, j, :]
            cordinarySet1[index] = cordinarySet[i, j, :]
            trainingSet2[index] = trainingSet[i, j + 30, :]
            cordinarySet2[index] = cordinarySet[i, j + 30, :]
            index = index + 1
    index = 0
    for i in range(30):
        for j in range(30):
            trainingSet4[index] = trainingSet[i + 30, j, :]
            cordinarySet4[index] = cordinarySet[i + 30, j, :]
            trainingSet3[index] = trainingSet[i + 30, j + 30, :]
            cordinarySet3[index] = cordinarySet[i + 30, j + 30, :]
            index = index + 1

    print(len(trainingSet1), len(cordinarySet1))

    datax1 = np.concatenate((trainingSet1, cordinarySet1), axis=1)
    datax2 = np.concatenate((trainingSet2, cordinarySet2), axis=1)
    datax3 = np.concatenate((trainingSet3, cordinarySet3), axis=1)
    datax4 = np.concatenate((trainingSet4, cordinarySet4), axis=1)
    centroid1 = centroid_of_area(trainingSet1)
    centroid2 = centroid_of_area(trainingSet2)
    centroid3 = centroid_of_area(trainingSet3)
    centroid4 = centroid_of_area(trainingSet4)
    return (datax1, centroid1), (datax2, centroid2), (datax3, centroid3), (datax4, centroid4)


def runClassfication(trainingSet, cordinarySet):
    """
    :param trainingSet: 源数据
    :param cordinarySet:
    :return: 聚类集合和相应中心
    """
    trainingSet1 = np.row_stack((trainingSet[0:63, :], trainingSet[126:217, :]))
    cordinarySet1 = np.row_stack((cordinarySet[0:63, :], cordinarySet[126:217, :]))
    trainingSet2 = trainingSet[63:126, :]
    cordinarySet2 = cordinarySet[63:126, :]
    trainingSet3 = trainingSet[217:298]
    cordinarySet3 = cordinarySet[217:298]
    trainingSet4 = trainingSet[298:355]
    cordinarySet4 = cordinarySet[298:355]
    trainingSet5 = trainingSet[355:467]
    cordinarySet5 = cordinarySet[355:467]

    datax1 = np.concatenate((trainingSet1, cordinarySet1), axis=1)
    datax2 = np.concatenate((trainingSet2, cordinarySet2), axis=1)
    datax3 = np.concatenate((trainingSet3, cordinarySet3), axis=1)
    datax4 = np.concatenate((trainingSet4, cordinarySet4), axis=1)
    datax5 = np.concatenate((trainingSet5, cordinarySet5), axis=1)
    centroid1 = centroid_of_area(trainingSet1)
    centroid2 = centroid_of_area(trainingSet2)
    centroid3 = centroid_of_area(trainingSet3)
    centroid4 = centroid_of_area(trainingSet4)
    centroid5 = centroid_of_area(trainingSet5)
    return (datax1, centroid1), (datax2, centroid2), (datax3, centroid3), (datax4, centroid4), (datax5, centroid5)


def centroid_of_area(featureSet):
    """
    返回子区域的中心
    :param featureSet: 子区域集合
    :return: 中心
    """
    length = len(featureSet)
    centroid = featureSet.sum(axis=0)
    centroid = centroid / length
    return centroid


def getClusters(classfication, number):
    """
    对每个子区域执行kmeans方法
    :param classfication: 各个子区域的值和中心点
    :return: 返回每个子区域的聚类结果和中心点
    """
    clusterResults = []
    for i in range(len(classfication)):
        clusterResults.append(runCluster(classfication[i][0], number))
    return clusterResults


def runCluster(dataSet, numbers):
    cluster = numbers
    centroid = kmeans(dataSet[:, :-2], cluster)[0]
    dataTag = [list() for i in range(cluster)]
    # 使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label
    # label = vq(dataSet[:, :-2], centroid)[0]
    label = []
    for i in range(len(dataSet[:, :-2])):
        dataTag[label[i]].append(dataSet[i])
    return dataTag, centroid


def runRealityClusterKnn(trainingSet, testingSet, originalTestingSet, cordinaryAllSet, cordinaryTestSet, classfication,
                         clusters, ifweight, clf,bayes):
    """
    :param trainingSet: 归一化后的训练集
    :param testingSet:  归一化后的测试机
    :param originalTestingSet: 没有归一化的测试机
    :param cordinaryAllSet: 训练集坐标
    :param cordinaryTestSet: 测试机坐标
    :param classfication: 聚类结果（分为5类，包含数据和中心）
    :param ifweight: 使用wknn还是knn
    :param clf : 训练好的svmc模型
    :return:
    """
    # 选择给RSS加权重需要改变 testingSet
    trainingByesData = []
    testingSet_cordinary = np.column_stack((testingSet, cordinaryTestSet))  # 将RSS值和位置信息合并在一起
    result = clusterKNN(testingSet_cordinary, trainingByesData,originalTestingSet, cordinaryTestSet, classfication, clusters, ifweight,
                          clf,bayes)
    print("平均误差为")
    print(result[0])
    return result[1]


# def runSimulateClusterKnn(trainingSet, testingSet, originalTestingSet, cordinaryAllSet, cordinaryTestSet, classfication,
#                           clusters, ifweight, clf):
#     testingSet_cordinary = np.column_stack((testingSet, cordinaryTestSet))
#     result = h_clusterKNN(testingSet_cordinary, originalTestingSet, cordinaryTestSet, classfication, clusters, ifweight,
#                           clf)
#     print("平均误差为")
#     print(result[0])
#     return result[1]


def judgeSimulate(testPoint):
    result = ""
    wifi1 = testPoint[0]
    wifi2 = testPoint[4]
    wifi3 = testPoint[8]

    if -19 >= wifi1 >= -57.65:
        result = '0'
    elif -23 >= wifi2 >= -61.65:
        result = '1'
    elif -17 >= wifi3 >= -55.65:
        result = '2'
    else:
        result = '3'
    return result


def judge(testPoint):
    result = ""
    wifi1 = testPoint[0]  # 2.4G判断哪个子区域
    wifi2 = testPoint[4]
    wifi3 = testPoint[8]

    if -61 <= wifi1 <= -22:
        if wifi3 <= -60:
            result = "AC"
        else:
            result = "D"
    elif wifi1 < -61:
        if wifi3 <= -57:
            result = "B"
        elif wifi2 <= -56:
            result = "F"
        else:
            result = "E"
    return result


def judgeCluster_Simulate(testPoint, classfication):  # 这个函数干嘛用的，和下面的好像重复了呢？？？？？#
    clusterResult = judge(testPoint)
    data = np.array([])
    index = -1
    if clusterResult == "0":
        data = classfication[0]
        index = 0
    if clusterResult == "1":
        data = classfication[1]
        index = 1
    if clusterResult == "2":
        data = classfication[2]
        index = 2
    if clusterResult == "3":
        data = classfication[3]
        index = 3
    return data, index


def judgeCluster(testPoint, classfication):
    """
    判断当前点属于哪个子区域
    :param testPoint: 测试点，未处理的原始数据
    :param source1:
    :param classfication:
    :return:
    """
    clusterResult = judge(testPoint)
    data = np.array([])
    index = -1
    if clusterResult == "AC":
        data = classfication[0]
        index = 0
    if clusterResult == "B":
        data = classfication[1]
        index = 1
    if clusterResult == "D":
        data = classfication[2]
        index = 2
    if clusterResult == "E":
        data = classfication[3]
        index = 3
    if clusterResult == "F":
        data = classfication[4]
        index = 4
    return data, index


def calculateCordinary(k, data, point, index, positions_test, ifweight, originalPoint):
    """
    :param  K:选取几个k近邻点
    :param  data: 所处聚类的数据
    :param  point: 测试点
    :param index: 第几个测试点
    :param position_test: 这些测试点的实际坐标
    :param  该函数目的是为了求出给定点point在data聚类的k个近邻点，返回的是定位的误差
    :return 返回和实际位置的误差，以及预测坐标
    """
    result = knn(point, data, k, ifweight, originalPoint)



    predic_position = result
    result = (result - positions_test[index]) ** 2
    error = result.sum(axis=0) ** 0.5  # 为什么#
    # print(error)
    count = 0
    # if (error > 2):  # 输出误差大于2的预测坐标和真实坐标
    #     print("预测坐标", predic_position)
    #     print("真实坐标", positions_test[index])
    #     print(error)
    return error, predic_position


def knn(testPoint, dataset, k, ifWeight, originalPoint):
    feature_set = np.array(dataset)
    feature_set = feature_set[:, :-2]
    datasetSize = len(feature_set)
    x = testPoint[:][:-2]
    diffMat = np.tile(x, (datasetSize, 1)) - feature_set
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistIndicies = distance.argsort()
    sumx = 0
    sumy = 0
    length = len(dataset[0])
    # ifWeight=1代表wknn，否则为knn
    if ifWeight == 1:
        k_sum = 0  # k_sum为所有权重之和
        weight = []
        for i in range(k):

            temp = x - dataset[sortedDistIndicies[i]][:-2]
            temp = temp ** 2
            temp = temp.sum(axis=0)
            temp = temp ** 0.5
            if temp == 0:
                temp = 0.000001
            temp = 1 / temp
            k_sum = k_sum + temp
            weight.append(temp)
        if k > datasetSize:
            for i in range(datasetSize):
                sumx = sumx + (weight[i] / k_sum) * dataset[sortedDistIndicies[i]][length - 2]
                sumy = sumy + (weight[i] / k_sum) * dataset[sortedDistIndicies[i]][length - 1]
                return sumx, sumy
        else:
            for i in range(k):
                x = dataset[sortedDistIndicies[i]][length - 2]
                y = dataset[sortedDistIndicies[i]][length - 1]
                print(x, y)
                sumx = sumx + (weight[i] / k_sum) * x
                sumy = sumy + (weight[i] / k_sum) * y
            print("jieshu")
            return sumx, sumy
    else:
        if k > datasetSize:
            for i in range(datasetSize):
                sumx = sumx + dataset[sortedDistIndicies[i]][length - 2]
                sumy = sumy + dataset[sortedDistIndicies[i]][length - 1]
                # print((sumx,sumy))
                return sumx / datasetSize, sumy / datasetSize
        else:
            print("output begin")
            for i in range(k):
                x = dataset[sortedDistIndicies[i]][length - 2]
                y = dataset[sortedDistIndicies[i]][length - 1]
                sumx = sumx + x
                sumy = sumy + y
                print((x, y))
            print("ouput stop")
            return sumx / k, sumy / k


def clusterKNN(testData,trainingByesData, originalTestSet, positions_test, classfication, clusters, ifweight, clf,bayes):
    """
    聚类KNN或者wknn
    :param testData: 训练集，包含数据和坐标（处理好的数据）
    :param originalTestSet: 未处理的测试集
    :param positions_test:  测试坐标
    :param classfication: 聚类结果
    :param clusters: 每个子区域的kmean后的结果
    :param ifweight: 是否wknn
    :param clf: smvc训练出的模型
    :return:
    """
    cdf = []
    index = 0
    error = 0
    len1 = len(positions_test)
    # 新建一个predict_cordinary保存预测的坐标值
    predict_cordinary = [None] * len1
    # fileObject = open(r"C:\Users\computer\Desktop\cdf_best.txt", "w")
    for i in range(len1):
        print(i)
        which_class = judgeCluster(originalTestSet[i][:], classfication)  # 这里判断出子区域是哪一个，是A、B、C、D等，每一行都是（2.4g,5g)这样的特征值。
        class_data = (which_class[0])[0]  # 对应子区域中的所有数据




        dataSet = class_data[:,:]  # 将4*6的长度转为3，对不同的nlos状态找不同的2.4还是5g信号

        testPoint = testData[i]  # 测试点

        numbers = 7
        cluster = numbers
        centroids = kmeans(dataSet[:, :-2], cluster)[0]
        dataTag = [list() for i in range(cluster)]
        # 使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label
        label = vq(dataSet[:, :-2], centroids)[0]
        for j in range(len(dataSet[:, :-2])):
            dataTag[label[j]].append(dataSet[j])
        # numbers = 5;
        #
        # label,centroids = kmeans(dataSet[:,:-2],numbers)
        # dataTag = [list() for i in range(numbers)]
        # for j in range(len(dataSet[:,:-2])):
        #     dataTag[label[j]].append(dataSet[j])
        """
        #获取cluster
        clusterData = clusters[label[1]]
        centroids = clusterData[1]
        datas = clusterData[0]
        """
        index = 0
        temp = 100000
        for j in range(len(centroids)):
            sub = testPoint[:-2] - centroids[j]
            differ = sub ** 2
            # print("DIFFER",differ)
            result = differ.sum(axis=0)
            if result < temp:
                temp = result
                index = j

        data = dataTag[index]

        # 这里直接把data = label[0]
        # data = (label[0])[0] #labell=[0]直接返回的是子区域中的所有点，而clusters代表的是这个子区域中的所有聚类。
        result = calculateCordinary(4, data, testPoint, i, positions_test, ifweight, originalTestSet[i][:])
        # print(result[0])
        x = result[0]
        cdf.append(x)
        # if result[0] > 4.5:
        #     x = random.randint(1,10)*0.5
        # fileObject.write(str(x))
        # fileObject.write("\n")
        error = error + result[0]
        predict_cordinary[i] = result[1]
        # fileObject.close()
    return error / len(positions_test), cdf


# def clusterKNN(testData, originalTestSet, positions_test, classfication,clusters,ifweight,clf):
#     """
#     聚类KNN或者wknn
#     :param testData: 训练集，包含数据和坐标（处理好的数据）
#     :param originalTestSet: 未处理的测试集
#     :param positions_test:  测试坐标
#     :param classfication: 聚类结果
#     :param clusters: 每个子区域的kmean后的结果
#     :param ifweight: 是否wknn
#     :param clf: smvc训练出的模型
#     :return:
#     """
#     index = 0
#     error = 0
#     len1 = len(positions_test)
#     # 新建一个predict_cordinary保存预测的坐标值
#     predict_cordinary = [None] * len1
#     #fileObject = open(r"C:\Users\computer\Desktop\cdf_best.txt", "w")
#     for i in range(len1):
#         which_class = judgeCluster(originalTestSet[i][:], classfication)   #这里判断出子区域是哪一个，是A、B、C、D等，每一行都是（2.4g,5g)这样的特征值。
#         class_data = (which_class[0])[0] #对应子区域中的所有数据
#
#         #判断该点处于何种环境
#         flag_wifi1 = clf.predict(originalTestSet[i][12:16].reshape(1,-1)) #使用5G，返回只有1或者-1 1代表los,
#         flag_wifi2 = clf.predict(originalTestSet[i][16:20].reshape(1,-1))
#         flag_wifi3 = clf.predict(originalTestSet[i][20:24].reshape(1,-1))
#
#         #根据NLOS的状态取不同的值
#         if flag_wifi1 == 1:  #1对应los
#             w1 = 12
#         else:
#             w1 = 0
#         if flag_wifi2 == 1:  #1对应los
#             w2 = 16
#         else:
#             w2 = 4
#         if flag_wifi3 == 1:  # 1对应los
#             w3 = 20
#         else:
#             w3 = 8
#
#         metric = [w1,w2,w3,24,25]
#         dataSet = class_data[:,metric]  #将4*6的长度转为3，对不同的nlos状态找不同的2.4还是5g信号
#         metric_testpoint = [w1,w2,w3,24,25] #取
#         testPoint = testData[i,metric_testpoint] #测试点
#
#         numbers = 5
#         cluster = numbers
#         centroids = kmeans(dataSet[:,:-2], cluster)[0]
#         dataTag = [list() for i in range(cluster)]
#         # 使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label
#         label = vq(dataSet[:,:-2], centroids)[0]
#         for j in range(len(dataSet[:, :-2])):
#             dataTag[label[j]].append(dataSet[j])
#
#         """
#         #获取cluster
#         clusterData = clusters[label[1]]
#         centroids = clusterData[1]
#         datas = clusterData[0]
#         """
#         index = 0
#         temp = 100000
#         for j in range(len(centroids)):
#             sub = testPoint[:-2] - centroids[j]
#             differ = sub ** 2
#             # print("DIFFER",differ)
#             result = differ.sum(axis=0)
#             if result < temp:
#                 temp = result
#                 index = j
#
#         data = dataTag[index]
#
#         #这里直接把data = label[0]
#         #data = (label[0])[0] #labell=[0]直接返回的是子区域中的所有点，而clusters代表的是这个子区域中的所有聚类。
#         result = calculateCordinary(3, data, testPoint, i, positions_test,ifweight, originalTestSet[i][:])
#         #print(result[0])
#         # x = result[0]
#         # if result[0] > 4.5:
#         #     x = random.randint(1,10)*0.5
#         # fileObject.write(str(x))
#         # fileObject.write("\n")
#         error = error + result[0]
#         predict_cordinary[i] = result[1]
#     #fileObject.close()
#     return error / len(positions_test), predict_cordinary
# def h_clusterKNN(testData, originalTestSet, positions_test, classfication, clusters, ifweight,
#                  clf):  # 由师兄的clusterKNN函数改编
#     """
#     聚类KNN或者wknn
#     :param testData: 训练集，包含数据和坐标（处理好的数据）
#     :param originalTestSet: 未处理的测试集
#     :param positions_test:  测试坐标
#     :param classfication: 聚类结果
#     :param clusters: 每个子区域的kmean后的结果
#     :param ifweight: 是否wknn
#     :param clf: smvc训练出的模型
#     :return:
#     """
#     # 选择给RSS加权重需要改变 testData
#     index = 0
#     error = 0
#     len1 = len(positions_test)
#     # 新建一个predict_cordinary保存预测的坐标值
#     predict_cordinary = [None] * len1
#     # fileObject = open(r"C:\Users\computer\Desktop\cdf_best.txt", "w")
#     for i in range(len1):  # 一个点一个点的去
#         # 判断
#         which_class = judgeCluster(originalTestSet[i][:], classfication)  # 这里判断出子区域是哪一个，是A、B、C、D等，每一行都是（2.4g,5g)这样的特征值。
#         class_data = (which_class[0])[0]  # 对应子区域中的所有数据    ，判断出某一个测试点在哪个子区域中后，选出该点对应子区域的所有数据  ，用于以后与该测试点的匹配
#
#         clf.predict()
#     # 判断该点处于何种环境           取用各个wifi的5G信号，隔6行是一个点位
#     weight_wifi1 = r[(6 * i) + 1]  # 使用5G信号来区分NOLS还是LOS
#     weight_wifi2 = r[(6 * i) + 3]
#     weight_wifi3 = r[(6 * i) + 5]
#
#     # 根据NLOS的状态取不同的值
#     # 判断wifi1是否处于NLOS状态
#     if weight_wifi1 <= 0.2:  # 小于等于0.2对应los，用5G，0.2是自己通过观察权重数据给出的阈值，经过多次试验设置为0.2效果最好
#         w1 = 12
#         w11 = 0
#     elif weight_wifi1 > 1:  # 大于1代表NLOS信号非常多，完全用2.4G信号  不用5G信号
#         w1 = 0
#         w11 = 12
#     else:
#         w1 = 0
#         w11 = 12
#
#     # 判断wifi2是否处于NLOS状态
#     if weight_wifi2 <= 0.2:  # 小于等于0.2对应los
#         w2 = 16
#         w22 = 4
#     elif weight_wifi2 > 1:  # 大于1代表NLOS信号非常多，完全用2.4G信号  不用5G信号（之后的权重处体现出来）
#         w2 = 4
#         w22 = 16
#     else:
#         w2 = 4
#         w22 = 16
#
#     # 判断wifi3是否处于NLOS状态
#     if weight_wifi3 <= 0.2:  # 小于等于0.2对应los
#         w3 = 20
#         w33 = 8
#     elif weight_wifi3 > 1:  # 大于1代表NLOS信号非常多，完全用2.4G信号  不用5G信号
#         w3 = 8
#         w33 = 20
#     else:
#         w3 = 8
#         w33 = 20
#
#     '''
#
#     开始计算第一个位置
#
#     '''
#     # 以下是用于匹配的数据库中的数据
#     metric = [w1, w2, w3, 24, 25]  # w是均值的角标，用均值来判断该测试点在哪个子区域
#     dataSet = class_data[:, metric]  # 将4*6的长度转为3，对不同的nlos状态找不同的2.4还是5g信号      ，class_data为选出的子区域中的所有数据
#     # 将子区域中的数据筛选出来以便于进行对比匹配，筛选出例如2.4G、5G、2.4G这样的
#     # 以下是测试点数据
#     metric_testpoint = [w1, w2, w3, 24, 25]  # 取
#     testPoint = testData[i, metric_testpoint]  # 测试点
#
#     numbers = 5
#     cluster = numbers
#     centroids = kmeans(dataSet[:, :-2], cluster)[0]  # 将选出的子区域中的数据拿到kmeans算法中进行该区域的聚类运算
#     dataTag = [list() for i in range(cluster)]
#     # 使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label
#     label = vq(dataSet[:, :-2], centroids)[0]
#     for j in range(len(dataSet[:, :-2])):
#         dataTag[label[j]].append(dataSet[j])
#
#     """
#     #获取cluster
#     clusterData = clusters[label[1]]
#     centroids = clusterData[1]
#     datas = clusterData[0]
#     """
#     index = 0
#     temp = 100000
#     for j in range(len(centroids)):
#         sub = testPoint[:-2] - centroids[j]  # 计算测试点与聚类中心的距离
#         differ = sub ** 2
#         # print("DIFFER",differ)
#         result = differ.sum(axis=0)
#         if result < temp:
#             temp = result  # 选出离测试点最近的聚类中心的位置
#             index = j
#
#     data = dataTag[index]  # 将选出来的聚类中心所在的类的数据赋给data
#
#     # 这里直接把data = label[0]
#     # data = (label[0])[0] #labell=[0]直接返回的是子区域中的所有点，而clusters代表的是这个子区域中的所有聚类。
#
#     # result = calculateCordinary(3, data, testPoint, i, positions_test,ifweight, originalTestSet[i][:])       此行是被我注释掉的  ，其他行都是师兄自己注释掉的
#
#     # print(result[0])
#     # x = result[0]
#     # if result[0] > 4.5:
#     #     x = random.randint(1,10)*0.5
#     # fileObject.write(str(x))
#     # fileObject.write("\n")
#     # error = error + result[0]                                              此行是被我注释掉的  ，其他行都是师兄自己注释掉的
#     # predict_cordinary[i] = result[1]                                       此行是被我注释掉的  ，其他行都是师兄自己注释掉的
#     # fileObject.close()
#
#     '''
#
#         开始计算第二个位置
#
#     '''
#     # 以下是用于匹配的数据库中的数据
#     metric2 = [w11, w22, w33, 24, 25]
#     dataSet2 = class_data[:, metric2]  # 将4*6的长度转为3，对不同的nlos状态找不同的2.4还是5g信号      ，class_data为选出的子区域中的所有数据
#     # 将子区域中的数据筛选出来以便于进行对比匹配，筛选出例如2.4G、5G、2.4G这样的
#     # 以下是测试点数据
#     metric_testpoint2 = [w11, w22, w33, 24, 25]  # 取
#     testPoint2 = testData[i, metric_testpoint2]  # 测试点
#
#     numbers2 = 5
#     cluster2 = numbers2
#     centroids2 = kmeans(dataSet2[:, :-2], cluster2)[0]  # 将选出的子区域中的数据拿到kmeans算法中进行该区域的聚类运算
#     dataTag2 = [list() for i in range(cluster2)]
#     # 使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label
#     label_2 = vq(dataSet2[:, :-2], centroids2)[0]
#     for j in range(len(dataSet2[:, :-2])):
#         dataTag2[label_2[j]].append(dataSet2[j])
#
#     """
#     #获取cluster
#     clusterData = clusters[label[1]]
#     centroids = clusterData[1]
#     datas = clusterData[0]
#     """
#     index = 0
#     temp = 100000
#     for j in range(len(centroids2)):
#         sub = testPoint2[:-2] - centroids2[j]  # 计算测试点与聚类中心的距离
#         differ = sub ** 2
#         # print("DIFFER",differ)
#         result = differ.sum(axis=0)
#         if result < temp:
#             temp = result  # 选出离测试点最近的聚类中心的位置
#             index = j
#
#     data2 = dataTag2[index]  # 将选出来的聚类中心所在的类的数据赋给data
#
#     # 这里直接把data = label[0]
#     # data = (label[0])[0] #labell=[0]直接返回的是子区域中的所有点，而clusters代表的是这个子区域中的所有聚类。
#     result = calculateCordinary(5, data, testPoint, i, positions_test, ifweight, originalTestSet[i][:], data2,
#                                 testPoint2, r, metric)  # 调用此方法 然后再放到KNN算法中，计算预测点和真实位置的误差
#     # 传入calculateCordinary函数的第一个参数如果设置为4，则跑出来的结果非常稳定，在2.05左右。如果设置为5，则有最好结果1.93米
#
#     # print(result[0])
#     # x = result[0]
#     # if result[0] > 4.5:
#     #     x = random.randint(1,10)*0.5
#     # fileObject.write(str(x))
#     # fileObject.write("\n")
#     error = error + result[0]  # 将误差累加
#     predict_cordinary[i] = result[1]  # predict_cordinary是数组，将每一次循环的计算结果全部存储下来
#     return error / len(positions_test), predict_cordinary
