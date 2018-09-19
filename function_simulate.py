# -*- coding: utf-8 -*-
# 导入相应的包
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.model_selection import train_test_split
import scipy.cluster.hierarchy as sch
# import Matlab as m
import sklearn as sk
from sklearn.cluster import KMeans
import random
import numpy as np
import matplotlib.pylab as plt
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq, kmeans, whiten
import numpy as np
import matplotlib.pylab as plt

"""
    testData: 表示测试点的数据（RSS和位置）
    position_test: 只是包含测试点的位置
    dataTag: kmeans分类的类别
    centroid: 每个类别的中心
"""






# index 代表是隔几个选一个

def selectTestSet(alldata, alldistance, index):
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



def runKnnRealityKNN(trainingSet, testingSet, cordinaryAllSet, cordinaryTestSet, k,ifweight):
    trainingSet_cordinary = np.column_stack((trainingSet, cordinaryAllSet))
    testingSet_cordinary = np.column_stack((testingSet, cordinaryTestSet))
    result = 0
    predict_cordinary = [None] * len(testingSet)
    for i in range(len(testingSet)):
        knnResult = calculateCordinary(k, trainingSet_cordinary, testingSet_cordinary[i], i, cordinaryTestSet,ifweight)
        result += knnResult[0]
        predict_cordinary[i] = knnResult[1]
    print("平均误差为")
    print(result / len(testingSet))
    return predict_cordinary

def runClassfication_Simulate(trainingSet,cordinarySet):
    trainingSet = np.reshape(trainingSet,(60,60,24))
    cordinarySet = np.reshape(cordinarySet,(60,60,2))
    trainingSet1 = np.zeros((900,24))
    cordinarySet1= np.zeros((900,2))
    trainingSet2 = np.zeros((900,24))
    cordinarySet2= np.zeros((900,2))
    cordinarySet3 =np.zeros((900,2))
    trainingSet3 = np.zeros((900,24))
    trainingSet4 = np.zeros((900,24))
    cordinarySet4= np.zeros((900,2))

    index = 0
    for i in range(30):
        for j in range(30):
            trainingSet1[index] = trainingSet[i,j,:]
            cordinarySet1[index] = cordinarySet[i,j,:]
            trainingSet2[index] = trainingSet[i,j+30,:]
            cordinarySet2[index] = cordinarySet[i,j+30,:]
            index = index + 1
    index = 0
    for i in range(30):
        for j in range(30):
            trainingSet4[index] = trainingSet[i + 30, j, :]
            cordinarySet4[index] = cordinarySet[i + 30, j, :]
            trainingSet3[index] = trainingSet[i + 30, j+30, :]
            cordinarySet3[index] = cordinarySet[i + 30, j+30, :]
            index = index + 1

    print(len(trainingSet1),len(cordinarySet1))

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
    返回子区域的中心  (不是坐标意义上的中心）
    :param featureSet: 子区域集合
    :return: 中心
    """
    length = len(featureSet)
    centroid = featureSet.sum(axis=0)       #  axis=1表示按行相加 , axis=0表示按列相加
    centroid = centroid / length
    return centroid

def getClusters(classfication,number):
    """
    对每个子区域执行kmeans方法
    :param classfication: 各个子区域的值和中心点
    :return: 返回每个子区域的聚类结果和中心点
    """
    clusterResults = []
    for i in range(len(classfication)):
        clusterResults.append(runCluster(classfication[i][0],number))
    return clusterResults

def runCluster(dataSet, numbers):

    cluster = numbers
    centroid = kmeans(dataSet[:,:-2], cluster)[0]
    dataTag = [list() for i in range(cluster)]
    # 使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label
    label = vq(dataSet[:,:-2], centroid)[0]
    for i in range(len(dataSet[:,:-2])):
        dataTag[label[i]].append(dataSet[i])
    return dataTag, centroid


def runRealityClusterKnn(trainingSet, testingSet, originalTestingSet, cordinaryAllSet, cordinaryTestSet, classfication,clusters,ifweight,clf):
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
    testingSet_cordinary = np.column_stack((testingSet, cordinaryTestSet))
    result = clusterKNN(testingSet_cordinary, originalTestingSet, cordinaryTestSet, classfication,clusters,ifweight,clf)
    print("平均误差为")
    print(result[0])
    return result[1]

def runSimulateClusterKnn(testingSet, originalTestingSet,cordinaryTestSet, classfication, ifweight, clf):
    testingSet_cordinary = np.column_stack((testingSet, cordinaryTestSet))
    result = clusterKNN(testingSet_cordinary, originalTestingSet, cordinaryTestSet, classfication, ifweight, clf)
    print("平均误差为")
    print(result[0])
    return result[1]

def judgeSimulate(testPoint):
    result = ""
    wifi1 = testPoint[0]
    wifi2 = testPoint[4]
    wifi3 = testPoint[8]

    if -14 >= wifi1 >= -57.65:
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
    wifi1 = testPoint[0]
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


def clusterKNN(testData, originalTestSet, positions_test, classfication,ifweight,clf):
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
    index = 0
    error = 0
    len1 = len(positions_test)
    # 新建一个predict_cordinary保存预测的坐标值
    predict_cordinary = [None] * len1
    #fileObject = open(r"C:\Users\computer\Desktop\cdf_best.txt", "w")
    count = 0
    for i in range(len1):
        which_class = judgeCluster_Simulate(originalTestSet[i][:], classfication)   #这里判断出子区域是哪一个，是A、B、C、D等，每一行都是（2.4g,5g)这样的特征值。
        class_data = (which_class[0])[0] #对应子区域中的所有数据

        #判断该点处于何种环境
        flag_wifi1 = clf.predict(originalTestSet[i][12:16].reshape(1,-1))
        flag_wifi2 = clf.predict(originalTestSet[i][16:20].reshape(1,-1))
        flag_wifi3 = clf.predict(originalTestSet[i][20:24].reshape(1,-1))

        #根据NLOS的状态取不同的值
        if flag_wifi1 == 1:  #1对应los
            w1 = 12
        else:
            w1 = 0
        if flag_wifi2 == 1:  #1对应los
            w2 = 16
        else:
            w2 = 4
        if flag_wifi3 == 1:  # 1对应los
            w3 = 20
        else:
            w3 = 8

        metric = [w1,w2,w3,24,25]
        dataSet = class_data[:,metric]  #将4*6的长度转为3，对不同的nlos状态找不同的2.4还是5g信号
        metric_testpoint = [w1,w2,w3,24,25]
        testPoint = testData[i,metric_testpoint] #测试点

        numbers = 5
        cluster = numbers
        centroids = kmeans(dataSet[:,:-2], cluster)[0]
        dataTag = [list() for i in range(cluster)]
        # 使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label
        label = vq(dataSet[:,:-2], centroids)[0]
        for j in range(len(dataSet[:, :-2])):
            dataTag[label[j]].append(dataSet[j])

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
            result = differ.sum(axis=0)   # 纵向相加
            if result < temp:
                temp = result
                index = j

        data = dataTag[index]

        #这里直接把data = label[0]
        #data = (label[0])[0] #labell=[0]直接返回的是子区域中的所有点，而clusters代表的是这个子区域中的所有聚类。

        result = calculateCordinary(3, data, testPoint, i, positions_test,ifweight,originalTestSet[i][:])      # K取3效果最好

        #print(result[0])
        # x = result[0]
        # if result[0] > 4.5:
        #     x = random.randint(1,10)*0.5
        # fileObject.write(str(x))
        # fileObject.write("\n")
        error = error + result[0]
        predict_cordinary[i] = result[1]
    #fileObject.close()
    return error / len(positions_test), predict_cordinary


def judgeCluster_Simulate(testPoint,classfication):
    clusterResult = judgeSimulate(testPoint)
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
    return data,index



def calculateCordinary(k, data, point, index, positions_test,ifweight,originalPoint):
    """
    :param  K:选取几个k近邻点
    :param  data: 所处聚类的数据
    :param  point: 测试点
    :param index: 第几个测试点
    :param position_test: 这些测试点的实际坐标
    :param  该函数目的是为了求出给定点point在data聚类的k个近邻点，返回的是定位的误差
    :return 返回和实际位置的误差，以及预测坐标
    """
    #   调用了KNN算法！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    result = knn(point, data, k,ifweight,originalPoint)

    predic_position = result
    result = (result - positions_test[index]) ** 2
    error = result.sum(axis=0) ** 0.5             #   相当于勾股定理  求两个坐标之间的距离，也就是求对角线之间的长度
    #print(error)

    if(error > 2):
        print("预测坐标", predic_position)
        print("真实坐标", positions_test[index])
        print(error)
    return error, predic_position

def knn(testPoint, dataset, k, ifWeight,originalPoint):     #选取了K个临近点
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
    #ifWeight=1代表wknn，否则为knn
    if ifWeight == 1:
        k_sum = 0
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
                print(x,y)
                sumx = sumx + (weight[i] / k_sum) * x
                sumy = sumy + (weight[i] / k_sum) * y
            print("jieshu")
            return sumx, sumy
    else:
        if k > datasetSize:
            for i in range(datasetSize):
                sumx = sumx + dataset[sortedDistIndicies[i]][length - 2]
                sumy = sumy + dataset[sortedDistIndicies[i]][length - 1]
                #print((sumx,sumy))
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



