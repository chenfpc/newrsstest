# -*- coding: utf-8 -*- 
# 导入相应的包
from sklearn import preprocessing
import navieBayes as byes
import fucntion as f
import scipy.io as sio
from sklearn.model_selection import train_test_split
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq, kmeans, whiten
import numpy as np
import matplotlib.pylab as plt
import svm
import function_simulate as f_sim

import kalman_filter as k

# BASE_URL = r"C:\Users\chenf\Desktop\cailiao\data\\"
BASE_URL = r'C:\Users\chenf\Desktop\\'
H_BASE_URL = r"C:\Users\chenf\Desktop\cailiao\data\\"

# 真实情况的代码！！！！！！！！！！！！！！！！！！！！！！！！！！

#'''
# 训练数据 478
trainingAll = BASE_URL+"all.txt"
training2_4g = BASE_URL + 'all2_4.txt'
training5g = BASE_URL + 'all5.txt'

# 测试数据  74
testingAll = BASE_URL + 'test_all.txt'
testing2_4g = BASE_URL + 'test_2.4g.txt'
testing5g = BASE_URL + 'test_5g.txt'

# 坐标数据
cordinaryAll = BASE_URL + 'position.txt'
cordinaryTest = BASE_URL + 'position_test.txt'

# 数据准备
trainingSet = np.loadtxt(trainingAll)
# 设置数据只包含rss值
# temp = trainingSet[:,0]
# temp = np.column_stack((temp,trainingSet[:,4]))
# temp = np.column_stack((temp,trainingSet[:,8]))
# trainingSet = temp
originalTrainingSet = trainingSet

# 2.4G信号用于子区域划分
# trainingSet_24 = np.loadtxt(training2_4g)

#设置数据只包含rss值
testingSet = np.loadtxt(testingAll)
# temp = testingSet[:,0]
# temp = np.column_stack((temp,testingSet[:,4]))
# temp = np.column_stack((temp,testingSet[:,8]))
# testingSet = temp
originalTestingSet = testingSet

cordinaryAllSet = np.loadtxt(cordinaryAll)
cordinaryTestSet = np.loadtxt(cordinaryTest)    # 为测试点的坐标数据

# 将原始数据做归一化处理     （没有将归一化处理的数据传入到函数当中）
scaler = preprocessing.StandardScaler().fit(trainingSet)
trainingSet = scaler.transform(trainingSet)
testingSet = scaler.transform(testingSet)

# 传入svm参数
clf = svm.print_svm_score()  # 改动了师兄的！！！！！！！！！！！！！！！！！！！！！！！！！！

# 根据rss值划分区域，并返回聚类集合和中心
# 使用wifi 2.4g来进行划分
classfication = f.runClassfication(originalTrainingSet, cordinaryAllSet)       # 调用了函数！！！！！！！！！！！！！！！！！！！！！！

# 在每个子区域中在进行kemans
# clusters = f.getClusters(classfication, 4)                 # 调用了函数！！！！！！！！！！！！！！！！！！！！！！
clusters=[]
# 以下是自己写的
# 调用卡尔曼滤波方法！！！！！！！！！！！！！！！！！！！！
# h_testAll = H_BASE_URL + 'Htest_All.txt'
# h_testAll = np.loadtxt(h_testAll, dtype=int)
# r = k.all_kalman_filter(h_testAll)
# 自己写的结束
# cluster-knn(包括knn和wknn),kmeans-knn/wknn
bayes = byes.NaiveBayesContinuous()
x1 = []
x2 = []
for i in range(70):
    result = f.runRealityClusterKnn(6,originalTrainingSet, originalTestingSet, originalTestingSet,
                           cordinaryAllSet, cordinaryTestSet, classfication, clusters, 1, clf, bayes)
    x1.append(result[0])
    x2.append(result[1])
x1 = np.array(x1)

index1 = x1.argsort()
print(x1[index1[0]])
cdf = x2[index1[0]]
print(cdf)
cdf = np.array(cdf)
fileObject = open(BASE_URL + "cdf_myMethod_Knumber_wKNN_6.txt", "w")
for i in range(len(cdf)):
    fileObject.write(str(cdf[i]))
    fileObject.write("\n")
fileObject.close()
# knn
# f.runKnnRealityKNN(trainingSet, testingSet, cordinaryAllSet, cordinaryTestSet, 5, 0)          # 调用了函数！！！！！！！！！！！！！！！！！！！！！！

'''

# 以下是仿真的代码！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

#训练集，1——2.4是2.4G信号，1.mat是5G信号i
simulate_Train_2_4_Filename = BASE_URL + "1_2.4.mat"
simulate_Train_5_Filename = BASE_URL + "1.mat"
#测试数据，总共3600点，从中抽取100个点吧
simulate_Test_2_4_Filename = BASE_URL + "2_2.4.mat"
simulate_Test_5_Filename = BASE_URL + "2.mat"            #BASE_URL是什么！！！！！！！！！！！！！！           在第十四行

testData_2_4 = sio.loadmat(simulate_Test_2_4_Filename)["tempx"]
testData_5 = sio.loadmat(simulate_Test_5_Filename)["tempx"]
trainData_2_4 = sio.loadmat(simulate_Train_2_4_Filename)["tempx"]
trainData_5 = sio.loadmat(simulate_Train_5_Filename)["tempx"]

alldistance = sio.loadmat(simulate_Test_2_4_Filename)["distance"]

originalTrainingSet = np.concatenate((trainData_2_4, trainData_5), axis=1)
originalTestingSet = np.concatenate((testData_2_4, testData_5), axis=1)
simulatClassfication = f_sim.runClassfication_Simulate(originalTrainingSet, alldistance)          #调用了函数！！！！！！！！！！！！！！！！！！！！！！
testData = f_sim.selectTestSet(originalTestingSet, alldistance, 40)                                #调用了函数！！！！！！！！！！！！！！！！！！！！！！

clf_simulate = svm.print_simulate_svm_score()                                                    #调用了函数！！！！！！！！！！！！！！！！！！！！！！
f_sim.runSimulateClusterKnn(testData[0], testData[0], testData[1], simulatClassfication,0,clf_simulate)            #调用了函数！！！！！！！！！！！！！！！！！！！！！！
# f_sim是什么！！！！！！！！！！！！！

'''








