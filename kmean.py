import numpy as np

import matplotlib.pylab as plt


def dataLoader(file):
    # return np.array([i.split() for i in open(file)], dtype="float")
    return np.array(np.loadtxt("testSet.txt"))


def plus(dataSet, k=2):
    lenth, dim = dataSet.shape  # 获取数据维度

    _max = np.max(dataSet, axis=0)  # 线性映射最大值  axis=0列最大值

    _min = np.min(dataSet, axis=0)

    centers = []

    centers.append((_min + (_max - _min) * (np.random.rand(dim))))  # 生成一个随机向量

    centers = np.array(centers)  # 为了保证centers的矩阵结构,而不是向量结构

    for i in range(1, k):

        distanceS = []

        for row in dataSet:
            distanceS.append(np.min(np.linalg.norm(row - centers, axis=1)))  # 计算离多个中心的距离里面最近的那个..

        # 蒙特卡罗法, 假设总距离由各个距离条组成,落在距离条长的上面概率大,可用概率求长条,这里反过来用

        temp = sum(distanceS) * np.random.rand()

        for j in range(lenth):

            temp -= distanceS[j]  # 依次剥离距离条

            if temp < 0:
                centers = np.append(centers, [dataSet[j]], axis=0)  # 保持0轴不塌陷

                break

    return centers


def kmeans(dataSet, k, maxIter=300):
    print("***********************************")
    # initialize with ++

    centers = plus(dataSet, k)

    #
    # plt.scatter(*centers.T, s=200)

    def getLabel(data):

        distanceS = np.linalg.norm(data - centers, axis=1)  # 注意axis是等于1的...

        return np.where(distanceS == np.min(distanceS))[0][0]

    labels = np.ones(len(dataSet))

    j = 0

    while 1 and j < maxIter:

        j += 1


        print(dataSet)
        print(centers)
        label_new = np.array(list(map(getLabel, dataSet)))  # 生成新的标签
        print("lable_new")
        print(label_new)
        if sum(np.abs(labels - label_new)) == 0:  # 判断标签是否改变

            break

        labels = label_new

        print("kaishi")
        print(j)
        for i in range(k):
            print(i)
            print(labels)
            print(dataSet[labels == i])
            centers[i] = np.mean(dataSet[labels == i], axis=0)  # 更新聚类中心

        print(centers)
    SSE = sum(
        [sum([(j - centers[i]).dot(j - centers[i]) for j in dataSet[labels == i]]) for i in range(k)])  # 计算误差平方和

    print("SSE: ", SSE)

    return label_new, centers


# datas = dataLoader("testSet.txt")
#
# labels, centers = kmeans(datas, k=4)
#
# plt.figure(figsize=(15, 12.18))distanceS
# for i in set(labels):
#     plt.scatter(*(datas[labels == i].T), color=np.random.rand(3), s=100)
#
#     plt.scatter(*centers[i], marker='^', s=200)
# plt.show()