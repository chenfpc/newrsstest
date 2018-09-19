import numpy as np


def train(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 计算class为1的概率
    p0Num = zeros(numWords)  # 初始化class为0的特征
    p1Num = zeros(numWords)  # 初始化class为1的特征
    p0Denmo = 0.0
    p1Denmo = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denmo += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denmo += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denmo
    p0Vect = p0Num / p0Denmo
    return p0Vect, p1Vect, pAbusive
