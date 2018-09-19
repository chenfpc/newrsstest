# -*- coding: utf-8 -*-
# 导入相应的包
from sklearn import  preprocessing

import scipy.io as sio
import numpy as np
import numpy
import pylab
#import function_simulate as f_sim




# -*- coding=utf-8 -*-
# Kalman filter example demo in Python

# A Python implementation of the example given in pages 11-15 of "An
# Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# University of North Carolina at Chapel Hill, Department of Computer
# Science, TR 95-041,
# http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html

# by Andrew D. Straw
# coding:utf-8


# 这里是假设A=1，H=1的情况


def all_kalman_filter(h_testAll):
    kalman_filter_result = [[0] * h_testAll[0] for i in range(len(h_testAll))]  # 创建二维数组，列数为h_testAll[0]（10列），行数为h_testAll（444行）
    for i in range(0, len(h_testAll)):
        kalman_filter_result[i] = kalman_filter(h_testAll, i)        # kalman_filter_result包含了全部数据的滤波结果
        # print(kalman_filter_result)
    return judge_NLOS(kalman_filter_result, h_testAll)        # 返回计算出的所有权重，此处权重用r表示


def kalman_filter(h_testAll, i):           # 将数据进行卡尔曼滤波
    # 参数初始化
    z = h_testAll[i]
    xhat = [0 for i in range(len(h_testAll[i]))]
    S = [0 for i in range(len(h_testAll[i]))]
    S[0] = z[0]
    Phat = [0 for i in range(len(h_testAll[i]))]
    P = [0 for i in range(len(h_testAll[i]))]
    P[0] = 5
    Q = 0.01
    H = 1
    R = 2
    F = 1
    K = [0 for i in range(len(h_testAll[i]))]

    for k in range(1, len(h_testAll[i])):
        # 预测
        xhat[k] = F * S[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
        Phat[k] = (F ** 2) * P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1

        # 更新
        K[k] = H * Phat[k] / ((H ** 2) * Phat[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
        S[k] = xhat[k] + K[k] * (z[k] - xhat[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        P[k] = (1 - H * K[k]) * Phat[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
    return S


def judge_NLOS(S, h_testAll):       # 传入滤波后的数据，判断出这笔数据包含NLOS数据的概率，也就是权重

    L_m = h_testAll  # 测量获得的原始数据
    L = S            # 经过卡尔曼滤波之后的数据
    # sum = [[0] * h_testAll[0] for i in range(len(h_testAll))]    # 创建二维数组，列数为h_testAll[0]（10列），行数为h_testAll（444行）
    sum = [0 for i in range(len(h_testAll))]
    temp1 = [0 for i in range(len(h_testAll[0]))]
    r = [0 for i in range(len(h_testAll))]
    for x in range(0, len(L)):                 # x为行数  一共444行
        for y in range(0, 10):          # y为每行的元素个数，每行一共十个元素，即十笔RSS数据
            temp1[y] = (L_m[x][y] - L[x][y]) ** 2
            sum[x] = temp1[y] + sum[x]
        r[x] = (sum[x] / len(L)) ** 0.5           # 将所有的权重都存储在了r中
        # print(r[x])
    #print(r)
    return r




