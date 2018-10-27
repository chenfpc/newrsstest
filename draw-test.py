# import matplotlib.pyplot as plt
# styles = ["-", "-"]
# data = [[1,2,3,4,5],[2.3,1.2,3.5,5.6]]
#
# import numpy as np
# import statsmodels.api as sm # recommended import according to the docs
# import matplotlib.pyplot as plt
# from pandas.core import datetools
# from scipy import stats
# from pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# import seaborn as sns
# BASE_URL = r'C:\Users\chenf\Desktop\\'
# # sample = np.random.uniform(0, 1, 50)
# # ecdf = sm.distributions.ECDF(sample)
# #
# # x = np.linspace(min(sample), max(sample))
# # y = ecdf(x)
# # plt.step(x, y)
# # plt.show()
# def drawCdf(drawData):
#     index = 0
#     for data in drawData:
#         data.sort()
#         plotdata = [[],[]]
#         plotdata[0] = data
#         count = len(plotdata[0])
#         for i in range(count):
#             plotdata[1].append((i+1)/count)
#         plt.plot(plotdata[0],plotdata[1],styles[index],lineWidth=2)
#         index = index + 1
#     plt.show()
#
#
#
# #drawCdf(data)
#
#
# cdf_best = np.loadtxt(BASE_URL + "cdf_best.txt")
# # cdf_knn = np.loadtxt(r"C:\Users\u123\Desktop\cdf_knn.txt")
# cdf_clusterknn = np.loadtxt(BASE_URL + "cdf_clusterknn.txt")
# hist_best,bin_best = np.histogram(cdf_best)
# fig_best = np.cumsum(hist_best/sum(hist_best))
# # hist_knn,bin_knn = np.histogram(cdf_knn)
# # fig_knn = np.cumsum(hist_knn/sum(hist_knn))
# hist_cluster,bin_cluster = np.histogram(cdf_clusterknn)
# fig_cluster = np.cumsum(hist_cluster/sum(hist_cluster))
# bin_best[0]=0
# fig_best = fig_best.tolist()
# fig_best.insert(0,0)
# p1 = plt.plot(bin_best[0:],fig_best,"-r*",label='本文方法')
#
# bin_cluster[0] = 0
# fig_cluster = fig_cluster.tolist()
# fig_cluster.insert(0,0)
# p2 = plt.plot(bin_cluster[0:],fig_cluster,"--b+",label='clusterknn方法')
# # bin_knn[0] = 0
# # fig_knn = fig_knn.tolist()
# # fig_knn.insert(0,0)
# # p3 = plt.plot(bin_knn[0:],fig_knn,":ko",label='K近邻方法')
#
# plt.xlabel("location error（m)")
# plt.ylabel("CDF")
# plt.legend()
# x_ticks = np.arange(0,6.5,0.5)
# y_ticks = np.arange(0,1.1,0.1)
# print(x_ticks)
# plt.xlim([0,6])
# plt.ylim([0,1])
# plt.xticks(x_ticks)
# plt.yticks(y_ticks)
# plt.show()

import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt
from pandas.core import datetools
from scipy import stats
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
import seaborn as sns
BASE_URL = r'C:\Users\chenf\Desktop\\'

def drawLines(txtUrl, style, descrp):
    x = np.loadtxt(BASE_URL+txtUrl)
    hist,bin = np.histogram(x)
    fig = np.cumsum((hist/sum(hist)))
    bin[0] = 0
    fig = fig.tolist()
    fig.insert(0,0)
    plt.plot(bin[0:],fig,style,label=descrp)

# # wknn ,k 取值
# drawLines("cdf_myMethod_Knumber_wKNN_3.txt","--g+",'wknn,k=3')
# drawLines("cdf_myMethod_Knumber_wKNN_4.txt","--y*",'wknn,k=4')
# drawLines("cdf_myMethod_Knumber_wKNN_5.txt","--r*",'wknn,k=5')
# drawLines("cdf_myMethod_Knumber_wKNN_6.txt","--b+",'wknn,k=6')
# plt.savefig("Mymethod-wknn.png")
# plt.title("wknn,when k=3,4,5,6")
#knn , k取值
# drawLines("cdf_myMethod_Knumber_KNN_3.txt","--g+",'wknn,k=3')
# drawLines("cdf_myMethod_Knumber_KNN_4.txt","--y*",'wknn,k=4')
# drawLines("cdf_myMethod_Knumber_KNN_5.txt","--r*",'wknn,k=5')
# drawLines("cdf_myMethod_Knumber_KNN_6.txt","--b+",'wknn,k=6')
# plt.savefig("Mymethod-knn.png")
# plt.title("knn,when k=3,4,5,6")

drawLines("cdf_simulate_myMethod_wKNN.txt","--g+","myMethod with wknn")
drawLines("cdf_simulate_myMethod_KNN.txt","--b+","myMethod with knn")
plt.xlabel("location error（m)")
plt.ylabel("CDF")
plt.legend()
x_ticks = np.arange(0,8,0.5)
y_ticks = np.arange(0,1.1,0.1)
print(x_ticks)
plt.xlim([0,8])
plt.ylim([0,1])
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.show()