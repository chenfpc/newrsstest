import matplotlib.pyplot as plt
styles = ["-", "-"]
data = [[1,2,3,4,5],[2.3,1.2,3.5,5.6]]

import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt

BASE_URL = r'C:\Users\chenf\Desktop\\'
# sample = np.random.uniform(0, 1, 50)
# ecdf = sm.distributions.ECDF(sample)
#
# x = np.linspace(min(sample), max(sample))
# y = ecdf(x)
# plt.step(x, y)
# plt.show()
def drawCdf(drawData):
    index = 0
    for data in drawData:
        data.sort()
        plotdata = [[],[]]
        plotdata[0] = data
        count = len(plotdata[0])
        for i in range(count):
            plotdata[1].append((i+1)/count)
        plt.plot(plotdata[0],plotdata[1],styles[index],lineWidth=2)
        index = index + 1
    plt.show()

#drawCdf(data)
from pandas.core import datetools
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
import seaborn as sns
cdf_best = np.loadtxt(BASE_URL + "cdf_best.txt")
# cdf_knn = np.loadtxt(r"C:\Users\u123\Desktop\cdf_knn.txt")
# cdf_clusterknn = np.loadtxt(r"C:\Users\u123\Desktop\cdf_clusterknn.txt")
hist_best,bin_best = np.histogram(cdf_best)
fig_best = np.cumsum(hist_best/sum(hist_best))
# hist_knn,bin_knn = np.histogram(cdf_knn)
# fig_knn = np.cumsum(hist_knn/sum(hist_knn))
# hist_cluster,bin_cluster = np.histogram(cdf_clusterknn)
# fig_cluster = np.cumsum(hist_cluster/sum(hist_cluster))
bin_best[0]=0
fig_best = fig_best.tolist()
fig_best.insert(0,0)
p1 = plt.plot(bin_best[0:],fig_best,"-k*",label='本发明方法')

# bin_cluster[0] = 0
# fig_cluster = fig_cluster.tolist()
# fig_cluster.insert(0,0)
# p2 = plt.plot(bin_cluster[0:],fig_cluster,"--k+",label='聚类K近邻方法')
# bin_knn[0] = 0
# fig_knn = fig_knn.tolist()
# fig_knn.insert(0,0)
# p3 = plt.plot(bin_knn[0:],fig_knn,":ko",label='K近邻方法')

plt.xlabel("定位误差（m)")
plt.ylabel("CDF")
plt.legend()
x_ticks = np.arange(0,6.5,0.5)
y_ticks = np.arange(0,1.1,0.1)
print(x_ticks)
plt.xlim([0,6])
plt.ylim([0,1])
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.show()