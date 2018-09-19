from sklearn import preprocessing
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

H_BASE_URL = r"C:\Users\Admin\Desktop\data2\\"
h_testAll = H_BASE_URL + 'Htest_All.txt'

h_testAll = np.loadtxt(h_testAll, dtype=int)

r = k.all_kalman_filter(h_testAll)

#print(len(result))

