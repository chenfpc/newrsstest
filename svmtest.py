from sklearn import svm
import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([-2, -2, -1, -1])
c = np.array([[-1,-1]])
clf = svm.SVC(probability=True)
clf.fit(X, y)
print(clf.predict_prjjhjkhkmkoba(c))