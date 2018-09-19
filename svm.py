import numpy as np
def print_svm_score():
    """
    :return: 主要返回的是svm训练出来的clf svc模型
    """
    from sklearn import svm
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    dataFileName = r'D:\范师兄材料\data\all5.txt'
    originalData = np.loadtxt(dataFileName)
    label = []
    newData = originalData[126:217, 0:4]
    for i in range(91):
        label.append(1)
    for i in range(91):
        label.append(-1)
    newData = np.concatenate((newData, originalData[126:217, 8:12]), axis=0)
    for i in range(112):
        label.append(1)
    for i in range(112):
        label.append(-1)
    newData = np.concatenate((newData, originalData[355:467, 8:12]), axis=0)
    newData = np.concatenate((newData, originalData[355:467, 0:4]), axis=0)
    for i in range(120):
        label.append(1)
    for i in range(120):
        label.append(-1)
    label = np.array(label)
    newData = np.concatenate((newData, originalData[235:355, 4:8]), axis=0)
    newData = np.concatenate((newData, originalData[235:355, 0:4]), axis=0)

    trainData, testData, trainLabel, testLabel = train_test_split(newData, label, train_size=0.8)
    clf = svm.SVC(probability=True) #直接使用的是5G的信号，因为5G对nlos影响很大，所以判别很明显
    clf.fit(trainData, trainLabel)
    #result = clf.predict(testData)
    #score = accuracy_score(testLabel, result)
    return clf

def print_simulate_svm_score():
    """
    :return: 主要返回的是svm训练出来的clf svc模型
    """
    import scipy.io as sio
    from sklearn import svm
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    dataFileName = r'D:\范师兄材料\data\1.mat'
    originalData = sio.loadmat(dataFileName)["tempx"]
    label = []

    trainingSet = np.reshape(originalData, (60, 60, 12))
    trainingSet1 = np.zeros((900, 12))

    trainingSet2 = np.zeros((900, 12))

    trainingSet3 = np.zeros((900, 12))
    trainingSet4 = np.zeros((900, 12))


    index = 0
    for i in range(30):
        for j in range(30):
            trainingSet1[index] = trainingSet[i, j, :]

            trainingSet2[index] = trainingSet[i, j + 30, :]

            index = index + 1
    index = 0
    for i in range(30):
        for j in range(30):
            trainingSet4[index] = trainingSet[i + 30, j, :]

            trainingSet3[index] = trainingSet[i + 30, j + 30, :]

            index = index + 1

    newData = trainingSet1[:, 4:8]
    for i in range(900):
        label.append(1)
    for i in range(900):
        label.append(-1)
    newData = np.concatenate((newData, trainingSet3[:, 4:8]), axis=0)


    label = np.array(label)

    trainData, testData, trainLabel, testLabel = train_test_split(newData, label, train_size=0.8)
    clf = svm.SVC( probability=True) #直接使用的是5G的信号，因为5G对nlos影响很大，所以判别很明显
    clf.fit(trainData, trainLabel)
    #result = clf.predict(testData)
    #score = accuracy_score(testLabel, result)
    return clf

