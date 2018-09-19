import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import  FormatStrFormatter
import svm
scores = []
for i in range(20):
    scores.append(svm.print_svm_score()*100)
plt.title("SVMC分类效果图")
plt.grid(True)
plt.xlabel("次数")
plt.ylabel("准确率(%)")
plt.ylim(95,100)
plt.plot(range(0,20,1),scores,"r-")
plt.show()