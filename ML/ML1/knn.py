'''
开发时间：2021/9/8 17:15

python 3.8.5

开发人；Lasseford Wang

'''

from numpy import *
import operator
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def classify0(inX,dataSet,labels,k):#knn分类器
    #首先计算已知类别数据集与当前点的距离
    dataSetSize=dataSet.shape[0]#读取数据集的行数，声明为dataSetSize
    diffMat = tile(inX,(dataSetSize,1))-dataSet#
    sqDiffMat = diffMat**2#平方
    sqDistances = sqDiffMat.sum(axis=1)#相加
    distances = sqDistances**0.5#开方


    #按照距离递增次序排序
    sortedDistindicies = distances.argsort()#返回排序结果的索引
    classCount = {}#新建一个词典用来计数

    #选取与当前点距离最小的k个点并且确定出现频率
    for i in range(k):
        voteIlabel = labels[sortedDistindicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1

    #将出现频率进行排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def datareading(filename):
    #读取文件
    fo=open(filename)
    data = fo.readlines()
    fo.close()

    row = len(data)#计算数据项的数量
    dataMat = zeros((row,256))#初始化为零矩阵
    dataLabels = []#新建一个列表，用来存储label

    for i in range(row):
        vals = data[i].split()
        dataMat[i,:] = vals[0:256]
        dataLabels.append(vals[256:].index('1'))

    return dataMat,dataLabels

def test(k):
    dataMat,dataLabels = datareading('D:\\Data\\实验一\\semeion_train.csv')
    testMat,testLabels = datareading('D:\\Data\\实验一\\semeion_test.csv')
    m = testMat.shape[0]
    testnum = int(m*0.1)
    TNum = 0
    for i in range(testnum):
        ans = classify0(testMat[i,:],dataMat,dataLabels,k)
        if(ans==testLabels[i]): TNum+=1
    print("When k is ",k,",the Precision is",TNum/testnum)


    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(dataMat,dataLabels)
    print("When k is ", k, ",the Precision is", knn.score(testMat,testLabels)," by sklearn")



    return TNum/testnum


x=[]
y=[]
for i in range(1,21):
    x.append(i)
    y.append(test(i))

plt.plot(x,y,'b-.')
plt.xlim(0,20)
plt.ylim(0.5,1)
plt.show()