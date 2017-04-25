
import math
import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def stepFunction(w,x):
    return sigmoid(np.inner(w,x))

def Lsimple(w):
    value=(stepFunction(w,[1,0])-1)**2 + (stepFunction(w,[0,1]))**2 + (stepFunction(w,[1,1])-1)**2
    return value

def derivertSigmoidWithInnerProduct(w,x,derivertPåWindex):
    value=  (stepFunction(w,x)**2) * x[derivertPåWindex]* np.exp(-np.inner(w,x))
    return value


def deltaLsimple(w):
    dw1 = (stepFunction(w,[1,0])-1)*derivertSigmoidWithInnerProduct(w,[1,0],0)+\
      (stepFunction(w,[1,1])-1)*derivertSigmoidWithInnerProduct(w,[1,1],0)

    dw2 =  (stepFunction(w,[0,1]))*derivertSigmoidWithInnerProduct(w,[0,1],1) + \
         (stepFunction(w,[1,1])-1)*derivertSigmoidWithInnerProduct(w,[1,1],1)

    return (dw1,dw2)

def updateRule(oldW, learningRate):
    oldW1=oldW[0]
    oldW2=oldW[1]
    gradients= deltaLsimple(oldW)
    newW1 = oldW1 - learningRate * gradients[0]
    newW2 = oldW2 - learningRate * gradients[1]
    # print("gradients: ",gradients)
    # print("w: " ,newW1,newW2)
    return [newW1,newW2]

def runGradientDecent(iterations, learningRate, oneRunValues):
    nextW = [-7, -7]
    oneRunScore=[]
    # print("score: ", Lsimple(nextW))

    for n in range(iterations):
        nextW = updateRule(nextW, learningRate)
        oneRunValues.append(nextW)
        oneRunScore.append(Lsimple(nextW))
        # print("score: ",Lsimple(nextW) )
    return oneRunScore

def findBestW(wGrid,LsimpleListe):
    for row in range(len(wGrid)):
        for colum in range(len(wGrid)):
            # LsimpleList[row,colum]=(Lsimple(wGrid[row,colum]))
            LsimpleList.append(Lsimple(wGrid[row, colum]))

    # min Lsimple(w)
    minsteVerdi = min(LsimpleList)
    # finn tilhørende w verdi
    index = LsimpleList.index(minsteVerdi)
    row = math.floor(index / rowNumber)
    colum = index - (row * rowNumber)
    # print(row)
    # print(colum)
    minWValue = wGrid[row, colum]

    print("minste verdi: ", minsteVerdi, "med wi :", minWValue)


#create wGrid
rowNumber=13

wGrid=np.zeros((rowNumber,rowNumber,2))
for row in range(len(wGrid)):
    for colum in range(len(wGrid)):
        wGrid[row,colum]= [row-6,colum-6]

# print(wGrid)

##put w into l simple
LsimpleList=[]
findBestW(wGrid,LsimpleList)

LowestValuesFromGradientDecent=[]
oneRunW=[]
allLearningRateRuns=[]
# for generation in [1,10,100,1000]:
# for stepLength in [0.0001,0.001,0.01,0.1,1,10,100,1000]:


for stepLength in [0.1,0.01,0.0001]:
    allLearningRateRuns.append(runGradientDecent(10000,stepLength,oneRunW))
    # LowestValuesFromGradientDecent.append(Lsimple(nextW))


import matplotlib.pyplot as plt
plt.figure(1)

# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.

labels=['0,1','0,01','0.0001']
for oneRunScore,label in zip(allLearningRateRuns,labels):
    plt.plot(oneRunScore,label=label)
#
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

# # plt.suptitle('test title', fontsize=20)
# plt.yscale("log")
# plt.ylabel('Value of Lsimple(w)')
# plt.grid(True)

plt.show()