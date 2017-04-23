
import math
import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def stepFunction(w,x):
    return sigmoid(np.inner(w,x))



def Lsimple(w):
    value=pow(stepFunction(w,[0,1])-1,2) + pow(stepFunction(w,[0,1]),2)  + pow(stepFunction(w,[1,1])-1,2)
    return value

def derivertSigmoidWithInnerProduct(w,x,derivertPåWindex):
    value= -1*  math.pow(stepFunction(w,x),2)
    value=value* -x[derivertPåWindex]* np.exp(-np.inner(w,x))
    return value


def deltaLsimple(w):
    dw1 = 2*(stepFunction(w,[1,0])-1)*derivertSigmoidWithInnerProduct(w,[1,0],0)+\
      2*(stepFunction(w,[0,1]))*derivertSigmoidWithInnerProduct(w,[0,1],0) +\
      2*(stepFunction(w,[1,1])-1)*derivertSigmoidWithInnerProduct(w,[1,1],0)

    dw2 = 2*(stepFunction(w,[1,0])-1)*derivertSigmoidWithInnerProduct(w,[1,0],1) +\
         2*(stepFunction(w,[0,1]))*derivertSigmoidWithInnerProduct(w,[0,1],1) + \
         2*(stepFunction(w,[1,1])-1)*derivertSigmoidWithInnerProduct(w,[1,1],1)
    # print("w1: ",dw1 ," w2: " ,dw2)
    return (dw1,dw2)

def updateRule(oldW,stepLength):
    oldW1=oldW[0]
    oldW2=oldW[1]
    gradients= deltaLsimple(oldW)
    newW1 = oldW1 - stepLength * gradients[0]
    newW2 = oldW2 - stepLength * gradients[1]
    print("gradients: ",gradients)
    print("w: " ,newW1,newW2)
    return [newW1,newW2]

def runGradientDecent(generations,stepLength,oneRunValues,oneRunScore):
    nextW = [-3, 5]

    for n in range(generations):
        nextW = updateRule(nextW, stepLength)
        oneRunValues.append(nextW)
        oneRunScore.append(Lsimple(nextW))
        print("score: ",Lsimple(nextW) )
    return nextW

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
oneRunScore=[]
# for generation in [1,10,100,1000]:
# for stepLength in [0.0001,0.001,0.01,0.1,1,10,100,1000]:
for stepLength in [0.11]:
    nextW=runGradientDecent(1000,stepLength,oneRunW,oneRunScore)
    # LowestValuesFromGradientDecent.append(Lsimple(nextW))

oneRunLowestScore= min(oneRunScore)
index= oneRunScore.index(oneRunLowestScore)
bestWithGradientDecentW = oneRunW[index]


print("best score: ",oneRunLowestScore, "with w: ", bestWithGradientDecentW)

import matplotlib.pyplot as plt
plt.plot(oneRunScore)
# plt.plot(oneRunW)

# plt.suptitle('test title', fontsize=20)
plt.ylabel('Value of Lsimple (loss)')
plt.show()