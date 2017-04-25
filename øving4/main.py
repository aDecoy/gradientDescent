
import math
import numpy as np
import tkinter

def innerProduct(w,x):
    product=0
    for i in range(len(w)):
        product+=w[i]*x[i]
    return product


def stepFunction(w,x):
    value = 1/(1+pow(math.e,-1* innerProduct(w,x)))
    return value


def Lsimple(w):
    value=pow(stepFunction(w,[1,0])-1,2) + pow(stepFunction(w,[0,1]),2)  + pow(stepFunction(w,[1,1])-1,2)
    return value


#create wGrid
rowNumber=13

wGrid=np.zeros((rowNumber,rowNumber,2))
for row in range(len(wGrid)):
    for colum in range(len(wGrid)):
        wGrid[row,colum]= [row-6,colum-6]

print(wGrid)

##put w into l simple
LsimpleList=[]

for row in range(len(wGrid)):
    for colum in range(len(wGrid)):
        # LsimpleList[row,colum]=(Lsimple(wGrid[row,colum]))
        LsimpleList.append(Lsimple(wGrid[row,colum]))

#min Lsimple(w)
minsteVerdi= min(LsimpleList)
#finn tilhørende w verdi
index = LsimpleList.index(minsteVerdi)
row = math.floor(index/rowNumber)
colum = index-(row*rowNumber)
print(row)
print(colum)
minWValue= wGrid[row,colum]

print("minste verdi: ",minsteVerdi, "med wi :",minWValue)

#show all results for w in a graph
import matplotlib.pyplot as plt
plt.plot(LsimpleList)
plt.ylabel('Value')
plt.show()