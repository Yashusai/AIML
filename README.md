# AIML
3.......CANDIDATE ELIMINATION..................

import numpy as np 
import pandas as pd
data=pd.DataFrame(data=pd.read_csv('data.csv')) 
concepts=np.array(data.iloc[:,0:-1]) 
target=np.array(data.iloc[:,-1]) 
def learn(concepts,target): 
specific_h=concepts[0].copy() 
print("intilization of specific_h and general_h") 
print(specific_h) 
general_h=[["?" for i in range(len(specific_h))] for i in range(len(specific_h))] 
print(general_h) for i,h in enumerate(concepts): 
if target[i]=="yes": 
for x in range(len(specific_h)): 
if h[x]!=specific_h[x]: 
specific_h[x]='?' 
general_h[x][x]='?' 
if target[i]=="no": 
for x in range(len(specific_h)): 
if h[x]!=specific_h[x]: 
general_h[x][x]=specific_h[x]
else: 
general_h[x][x]='?' 
print("steps of candidate elimination algorithm",i+1) 
print("specific_h",i+1,"\n") 
print(specific_h)
print("general_h",i+1,"\n") 
print(general_h) 
indices=[i for i,val in enumerate(general_h) if val==['?','?','?','?','?','?']]
for i in indices:
general_h.remove(['?','?','?','?','?','?']) 
return specific_h,general_h 
s_final,g_final=learn(concepts,target) 
print("Final specific_h:",s_final,sep="\n") 
print("final general_h:",g_final,sep="\n")

Dataset: (file name: data.csv) 
sunny,warm,normal,strong,warm,same,yes
sunny,warm,high,strong,warm,same,yes
rain,cold,high,strong,warm,change,no
sunny,warm,hgh,strong,cool,change,yes



4.........ID3 ALGORITHM...........

import numpy as np
import math
from data_loader import read_data
class Node:
 def __init__(self,attribute):
 self.attribute=attribute
 self.children=[]
 self.answer=""
 #def __str__(self):
 #return self.attribute
def sub(data,col,delete):
 dict={}
 items=np.unique(data[:,col])
 count=np.zeros((items.shape[0],1),dtype=np.int32)
 for x in range(items.shape[0]):
 for y in range(data.shape[0]):
 if data[y,col]==items[x]:
 count[x]+=1
 for x in range(items.shape[0]):
 dict[items[x]]=np.empty((int(count[x]),data.shape[1]),dtype='|S32')
 pos=0
 for y in range(data.shape[0]):
 if data[y,col]==items[x]:
 dict[items[x]][pos]=data[y]
 pos+=1
 if delete:
 dict[items[x]]=np.delete(dict[items[x]],col,1)
 return items,dict
def entropy(s):
 items=np.unique(s)
 if items.size==1:
 return 0
 counts=np.zeros((items.shape[0],1))
 sums=0
 for x in range(items.shape[0]):
 counts[x]=sum(s==items[x])/(s.size*1.0)
 for count in counts:
 sums+=-1*count*math.log(count,2)
 return sums
def gain(data,col):
 items,dict=sub(data,col,delete=False)
 total_size=data.shape[0]
 entropies=np.zeros((items.shape[0],1))
 intrinsic=np.zeros((items.shape[0],1))
 for x in range((items.shape[0])):
 ratio=dict[items[x]].shape[0]/(total_size*1.0)
 entropies[x]=ratio*entropy(dict[items[x]][:,-1])
 intrinsic[x]=ratio*math.log(ratio,2)
 total_entropy=entropy(data[:,-1])
 iv =-1*sum(intrinsic)
 for x in range(entropies.shape[0]):
 total_entropy-=entropies[x]
 return (total_entropy/iv)
def create(data,metadata):
 if(np.unique(data[:,-1])).shape[0]==1:
 node=Node("")
 node.answer=np.unique(data[:,-1])[0]
 return node
 gains=np.zeros((data.shape[1]-1,1))
 for col in range(data.shape[-1]-1):
 gains[col]=gain(data,col)
 split=np.argmax(gains)
 node=Node(metadata[split])
 metadata=np.delete(metadata,split,0)
 items,dict=sub(data,split,delete=True)
 for x in range(items.shape[0]):
 child=create(dict[items[x]],metadata)
 node.children.append((items[x],child))
 return node
def empty(size):
 s=""
 for x in range(size):
 s+=""
 return s
def print_tree(node,level):
 if node.answer!="":
 print(empty(level),node.answer)
 return
 print(empty(level),node.attribute)
 for value,n in node.children:
 print(empty(level+1),value)
 print_tree(n,level+2)
metadata,traindata=read_data("data1.csv")
data=np.array(traindata)
node=create(data,metadata)
print_tree(node,0)
data_loader.py [another supporting file]
import csv
def read_data(filename):
 with open(filename, 'r') as csvfile:
 datareader = csv.reader(csvfile, delimiter=',')
 headers = next(datareader)
 metadata = []
 traindata = []
 for name in headers:
 metadata.append(name)
 for row in datareader:
 traindata.append(row)
 return (metadata, traindata)
 
Data set: (file name: data1.csv)
outlook,temprature,humidity,wind,palytennis
sunny,hot,high,weak,no
sunny,hot,high,strong,no
overcast,hot,high,weak,yes
rain,mild,high,weak,yes
rain,cool,normal,weak,yes
rain,cool,normal,strong,no
overcast,cool,normal,strong,yes
sunny,mild,high,weak,no
sunny,cool,normal,weak,yes
rain,mild,normal,weak,yes
sunny,mild,normal,strong,yes
overcast,mild,high,strong,yes
overcast,hot,normal,weak,yes
rain,mild,high,strong,no


5..........ARTIFICIAL NEURAL NETWORK.................

import numpy as np
x=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
x=x/np.amax(x,axis=0)
y=y/100
def sigmoid(x):
 return (1/(1+np.exp(-x)))
def derivatives_sigmoid(x):
 return x*(1-x)
epoch=7000
lr=0.1
inputlayer_neuron=2
hiddenlayer_neuron=3
output_neuron=1
wh=np.random.uniform(size=(inputlayer_neuron,hiddenlayer_neuron))
bh=np.random.uniform(size=(1,hiddenlayer_neuron))
wout=np.random.uniform(size=(hiddenlayer_neuron,output_neuron))
bout=np.random.uniform(size=(1,output_neuron))
for i in range(epoch):
 hinp1=np.dot(x,wh)
 hinp=hinp1+bh
 hlayer_act=sigmoid(hinp)
 outinp1=np.dot(hlayer_act,wout)
 outinp=outinp1+bout
 output=sigmoid(outinp)
 EO=y-output
 outgrad=derivatives_sigmoid(output)
 d_output=EO*outgrad
 EH=d_output.dot(wout.T)
 hiddengrad=derivatives_sigmoid(hlayer_act)
 d_hiddenlayer=EH*hiddengrad
 wout+=hlayer_act.T.dot(d_output*lr)
 wh+=x.T.dot(d_hiddenlayer)*lr
print("input:\n"+str(x))
print("actual output:\n"+str(y))
print("predicted output:\n",output)


6...........BAYESIAN CLASSIFIER.......................

print("\nNaive Bayes Classifier for concept learning problem")
import csv
import random
import math
import operator
def safe_div(x, y):
 if y == 0:
 return 0
 return x / y
def loadCsv(filename):
 lines = csv.reader(open(filename))
 dataset = list(lines)
 for i in range(len(dataset)):
 dataset[i] = [float(x) for x in dataset[i]]
 return dataset
def splitDataset(dataset, splitRatio):
 trainSize = int(len(dataset) * splitRatio)
 trainSet = []
 copy = list(dataset)
 i = 0
 while len(trainSet) < trainSize:
 # index = random.randrange(len(copy))
 trainSet.append(copy.pop(i))
 return [trainSet, copy]
def separateByClass(dataset):
 separated = {}
 for i in range(len(dataset)):
 vector = dataset[i]
 if (vector[-1] not in separated):
 separated[vector[-1]] = []
 separated[vector[-1]].append(vector)
 return separated
def mean(numbers):
 return safe_div(sum(numbers), float(len(numbers)))
def stdev(numbers):
 avg = mean(numbers)
 variance = safe_div(sum([pow(x - avg, 2) for x in numbers]), float(len(numbers) - 1))
 return math.sqrt(variance)
def summarize(dataset):
 summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
 del summaries[-1]
 return summaries
def summarizeByClass(dataset):
 separated = separateByClass(dataset)
 summaries = {}
 for classValue, instances in separated.items():
 summaries[classValue] = summarize(instances)
 return summaries
def calculateProbability(x, mean, stdev):
 exponent = math.exp(-safe_div(math.pow(x - mean, 2), (2 * math.pow(stdev, 2))))
 final = safe_div(1, (math.sqrt(2 * math.pi) * stdev)) * exponent
 return final
def calculateClassProbabilities(summaries, inputVector):
 probabilities = {}
 for classValue, classSummaries in summaries.items():
 probabilities[classValue] = 1
 for i in range(len(classSummaries)):
 mean, stdev = classSummaries[i]
 x = inputVector[i]
 probabilities[classValue] *= calculateProbability(x, mean, stdev)
 return probabilities
def predict(summaries, inputVector):
 probabilities = calculateClassProbabilities(summaries, inputVector)
 bestLabel, bestProb = None, -1
 for classValue, probability in probabilities.items():
 if bestLabel is None or probability > bestProb:
 bestProb = probability
 bestLabel = classValue
 return bestLabel
def getPredictions(summaries, testSet):
 predictions = []
 for i in range(len(testSet)):
 result = predict(summaries, testSet[i])
 predictions.append(result)
 return predictions
def getAccuracy(testSet, predictions):
 correct = 0
 for i in range(len(testSet)):
 if testSet[i][-1] == predictions[i]:
 correct += 1
 accuracy = safe_div(correct, float(len(testSet))) * 100.0
 return accuracy
def main():
 filename = 'NaiveBayes ConceptLearning.csv'
 splitRatio = 0.75
 dataset = loadCsv(filename)
 trainingSet, testSet = splitDataset(dataset, splitRatio)
 print('Split {0} rows into'.format(len(dataset)))
 print('Number of Training data: ' + (repr(len(trainingSet))))
 print('Number of Test Data: ' + (repr(len(testSet))))
 print("\nThe values assumed for the concept learning attributes are\n")
 print(
 "OUTLOOK=> Sunny=1 Overcast=2 Rain=3\nTEMPERATURE=> Hot=1 Mild=2
Cool=3\nHUMIDITY=> High=1 Normal=2\nWIND=> Weak=1 Strong=2")
 print("TARGET CONCEPT:PLAY TENNIS=> Yes=10 No=5")
 print("\nThe Training set are:")
 for x in trainingSet:
 print(x)
 print("\nThe Test data set are:")
 for x in testSet:
 print(x)
 print("\n")
 # prepare model
 summaries = summarizeByClass(trainingSet)
 # test model
 predictions = getPredictions(summaries, testSet)
 actual = []
 for i in range(len(testSet)):
 vector = testSet[i]
 actual.append(vector[-1])
 # Since there are five attribute values, each attribute constitutes to 20% accuracy. So if all attributes
match with predictions then 100% accuracy
 print('Actual values: {0}%'.format(actual))
 print('Predictions: {0}%'.format(predictions))
 accuracy = getAccuracy(testSet, predictions)
 print('Accuracy: {0}%'.format(accuracy))
main()

Data Set: (file name: NaiveBayes ConceptLearning.csv)
1,1,1,1,5
1,1,1,2,5
2,1,1,2,10
3,2,1,1,10
3,3,2,1,10
3,3,2,2,5
2,3,2,2,10
1,2,1,1,5
1,3,2,1,10
3,2,2,2,10
1,2,2,2,10
2,2,1,2,10
2,1,2,1,10
3,2,1,2,5
1,2,1,2,10
1,2,1,2,5

7..........EM ALGORITHM..........

from sklearn.cluster import KMeans
#from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("kmeansdata.csv")
df1=pd.DataFrame(data)
print(df1)
f1 = df1['Distance_Feature'].values
f2 = df1['Speeding_Feature'].values
X=np.matrix(list(zip(f1,f2)))
plt.plot()
plt.xlim([0, 100])
plt.ylim([0, 50])
plt.title('Dataset')
plt.ylabel('Speeding_Feature')
plt.xlabel('Distance_Feature')
plt.scatter(f1,f2)
plt.show()
# create new plot and data
plt.plot()
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']
# KMeans algorithm
#K = 3
kmeans_model = KMeans(n_clusters=3).fit(X)
plt.plot()
for i, l in enumerate(kmeans_model.labels_):
 plt.plot(f1[i], f2[i], color=colors[l], marker=markers[l],ls='None')
 plt.xlim([0, 100])
 plt.ylim([0, 50])
plt.show()

Data Set: ( filename: kmeansdata.csv)
Driver_ID,Distance_Feature,Speeding_Feature
3423311935,71.24,28
3423313212,52.53,25
3423313724,64.54,27
3423311373,55.69,22
3423310999,54.58,25
3423313857,41.91,10
3423312432,58.64,20
3423311434,52.02,8
3423311328,31.25,34
3423312488,44.31,19
3423311254,49.35,40
3423312943,58.07,45
3423312536,44.22,22
3423311542,55.73,19
3423312176,46.63,43
3423314176,52.97,32
3423314202,46.25,35
3423311346,51.55,27
3423310666,57.05,26
3423313527,58.45,30
3423312182,43.42,23
3423313590,55.68,37
3423312268,55.15,18


8.............KNN............

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
iris=datasets.load_iris()
iris_data=iris.data
iris_labels=iris.target
print(iris_data)
print(iris_labels)
x_train, x_test, y_train, y_test=train_test_split(iris_data,iris_labels,test_size=0.30)
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print('confusion matrix is as follows')
print(confusion_matrix(y_test,y_pred))
print('Accuracy metrics')
print(classification_report(y_test,y_pred))
Data Set:
5.1,3.5,1.4,0.2,Iris-setosa,
4.9,3,1.4,0.2,Iris-setosa,
4.7,3.2,1.3,0.2,Iris-setosa,
4.6,3.1,1.5,0.2,Iris-setosa,
5,3.6,1.4,0.2,Iris-setosa,
5.4,3.9,1.7,0.4,Iris-setosa,
4.6,3.4,1.4,0.3,Iris-setosa,
5,3.4,1.5,0.2,Iris-setosa,
4.4,2.9,1.4,0.2,Iris-setosa,
4.9,3.1,1.5,0.1,Iris-setosa,
5.4,3.7,1.5,0.2,Iris-setosa,
4.8,3.4,1.6,0.2,Iris-setosa,
4.8,3,1.4,0.1,Iris-setosa,
4.3,3,1.1,0.1,Iris-setosa,
5.8,4,1.2,0.2,Iris-setosa,
5.7,4.4,1.5,0.4,Iris-setosa,
5.4,3.9,1.3,0.4,Iris-setosa,
5.1,3.5,1.4,0.3,Iris-setosa,
5.7,3.8,1.7,0.3,Iris-setosa,
5.1,3.8,1.5,0.3,Iris-setosa,
5.4,3.4,1.7,0.2,Iris-setosa,
5.1,3.7,1.5,0.4,Iris-setosa,
4.6,3.6,1,0.2,Iris-setosa,
5.1,3.3,1.7,0.5,Iris-setosa,
4.8,3.4,1.9,0.2,Iris-setosa,
5,3,1.6,0.2,Iris-setosa,
5,3.4,1.6,0.4,Iris-setosa,
5.2,3.5,1.5,0.2,Iris-setosa,
5.2,3.4,1.4,0.2,Iris-setosa,
4.7,3.2,1.6,0.2,Iris-setosa,
4.8,3.1,1.6,0.2,Iris-setosa,
5.4,3.4,1.5,0.4,Iris-setosa,
5.2,4.1,1.5,0.1,Iris-setosa,
5.5,4.2,1.4,0.2,Iris-setosa,
4.9,3.1,1.5,0.1,Iris-setosa,
5,3.2,1.2,0.2,Iris-setosa,
5.5,3.5,1.3,0.2,Iris-setosa,
4.9,3.1,1.5,0.1,Iris-setosa,
4.4,3,1.3,0.2,Iris-setosa,
5.1,3.4,1.5,0.2,Iris-setosa,
5,3.5,1.3,0.3,Iris-setosa,
4.5,2.3,1.3,0.3,Iris-setosa,
4.4,3.2,1.3,0.2,Iris-setosa,
5,3.5,1.6,0.6,Iris-setosa,
5.1,3.8,1.9,0.4,Iris-setosa,
4.8,3,1.4,0.3,Iris-setosa,
5.1,3.8,1.6,0.2,Iris-setosa,
4.6,3.2,1.4,0.2,Iris-setosa,
5.3,3.7,1.5,0.2,Iris-setosa,
5,3.3,1.4,0.2,Iris-setosa,
7,3.2,4.7,1.4,Iris-versicolor,
6.4,3.2,4.5,1.5,Iris-versicolor,
6.9,3.1,4.9,1.5,Iris-versicolor,
5.5,2.3,4,1.3,Iris-versicolor,
6.5,2.8,4.6,1.5,Iris-versicolor,
5.7,2.8,4.5,1.3,Iris-versicolor,
6.3,3.3,4.7,1.6,Iris-versicolor,
4.9,2.4,3.3,1,Iris-versicolor,
6.6,2.9,4.6,1.3,Iris-versicolor,
5.2,2.7,3.9,1.4,Iris-versicolor,
5,2,3.5,1,Iris-versicolor,
5.9,3,4.2,1.5,Iris-versicolor,
6,2.2,4,1,Iris-versicolor,
6.1,2.9,4.7,1.4,Iris-versicolor,
5.6,2.9,3.6,1.3,Iris-versicolor,
6.7,3.1,4.4,1.4,Iris-versicolor,
5.6,3,4.5,1.5,Iris-versicolor,
5.8,2.7,4.1,1,Iris-versicolor,
6.2,2.2,4.5,1.5,Iris-versicolor,
5.6,2.5,3.9,1.1,Iris-versicolor,
5.9,3.2,4.8,1.8,Iris-versicolor,
6.1,2.8,4,1.3,Iris-versicolor,
6.3,2.5,4.9,1.5,Iris-versicolor,
6.1,2.8,4.7,1.2,Iris-versicolor,
6.4,2.9,4.3,1.3,Iris-versicolor,
6.6,3,4.4,1.4,Iris-versicolor,
6.8,2.8,4.8,1.4,Iris-versicolor,
6.7,3,5,1.7,Iris-versicolor,
6,2.9,4.5,1.5,Iris-versicolor,
5.7,2.6,3.5,1,Iris-versicolor,
5.5,2.4,3.8,1.1,Iris-versicolor,
5.5,2.4,3.7,1,Iris-versicolor,
5.8,2.7,3.9,1.2,Iris-versicolor,
6,2.7,5.1,1.6,Iris-versicolor,
5.4,3,4.5,1.5,Iris-versicolor,
6,3.4,4.5,1.6,Iris-versicolor,
6.7,3.1,4.7,1.5,Iris-versicolor,
6.3,2.3,4.4,1.3,Iris-versicolor,
5.6,3,4.1,1.3,Iris-versicolor,
5.5,2.5,4,1.3,Iris-versicolor,
5.5,2.6,4.4,1.2,Iris-versicolor,
6.1,3,4.6,1.4,Iris-versicolor,
5.8,2.6,4,1.2,Iris-versicolor,
5,2.3,3.3,1,Iris-versicolor,
5.6,2.7,4.2,1.3,Iris-versicolor,
5.7,3,4.2,1.2,Iris-versicolor,
5.7,2.9,4.2,1.3,Iris-versicolor,
6.2,2.9,4.3,1.3,Iris-versicolor,
5.1,2.5,3,1.1,Iris-versicolor,
5.7,2.8,4.1,1.3,Iris-versicolor,
6.3,3.3,6,2.5,Iris-virginica,
5.8,2.7,5.1,1.9,Iris-virginica,
7.1,3,5.9,2.1,Iris-virginica,
6.3,2.9,5.6,1.8,Iris-virginica,
6.5,3,5.8,2.2,Iris-virginica,
7.6,3,6.6,2.1,Iris-virginica,
4.9,2.5,4.5,1.7,Iris-virginica,
7.3,2.9,6.3,1.8,Iris-virginica,
6.7,2.5,5.8,1.8,Iris-virginica,
7.2,3.6,6.1,2.5,Iris-virginica,
6.5,3.2,5.1,2,Iris-virginica,
6.4,2.7,5.3,1.9,Iris-virginica,
6.8,3,5.5,2.1,Iris-virginica,
5.7,2.5,5,2,Iris-virginica,
5.8,2.8,5.1,2.4,Iris-virginica,
6.4,3.2,5.3,2.3,Iris-virginica,
6.5,3,5.5,1.8,Iris-virginica,
7.7,3.8,6.7,2.2,Iris-virginica,
7.7,2.6,6.9,2.3,Iris-virginica,
6,2.2,5,1.5,Iris-virginica,
6.9,3.2,5.7,2.3,Iris-virginica,
5.6,2.8,4.9,2,Iris-virginica,
7.7,2.8,6.7,2,Iris-virginica,
6.3,2.7,4.9,1.8,Iris-virginica,
6.7,3.3,5.7,2.1,Iris-virginica,
7.2,3.2,6,1.8,Iris-virginica,
6.2,2.8,4.8,1.8,Iris-virginica,
6.1,3,4.9,1.8,Iris-virginica,
6.4,2.8,5.6,2.1,Iris-virginica,
7.2,3,5.8,1.6,Iris-virginica,
7.4,2.8,6.1,1.9,Iris-virginica,
7.9,3.8,6.4,2,Iris-virginica,
6.4,2.8,5.6,2.2,Iris-virginica,
6.3,2.8,5.1,1.5,Iris-virginica,
6.1,2.6,5.6,1.4,Iris-virginica,
7.7,3,6.1,2.3,Iris-virginica,
6.3,3.4,5.6,2.4,Iris-virginica,
6.4,3.1,5.5,1.8,Iris-virginica,
6,3,4.8,1.8,Iris-virginica,
6.9,3.1,5.4,2.1,Iris-virginica,
6.7,3.1,5.6,2.4,Iris-virginica,
6.9,3.1,5.1,2.3,Iris-virginica,
5.8,2.7,5.1,1.9,Iris-virginica,
6.8,3.2,5.9,2.3,Iris-virginica,
6.7,3.3,5.7,2.5,Iris-virginica,
6.7,3,5.2,2.3,Iris-virginica,
6.3,2.5,5,1.9,Iris-virginica,
6.5,3,5.2,2,Iris-virginica,
6.2,3.4,5.4,2.3,Iris-virginica,
5.9,3,5.1,1.8,Iris-virginica,


9...........LOCALLY WEIGHTED REGRESION..............

from math import ceil
import numpy as np
from scipy import linalg
def lowess(x, y, f=2./3., iter=3):
 n = len(x)
 r = int(ceil(f*n))
 h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
 w = np.clip(np.abs((x[:,None] - x[None,:]) / h), 0.0, 1.0)
 w = (1 - w**3)**3
 yest = np.zeros(n)
 delta = np.ones(n)
 for iteration in range(iter):
 for i in range(n):
 weights = delta * w[:,i]
 b = np.array([np.sum(weights*y), np.sum(weights*y*x)])
 A = np.array([[np.sum(weights), np.sum(weights*x)],
 [np.sum(weights*x), np.sum(weights*x*x)]])
 beta = linalg.solve(A, b)
 yest[i] = beta[0] + beta[1]*x[i]
 residuals = y - yest
 s = np.median(np.abs(residuals))
 delta = np.clip(residuals / (6.0 * s), -1, 1)
 delta = (1 - delta**2)**2
 return yest
if __name__ == '__main__':
 import math
 n = 100
 x = np.linspace(0, 2 * math.pi, n)
 print("==========================values of x=====================")
 print(x)
 y = np.sin(x) + 0.3*np.random.randn(n)
 print("================================Values of y===================")
 print(y)
 f = 0.25
 yest = lowess(x, y, f=f, iter=3)
 import pylab as pl
 pl.clf()
 pl.plot(x, y, label='y noisy')
 pl.plot(x, yest, label='y pred')
 pl.legend()
 pl.show()
