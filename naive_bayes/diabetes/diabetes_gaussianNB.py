import pandas as pd
#import numpy as np

survival = pd.read_csv('survival.csv')
from  sklearn import metrics
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

train = survival[0:214]
y_train = train[train.columns[-1]]
x_train = train[train.columns[0:3]]

test = survival[214:-1]
y_test = test[test.columns[-1]]
x_test = test[test.columns[0:3]]
y_pred = gnb.fit(x_train, y_train).predict(x_test)

matriz = metrics.confusion_matrix(y_test,y_pred)


print(survival)
#print(matriz)



TP = 0
FP = 0
FN = 0
VN = 0
cont = 0
for i in y_test:
    if (i==1):
        if(y_pred[cont]==i):
            TP=TP+1
        else:
            FP=FP+1
    if (i==2):
        if(y_pred[cont]==i):
            VN=VN+1
        else:
            FN=FN+1
    cont=cont+1

#print(TP,'/',FP)
#print(FN,'/',VN)


acuracia = ((TP+VN)/(TP+FP+FN+VN))
#print(acuracia)

precisao_positivo = ((TP)/(TP+FP))
#print(precisao_positivo)
recall_positivo = ((TP)/(TP+FN))
#print(recall_positivo)

precisao_negativo = ((VN)/(VN+FN))
#print(precisao_negativo)
recall_negativo = ((VN)/(VN+FP))
#print(recall_negativo)



