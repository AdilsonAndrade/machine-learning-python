import pandas as pd
from  sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
gnb = GaussianNB()

acidentes = pd.read_csv('acidentes.csv')
base = acidentes


enc = preprocessing.LabelEncoder()
enc.fit(acidentes['BAIRRO'])
bairroEnc = enc.transform(acidentes['BAIRRO'])
base['BAIRRO'] = bairroEnc

#print(base['BAIRRO'])
#print(enc.classes_)

enc = preprocessing.LabelEncoder()
enc.fit(acidentes['TIPO'])
#print(enc.classes_)
tipoEnc = enc.transform(acidentes['TIPO'])
base['TIPO'] = tipoEnc

enc = preprocessing.LabelEncoder()
enc.fit(acidentes['OCORRENCIA'])
print(enc.classes_)
ocorrenciaEnc = enc.transform(acidentes['OCORRENCIA'])
base['OCORRENCIA']=ocorrenciaEnc

#print(base)
#print(enc.inverse_transform(classes))


train = base[0:857]
y_train = train[train.columns[-1]]
x_train = train[train.columns[0:2]]

test = base[857:-1]
y_test = test[test.columns[-1]]
x_test = test[test.columns[0:2]]

y_pred = gnb.fit(x_train, y_train).predict(x_test)

matriz = metrics.confusion_matrix(y_test,y_pred)

#print(y_test)
#print(base.iloc[:, 2])
print(matriz)

TP = 0
FP = 0
FN = 0
VN = 0
cont = 0
for i in y_test:
    if (i==0):
        if(y_pred[cont]==i):
            TP=TP+1
        else:
            FN=FN+1
    if (i==1):
        if(y_pred[cont]==i):
            VN=VN+1
        else:
            FP=FP+1
    cont=cont+1

print(TP,'/',FN)
print(FP,'/',VN)


acuracia = ((TP+VN)/(TP+FP+FN+VN))
print('Acurácia:',acuracia)
precisao = ((TP)/(TP+FP))
print('Precisão:',precisao)
recall = ((TP)/(TP+FN))
print('Recal:',recall)
f1score = ((2 * precisao * recall) / (precisao + recall))
print('F1 Score:',f1score)



