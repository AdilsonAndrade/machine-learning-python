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

#print(base)
#print(enc.classes_)

enc = preprocessing.LabelEncoder()
enc.fit(acidentes['TIPO'])
#print(enc.classes_)
tipoEnc = enc.transform(acidentes['TIPO'])
base['TIPO'] = tipoEnc

#print(base)
#print(enc.inverse_transform(classes))


enc = preprocessing.LabelEncoder()
enc.fit(acidentes['OCORRENCIA'])
#print(enc.classes_)
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


#print(base.iloc[:, 2])
print(matriz)
