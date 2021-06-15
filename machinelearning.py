# Importando as bibliotecas necessárias:
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,export_graphviz #arvoredeDecisao e #renderizacao
from sklearn.model_selection import train_test_split #divisaoDadosDeTeste
from sklearn import metrics
import numpy as np
#import pydot
#import graphviz

# Carregando a base de dados:
data = pd.read_csv('iris.data')
data.head()

# Mostrando informações da base de dados
data.info()

# Dividindo os dados em treino e teste:
X_train, X_test, y_train, y_test = train_test_split(data.drop("type",axis=1),data["type"],test_size=0.3)

print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

# Instânciando o objeto classificador:
classifier = DecisionTreeClassifier()

# Treinando o modelo de arvore de decisão:
classifier = classifier.fit(X_train,y_train)

print(classifier.feature_importances_)

for feature,importancia in zip(data.columns,classifier.feature_importances_):
    print("{}:{}".format(feature, importancia))

resultado = classifier.predict(X_test)
print(resultado)


print(metrics.classification_report(y_test,resultado))
