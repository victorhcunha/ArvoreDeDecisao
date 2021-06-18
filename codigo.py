# Importando as bibliotecas necessárias:
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,export_graphviz #arvoredeDecisao e #renderizacao
from sklearn.model_selection import train_test_split #divisaoDadosDeTeste
from sklearn import metrics
import numpy as np
# Renderização gráfica
import pydot
import graphviz
# Renderização interativa
from ipywidgets import interactive
from IPython.display import SVG,display
from graphviz import Source

# Carregando a base de dados:
data = pd.read_csv('iris.data')
data.head()

# Mostrando informações da base de dados:
data.info()

# Dividindo os dados em treino e teste:
X_train, X_test, y_train, y_test = train_test_split(data.drop("class",axis=1),data["class"],test_size=0.3)

# Mostrando a forma dos dados:
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

# Instânciando o objeto classificador:
classifier = DecisionTreeClassifier()

# Treinando o modelo de arvore de decisão:
classifier = classifier.fit(X_train,y_train)

# Verificando os atributos mais importantes/relevantes para o modelo treinado:
print(classifier.feature_importances_)

for feature,importancia in zip(data.columns,classifier.feature_importances_):
    print("{}:{}".format(feature, importancia))

# Executando o método predict:
resultado = classifier.predict(X_test)
print(resultado)

# Relatório de métricas:
print(metrics.classification_report(y_test,resultado))

# Renderização de Árvore:
dot_data = export_graphviz( 
         classifier, 
         out_file=None,
         feature_names=data.drop('class',axis=1).columns,
         class_names=['Iris-virginica','Iris-setosa', 'Iris-versicolor'],  
         filled=True, rounded=True,
         proportion=True,
         node_ids=True,
         rotate=False,
         label='all',
         special_characters=True
        )

graph = graphviz.Source(dot_data)  
graph

# Renderização de árvore interativa:
# feature matrix
X,y = data.drop('class',axis=1),data['class']

# features
features_label = data.drop('class',axis=1).columns

# clases
class_label = ['Iris-virginica','Iris-setosa', 'Iris-versicolor']


def arvoreinterativa(crit, split, depth, min_samples_split, min_samples_leaf=0.2):
    estimator = DecisionTreeClassifier(
           random_state = 0 
          ,criterion = crit
          ,splitter = split
          ,max_depth = depth
          ,min_samples_split=min_samples_split
          ,min_samples_leaf=min_samples_leaf
    )
    estimator.fit(X, y)
    graph = Source(export_graphviz(estimator
      , out_file=None
      , feature_names=features_label
      , class_names=class_label
      , impurity=True
      , filled = True))
    display(SVG(graph.pipe(format='svg')))
    return estimator

inter=interactive(arvoreinterativa 
   , crit = ["gini", "entropy"]
   , split = ["best", "random"]
   , depth=[1,2,3,4,5,10,20,30]
   , min_samples_split=(1,5)
   , min_samples_leaf=(1,5))

display(inter)
