# Importando as bibliotecas necessárias:
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,export_graphviz #arvoredeDecisao e #renderizacao
from sklearn.model_selection import train_test_split #divisaoDadosDeTeste
from sklearn import metrics
import numpy as np

# Carregando a base de dados:
iris = pd.read_csv('iris.data')
iris.head()

# Mostrando informações da base de dados
iris.info()
