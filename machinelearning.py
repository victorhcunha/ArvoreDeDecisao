# Importando as bibliotecas necessárias:
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,export_graphviz #arvoredeDecisao e #renderizacao
from sklearn.model_selection import train_test_split #divisaoDadosDeTeste
from sklearn import metrics
import numpy as np

# Carregando a base de dados:
df_diabetes = pd.read_csv('iris.data')
df_diabetes.head()
