# Relatório

### Artefatos Produzidos (disponíveis na pasta [Código Colab](https://github.com/victorhcunha/ArvoreDeDecisao/tree/main/C%C3%B3digo%20Colab))

- Árvore de decisão (sem parâmetros especificados)
- Árvore de decisão interativa (onde seus parâmetros podem ser alterados interativamente)
- Árvore de decisão com sua estrutura exposta textualmente (e parâmetros especificados)

### Desenvolvimento

- Primeiramente foi acrescentado manualmente o nome dos atributos no arquivo de dados do dataset Iris.  Atributos: sepal length in cm, sepal width in cm, petal length in cm, petal width in cm, class.
- Foi feita a importação das bibliotecas python necessárias.
- Foi feita a leitura dos dados contidos no arquivo do dataset.
- Utilizando o train_test_split os dados foram divididos entre dados de treino e dados de teste.
- A árvore foi criada com o DecisionTreeClassifier() e treinada com os dados.
- Foram exibidos os atributos mais importantes e o relatório de métricas.
- Utilizando a biblioteca graphviz a árvore foi gerada graficamente.

#### Ávores
- A primeira árvore foi gerada sem que nenhum de seus parâmetros fosse especificados (está disponível no [arquivo](https://github.com/victorhcunha/ArvoreDeDecisao/blob/main/C%C3%B3digo%20Colab/arvorededecisao1.ipynb)).
- A segunda árvore foi gerada com o uso da biblioteca ipywidgets, sendo uma árvore interativa, ou seja, o usuário pode modificar seus parâmetros ao mesmo tempo em que ela é visualizada (está disponível no [arquivo](https://github.com/victorhcunha/ArvoreDeDecisao/blob/main/C%C3%B3digo%20Colab/arvorededecisaointerativa.ipynb)).
- A terceira árvore trouxe uma forma diferente de visualização, com um aloritmo que imprimiu ela de forma textual, dessa vez aluns parâmetros foram especificados: max_depth=3, min_samples_split=2, min_samples_leaf=2. O graphviz também foi usado para gerá-la graficamente (está disponível no [arquivo](https://github.com/victorhcunha/ArvoreDeDecisao/blob/main/C%C3%B3digo%20Colab/estruturadaarvorededecisao.ipynb))

### Parâmetros utilizados
- max_depth: profundidade máxima da árvore
- min_samples_split: número mínimo de amostras necessárias para dividir um nó interno 
- min_samples_leaf: número mínimo de amostras necessárias para estar em um nó folha

### Comparação Árvore1 (sem parâmetro) vs Árvore3 (com parâmetros), respectivamente

#### Visualmente:

![Árvore1](https://github.com/victorhcunha/ArvoreDeDecisao/blob/main/Imagens/arvore1.png)
![Árvore3](https://github.com/victorhcunha/ArvoreDeDecisao/blob/main/Imagens/arvore3.png)

#### Relatório de métricas:

![Árvore1 - Métricas](https://github.com/victorhcunha/ArvoreDeDecisao/blob/main/Imagens/arvore1Metricas.png)
![Árvore3 - Métricas](https://github.com/victorhcunha/ArvoreDeDecisao/blob/main/Imagens/arvore3Metricas.png)


### Conclusão
A última árvore gerada acabou sendo considerada melhor que a primeira por manter um desempenho satisfatório sem crescer a um nível tão grande de profundidade.
