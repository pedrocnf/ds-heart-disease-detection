# HEART DISEASE DETECTION

## Main reference: https://medium.com/@pedrocnf/desvendando-o-cora%C3%A7%C3%A3o-um-projeto-de-classifica%C3%A7%C3%A3o-de-doen%C3%A7as-card%C3%ADacas-com-catboost-4daa8b0b7cff


# Instruções para Configurar o Ambiente e Visualizar os Notebooks

Este repositório contém código Python para análise de dados e modelos de machine learning. Para facilitar a configuração do ambiente e a visualização dos notebooks, siga as etapas abaixo:

## 1. Instalar o Miniconda

Para gerenciar as dependências do projeto, é recomendável usar o Miniconda, uma distribuição leve do Conda, que facilita a criação de ambientes isolados.

Se você ainda não tem o Miniconda instalado, siga as instruções em [Miniconda](https://docs.conda.io/en/latest/miniconda.html) para fazer o download e instalação no seu sistema.

## 2. Criar um Ambiente Conda

Uma vez que o Miniconda esteja instalado, você pode criar um ambiente Conda específico para este projeto utilizando o arquivo `environment.yml`. Este arquivo contém a lista de dependências necessárias, incluindo bibliotecas como Numpy, Matplotlib, Pandas, Scipy, Scikit-learn e LightGBM.

Execute o seguinte comando para criar o ambiente:

```bash
conda env create -f environment.yml
```

## Funções em utils.py

normality_test(df, colunas): Esta função é usada para realizar um teste de normalidade em colunas de um dataframe. Ela itera pelas colunas especificadas em colunas e realiza um teste de Shapiro-Wilk para verificar se os dados em cada coluna seguem uma distribuição normal. Ela retorna um DataFrame que inclui o nome da coluna, o valor-p (p-valor) do teste de normalidade e um indicador de se os dados são considerados normais (p-valor > 0,05).

differences_test(grupo1, grupo2, colunas): Esta função realiza um teste de diferenças entre dois grupos, grupo1 e grupo2, em relação às colunas especificadas em colunas`. Ela usa o teste t de Student independente (t-test) para verificar se há diferenças significativas entre os grupos em relação a cada coluna. Assim como a função anterior, ela retorna um DataFrame que inclui o nome da coluna, o valor-p do teste de diferenças e um indicador de se os grupos são considerados diferentes (p-valor > 0,05).

model_evaluation(params, X_train, y_train, X_test, y_test): Esta função é usada para avaliar o desempenho de um modelo de machine learning, no caso um modelo LightGBM. Ela recebe os parâmetros do modelo (ou usa os padrões se params for None) e os conjuntos de treinamento (X_train e y_train) e teste (X_test e y_test). A função treina o modelo no conjunto de treinamento, faz previsões no conjunto de teste e calcula várias métricas de desempenho, incluindo acurácia, F1-Score, recall, precisão e AUC-ROC. Além disso, ela gera gráficos, como a matriz de confusão e a curva ROC, para visualizar o desempenho do modelo. A função retorna um DataFrame com as métricas de desempenho.

Essas funções são úteis para realizar testes estatísticos, avaliar modelos de machine learning e visualizar os resultados de forma eficaz. Elas podem ser úteis em tarefas de análise de dados e modelagem preditiva.


Detection of Heart Disease using the database: Janosi,Andras, Steinbrunn,William, Pfisterer,Matthias, and Detrano,Robert. (1988). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.


