# **Detecção de Fraudes em Cartões de Crédito: Análise Comparativa de Modelos de Machine Learning**

**Trabalho de Conclusão de Curso (TCC) – Bacharelado em Ciência de Dados**
**Universidade Virtual do Estado de São Paulo (UNIVESP)**

## **Autores**
- Cleyton de Souza Santos (RA 2106566)
- Fernando Chaves Matos (RA 2013970)
- Jucileide de Andrade Viana (RA 2110414)
- Letícia Stahl de Góes (RA 2106144)
- Luiz Roberto Paviani (RA 2204859)
- Michael Gustavo dos Santos Florentino (RA 2103034)
- Rodrigo da Costa Aglinskas (RA 2103846)

**Orientadora:** Profa. Msc. Fernanda Pereira Guidotti Carneiro


## **Resumo do Projeto**
Este projeto avalia **qual algoritmo de Machine Learning oferece o melhor equilíbrio entre desempenho preditivo e eficiência computacional** para detecção de fraudes em cartões de crédito, utilizando o dataset [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### **Modelos Avaliados**
- Regressão Logística
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- MLP Classifier
- Stacking Ensemble



## **Metodologia**
### **Dataset:** 
- 284.807 transações (0.172% de fraudes).
  
### **Ferramentas:** 
- Databricks Serverless, MLflow, PySpark, Pandas.
  
### **Engenharia de Features:** 
- 22 features derivadas, balanceamento por pesos de classe, escalonamento robusto.

### **Avaliação de Desempenho Preditivo**
- **Métricas:** ROC-AUC, Average Precision (AP), Precision, Recall, F1-Score.
- **Score Ponderado:** `0.20×ROC-AUC + 0.25×AP + 0.15×Precision + 0.20×Recall + 0.20×F1-Score`.

### **Avaliação de Eficiência Computacional**
- **Métricas:** Tempo de inferência, uso de memória, CPU, threads.
- **Score de Eficiência:** `(1 - T_norm)×0.4 + (1 - M_norm)×0.4 + (1 - C_norm)×0.1 + (1 - N_norm)×0.1`.
  *(Tempo e memória têm peso maior: 40% cada.)*

### **Score de Trade-off**
- **Overall Score:** `Weighted_Score × 0.7 + Efficiency_Score × 0.3`.
  *(70% para desempenho preditivo, 30% para eficiência.)*

### **Escalabilidade**
- **Testes:** Volumes de 56.961 a 5.696.100 transações (fatores 1, 10, 50, 100).
  *(Simula cenários de produção e identifica gargalos.)*

### **Resultados**

Quanto ao desempenho preditivo:

| Model              | ROC AUC   | PR AUC    | Precision  | Recall    | F1-Score  | Source | Weighted Score |
|--------------------|-----------|-----------|------------|-----------|-----------|--------|----------------|
| XGBoost            | 0.9585    | 0.8145    | 0.9383     | 0.7755    | 0.8492    | mlflow | 0.8610         |
| RandomForest       | 0.9660    | 0.8105    | 0.9059     | 0.7857    | 0.8415    | mlflow | 0.8572         |
| CatBoost           | 0.9787    | 0.7721    | 0.9259     | 0.7653    | 0.8380    | mlflow | 0.8483         |
| LightGBM           | 0.9698    | 0.7830    | 0.8387     | 0.7959    | 0.8168    | mlflow | 0.8381         |
| MLP_Classifier     | 0.9754    | 0.7657    | 0.8941     | 0.7755    | 0.8306    | mlflow | 0.8418         |
| DecisionTree       | 0.8978    | 0.7321    | 0.8837     | 0.7755    | 0.8261    | mlflow | 0.8155         |
| Logistic_Regression| 0.9683    | 0.7357    | 0.8523     | 0.7653    | 0.8065    | mlflow | 0.8198         |

Quanto ao desempenho computacional nos dados de base (fator 1), sem análise de escalabilidade:
| Ranking | Modelo            | Tempo de Inferência (s) | Uso de Memória (MB) | Uso de CPU (%) | Nº Threads | Score Ef. Computacional |
|---------|-------------------|-----------|--------------|---------|---------|------------------|
| 1º      | CatBoost          | 0.094     | 699.79       | 16.35   | 30      | 0.603            |
| 2º      | DecisionTree      | 0.022     | 701.80       | 24.25   | 30      | 0.600            |
| 3º      | LightGBM          | 0.147     | 700.38       | 20.35   | 30      | 0.594            |
| 4º      | LogisticRegression| 0.022     | 701.80       | 56.75   | 30      | 0.567            |
| 5º      | RandomForest      | 0.560     | 699.65       | 22.45   | 30      | 0.559            |
| 6º      | XGBoost           | 0.752     | 699.44       | 25.45   | 30      | 0.541            |
| 7º      | MLP_Classifier    | 0.581     | 701.80       | 43.25   | 30      | 0.536            |
| 8º      | StackingEnsemble  | 2.136     | 701.80       | 43.25   | 40      | 0.412            |

Quanto à simulação de sobrecarga de dados:
**Score de Eficiência Computacional por Modelo e Volume de Dados**
| Modelo            | Score Fator 1 | Score Fator 100 | Variação (%) |
|-------------------|--------------|-----------------|--------------|
| LogisticRegression| 0.567        | 0.353           | -37.7        |
| DecisionTree      | 0.600        | 0.309           | -48.5        |
| RandomForest      | 0.559        | 0.087           | -84.5        |
| XGBoost           | 0.541        | 0.081           | -85.1        |
| CatBoost          | 0.603        | 0.086           | -85.7        |
| StackingEnsemble  | 0.412        | 0.054           | -86.8        |
| LightGBM          | 0.594        | 0.076           | -87.2        |
| MLPClassifier     | 0.536        | 0.062           | -88.4        |

**Score Médio Geral de Eficiência Computacional por Modelo**
| Ranking | Modelo             | Score Eficiência Médio | Análise                          |
|---------|--------------------|-----------------------|----------------------------------|
| 1º      | LogisticRegression | 0.474                 | Melhor escalabilidade geral      |
| 2º      | DecisionTree       | 0.465                 | Equilíbrio eficiência-velocidade |
| 3º      | CatBoost           | 0.325                 | Líder entre ensemble models      |
| 4º      | LightGBM           | 0.300                 | Eficiência moderada              |
| 5º      | RandomForest       | 0.225                 | Custo de bagging evidente        |
| 6º      | MLPClassifier      | 0.220                 | Redes neurais ineficientes       |
| 7º      | XGBoost            | 0.219                 | Alta complexidade penaliza       |
| 8º      | StackingEnsemble   | 0.165                 | Maior custo computacional        |

Quanto ao trade-off final (overall_score) (70% para desempenho preditivo, 30% para eficiência):
**Volume Base (Fator 1):**
| Ranking | Modelo             | Score Desempenho Preditivo | Score Ef. Comp. | Trade-off |
|---------|--------------------|---------------|------------------|--------------|
| 1º      | CatBoost           | 0.848         | 0.603            | 0.775        |
| 2º      | RandomForest       | 0.857         | 0.559            | 0.768        |
| 3º      | XGBoost            | 0.861         | 0.541            | 0.765        |
| 4º      | LightGBM           | 0.838         | 0.594            | 0.765        |
| 5º      | MLP_Classifier     | 0.842         | 0.536            | 0.750        |
| 6º      | LogisticRegression | 0.820         | 0.567            | 0.744        |
| 7º      | DecisionTree       | 0.815         | 0.600            | 0.751        |
| 8º      | StackingEnsemble   | 0.842         | 0.412            | 0.713        |


**Volume Escalado (Fator 100):**
| Ranking | Modelo             | Trade-off | Variação vs Fator 1 |
|---------|--------------------|--------------|---------------------|
| 1º      | LogisticRegression | 0.680        | -8.6%               |
| 2º      | DecisionTree       | 0.663        | -11.6%              |
| 3º      | XGBoost            | 0.627        | -18.1%              |
| 4º      | RandomForest       | 0.626        | -18.5%              |
| 5º      | CatBoost           | 0.620        | -20.0%              |
| 6º      | LightGBM           | 0.609        | -20.3%              |
| 7º      | MLPClassifier      | 0.608        | -18.9%              |
| 8º      | StackingEnsemble   | 0.606        | -15.0%              |



## **Conclusões**
- **XGBoost:** Ideal para máxima precisão.
- **CatBoost:** Melhor trade-off em volumes moderados.
- **Regressão Logística:** Melhor escolha para alta escalabilidade e processamento em tempo real.

  
