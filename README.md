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




## **Conclusões**
- **XGBoost:** Ideal para máxima precisão.
- **CatBoost:** Melhor trade-off em volumes moderados.
- **Regressão Logística:** Melhor escolha para alta escalabilidade e processamento em tempo real.

  
