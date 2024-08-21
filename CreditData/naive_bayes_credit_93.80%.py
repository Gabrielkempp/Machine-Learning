import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

# Abre o arquivo 'credit.pkl' e carrega os dados serializados usando o pickle.
with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(f)

# Exibe as dimensões dos conjuntos de dados de treinamento e teste.
print(X_credit_treinamento.shape, Y_credit_treinamento.shape)
print(X_credit_teste.shape, Y_credit_teste.shape)

# Cria uma instância do modelo Gaussian Naive Bayes para classificação
naive_credit_data = GaussianNB()

# Treina o modelo Naive Bayes usando os dados de treinamento.
naive_credit_data.fit(X_credit_treinamento, Y_credit_treinamento)

# Usa o modelo treinado para fazer previsões com o conjunto de teste.
# O método .predict(X) faz com que o modelo aplique o que aprendeu durante o treinamento
# para prever as classes (rótulos).
previsoes =  naive_credit_data.predict(X_credit_teste)
print(previsoes)
print(Y_credit_teste)

# Faz uma comparacão entre os dados reais(Y_credit_teste) e os dados previstos (previsoes)
print(accuracy_score(Y_credit_teste, previsoes))
# print(confusion_matrix(Y_credit_teste, previsoes))
'''
[[428   8]
 [ 23  41]]
 428 -> Clientes que pagam e foram classificados corretemene como clientes que PAGAM
 8 -> Clientes que pagam e foram classificados incorretamente como clientes que NÃO PAGAM
 41 -> Clientes que não pagam e foram classificados corretamente como clientes que NÃO PAGAM
 21 -> Clientes que não pagam e foram classificados incorretamente como clientes que PAGAM
'''

# Cria uma matriz de confusão para avaliar o modelo Naive Bayes
cm = ConfusionMatrix(naive_credit_data)

# Treina o modelo com os dados de treinamento
cm.fit(X_credit_treinamento, Y_credit_treinamento)

# Usa o modelo para fazer previsões com os dados de teste e atualiza a matriz de confusão
cm.score(X_credit_teste, Y_credit_teste)
plt.show()

# Gera e imprime um relatório com informações detalhadas sobre o desempenho do modelo
# O relatório inclui precisão, recall, F1-score, e outras métricas
print(classification_report(Y_credit_teste, previsoes))
