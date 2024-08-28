import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

# Abre o arquivo 'credit.pkl' e carrega os dados serializados usando o pickle.
with open('Data/credit.pkl', 'rb') as f:
    X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(f)

# Cria uma instância do classificador K-Nearest Neighbors (KNN) com 5 vizinhos,
# distância Euclidiana ('minkowski' com p=2).
knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

# Treina o modelo KNN usando os dados de treinamento.
knn_credit.fit(X_credit_treinamento, Y_credit_treinamento)

# Realiza previsões no conjunto de dados de teste e armazena na variável 'previsoes'.
previsoes = knn_credit.predict(X_credit_teste)

print(accuracy_score(Y_credit_teste, previsoes))

# Cria uma matriz de confusão para avaliar o modelo Naive Bayes
cm = ConfusionMatrix(knn_credit)

# Treina o modelo com os dados de treinamento
cm.fit(X_credit_treinamento, Y_credit_treinamento)

# Usa o modelo para fazer previsões com os dados de teste e atualiza a matriz de confusão
cm.score(X_credit_teste, Y_credit_teste)
plt.show()

# Gera e imprime um relatório com informações detalhadas sobre o desempenho do modelo
# O relatório inclui precisão, recall, F1-score, e outras métricas
print(classification_report(Y_credit_teste, previsoes))
