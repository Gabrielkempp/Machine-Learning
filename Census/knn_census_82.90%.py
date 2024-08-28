import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

# Abre o arquivo 'credit.pkl' e carrega os dados serializados usando o pickle.
with open('Data/census.pkl', 'rb') as f:
    X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(f)

# Cria uma instância do classificador K-Nearest Neighbors (KNN) com 5 vizinhos,
knn_census = KNeighborsClassifier(n_neighbors=10)

# Treina o modelo KNN usando os dados de treinamento.
knn_census.fit(X_census_treinamento, Y_census_treinamento)

# Realiza previsões no conjunto de dados de teste e armazena na variável 'previsoes'.
previsoes = knn_census.predict(X_census_teste)

print(accuracy_score(Y_census_teste, previsoes))

# Cria uma matriz de confusão para avaliar o modelo Naive Bayes
cm = ConfusionMatrix(knn_census)

# Treina o modelo com os dados de treinamento
cm.fit(X_census_treinamento, Y_census_treinamento)

# Usa o modelo para fazer previsões com os dados de teste e atualiza a matriz de confusão
cm.score(X_census_teste, Y_census_teste)
plt.show()

# Gera e imprime um relatório com informações detalhadas sobre o desempenho do modelo
# O relatório inclui precisão, recall, F1-score, e outras métricas
print(classification_report(Y_census_teste, previsoes))
