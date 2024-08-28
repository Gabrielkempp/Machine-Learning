import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

# Abre o aquivo 'census.pkl' e carrega os dados atribuindo à variaveis
with open('Data/census.pkl', 'rb') as f:
    X_census_treinamento,Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(f)

# Exibe as dimensões dos dados de treinamento e teste.
# OneHotEncoder -> quantidade grande de colunas
print("Treinamento:",X_census_treinamento.shape, Y_census_treinamento.shape)
print("Teste:",X_census_teste.shape, Y_census_teste.shape)

# Cria uma instância do modelo Gaussian Naive Bayes para classificação
naive_census = GaussianNB()

# O modelo analisa os dados de entrada (X_census_treinamento) e os rótulos correspondentes (Y_census_treinamento),
# ajustando seus parâmetros internos para que possa reconhecer padrões.
naive_census.fit(X_census_treinamento, Y_census_treinamento)

# Atribui à variavel os dados que foram previstos pelo algoritmo
previsoes = naive_census.predict(X_census_teste)
print('Previsões: ',previsoes)

# Compara a classe exixtente com a previsão feita pelo algoritmo
print(f'Porcentagem de acerto: {(accuracy_score(Y_census_teste, previsoes)*100):.2f}%')
# Neste caso se não executar o escalonamento a probabilidade de acerto é de aprox. 70%

# Matriz de confusão
cm = ConfusionMatrix(naive_census)
cm.fit(X_census_treinamento, Y_census_treinamento)
cm.score(X_census_teste, Y_census_teste)
#plt.show()

print()
print(classification_report(Y_census_teste, previsoes))