from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import pickle

with open('census.pkl', 'rb') as f:
    X_census_treinamento,Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(f)

# Define a quantidade de arvores que serão usadas
random_forest_census = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)

# Treina o algoritmo com os dados de teste
random_forest_census.fit(X_census_treinamento, Y_census_treinamento)

# Testando o algoritmo com os dados de teste
previsoes = random_forest_census.predict(X_census_teste)

# Porcentagem de acertos é salvo na variavel
acertos = accuracy_score(Y_census_teste, previsoes)
print(acertos)

# Relatorio sobre o algoritmo é gerado e salvo na variavel
classificacao = classification_report(Y_census_teste, previsoes)
print(classificacao)

# Matriz de confusão
cm = ConfusionMatrix(random_forest_census)
cm.fit(X_census_treinamento, Y_census_treinamento)
cm.score(X_census_teste, Y_census_teste)
plt.show()






