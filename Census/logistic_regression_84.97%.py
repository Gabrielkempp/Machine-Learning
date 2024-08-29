import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Adiciona os dados as respectivas variaveis (Ja com Label encoder)
with open('Data/census.pkl', 'rb') as f:
    X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(f)

# Atribui o algoritmo a variavel
logistic_census = LogisticRegression(random_state=1)

# Faz o treinamento do algoritmo
logistic_census.fit(X_census_treinamento,Y_census_treinamento)

# Faz as previsões com base no algortmo
previsoes = logistic_census.predict(X_census_teste)

# Verifica a acuracia do algoritmo
print(accuracy_score(Y_census_teste,previsoes))
