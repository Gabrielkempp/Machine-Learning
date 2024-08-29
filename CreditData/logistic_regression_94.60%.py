import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Adiciona os dados as respectivas variaveis (Ja com Label encoder)
with open('Data/credit.pkl', 'rb') as f:
    X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(f)

# Atribui o algoritmo a variavel
logistic_credit = LogisticRegression(random_state=1)

# Faz o treinamento do algoritmo
logistic_credit.fit(X_credit_treinamento, Y_credit_treinamento)

# Parametro 'B0'
print(logistic_credit.intercept_)

# Coeficiente
print(logistic_credit.coef_)

# Faz as previsões com base no algortmo
previsoes = logistic_credit.predict(X_credit_teste)

# Verifica a acuracia do algoritmo
print(accuracy_score(Y_credit_teste, previsoes))