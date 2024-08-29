# Neste teste serão apagados os dados em que o risco seja 'moderado' fara facilitar a compreenção

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

# Importa os dados e atribui à variaveis
with open('Data/risco_credito.pkl','rb') as f:
    X_risco_credito, Y_risco_credito = pickle.load(f)

# Deleta os dados mencionados no cabecalho
X_risco_credito = np.delete(X_risco_credito,[2,7,11], axis=0)
Y_risco_credito = np.delete(Y_risco_credito,[2,7,11], axis=0)

# Atribui o algoritmo a variavel
logistic_risco_credito = LogisticRegression(random_state=1)

# Faz o treinamento do algoritmo
logistic_risco_credito.fit(X_risco_credito,Y_risco_credito)

# Parametro 'B0'
print(logistic_risco_credito.intercept_)

# Coeficiente "B1, B2, B3, B4"
print(logistic_risco_credito.coef_)

'''
Teste
hist: boa (0), divida: alta (0), garantia: nehuma (1), renda: >35 (2)
hist: ruim (2), divida: alta (0), garantia: adequada (0), renda: <15 (0)
'''
previsoes1 = logistic_risco_credito.predict([[0,0,1,2],[2,0,0,0]])
print(previsoes1)