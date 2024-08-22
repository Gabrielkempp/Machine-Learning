from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import pickle

with open('Data/credit.pkl', 'rb') as f:
    X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(f)

# Define a quantidade de arvores que serão usadas
random_forest_credit = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)

# Treina o algoritmo com os dados de teste
random_forest_credit.fit(X_credit_treinamento, Y_credit_treinamento)

# Testando o algoritmo com os dados de teste
previsoes = random_forest_credit.predict(X_credit_teste)

# Porcentagem de acertos é salvo na variavel
acertos = accuracy_score(Y_credit_teste, previsoes)
print(acertos)

# Relatorio sobre o algoritmo é gerado e salvo na variavel
classificacao = classification_report(Y_credit_teste, previsoes)
print(classificacao)

# Matriz de confusão
cm = ConfusionMatrix(random_forest_credit)
cm.fit(X_credit_treinamento, Y_credit_treinamento)
cm.score(X_credit_teste, Y_credit_teste)
plt.show()
