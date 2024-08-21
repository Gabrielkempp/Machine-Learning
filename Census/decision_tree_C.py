from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

# Carregando as variaveis salvas
with open('census.pkl','rb') as f:
    X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(f)

# Define a arvore de decisão e o random state para 0, isso exibe o mesmo resultado sempre
arvore_census = DecisionTreeClassifier(criterion='entropy', random_state=0)

# Treina o algoitmo com os dados de treinamento
arvore_census.fit(X_census_treinamento,Y_census_treinamento)

# Dados de teste são passados no algoritmo e o resultado é salvo na variavel
previsoes = arvore_census.predict(X_census_teste)

# Porcentagem de acertos é salvo na variavel
acertos = accuracy_score(Y_census_teste, previsoes)
print(acertos)

# Relatorio sobre o algoritmo é gerado e salvo na variavel
classificação = classification_report(Y_census_teste, previsoes)
print(classificação)

# Matriz de confusão
cm = ConfusionMatrix(arvore_census)
cm.fit(X_census_treinamento,Y_census_treinamento)
cm.score(X_census_teste,Y_census_teste)
plt.show()



