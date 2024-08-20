from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import pickle

# Adiciona os dados as respectivas variaveis (Ja com Label encoder)
with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(f)

# Criando um classificador de árvore de decisão usando a métrica de entropia
# A entropia é uma medida de impureza ou incerteza, utilizada para decidir como dividir os nós da árvore
arvore_credit = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

# O algoritmo vai ser treinado e a arvore de decisao gerada
arvore_credit.fit(X_credit_treinamento, Y_credit_treinamento)

# Algoritmo faz a previsão dos dados de teste
previsoes = arvore_credit.predict(X_credit_teste)
# print(previsoes)

# Importa e exibe a porcentagem de acerto (acurácia) do modelo treinado
from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(Y_credit_teste, previsoes))
print(classification_report(Y_credit_teste, previsoes))

from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
cm = ConfusionMatrix(arvore_credit)
cm.fit(X_credit_treinamento, Y_credit_treinamento)
cm.score(X_credit_teste, Y_credit_teste)
#plt.show()

from sklearn import tree
previsores = ['income', 'age', 'loan']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
# Cria a estrutura da arvore de decisão e exibe
tree.plot_tree(arvore_credit, feature_names= previsores, class_names=['0','1'], filled=True)
#plt.show()
fig.savefig('arvore_credit.png')

