from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import pickle

# Adiciona os dados as respectivas variaveis (Ja com Label encoder)
with open("Data/risco_credito.pkl", 'rb') as f:
    X_risco_credito, Y_risco_credito = pickle.load(f)

# Criando um classificador de árvore de decisão usando a métrica de entropia
# A entropia é uma medida de impureza ou incerteza, utilizada para decidir como dividir os nós da árvore
arvore_risco_credito = DecisionTreeClassifier(criterion='entropy')

# O algoritmo vai ser treinado e a arvore de decisao gerada
arvore_risco_credito.fit(X_risco_credito, Y_risco_credito)

# Exibe o ganho de informação de cada item previsor
print(arvore_risco_credito.feature_importances_)

# Lista criada para nomear os previsores da arvore de decisão
previsores = ['historico','divida', 'garantias', 'renda']
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
# Cria a estrutura da arvore de decisão e exibe
tree.plot_tree(arvore_risco_credito, feature_names= previsores, class_names=arvore_risco_credito.classes_, filled=True) # class_names -> alto, baixo, moderado
plt.show()

'''
TESTE 

hist: boa (0), divida: alta (0), garantia: nehuma (1), renda: >35 (2)
hist: ruim (2), divida: alta (0), garantia: adequada (0), renda: <15 (0)
'''

previsoes = arvore_risco_credito.predict([[0, 0, 1, 2],[2, 0, 0, 0]])
print(previsoes)