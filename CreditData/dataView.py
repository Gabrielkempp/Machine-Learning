import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

base_credit = pd.read_csv("Data/credit_data.csv")

count = np.unique(base_credit['default'], return_counts = True) # Contagem de valores unicos
'''
# Exibe detalhes
print(f'Valores unicos: {count[0]}\n'
      f'Contagem de cada valor:\n'
      f'   {count[0][0]}: {count[1][0]} pessoas que pagaram o emprestimo.\n'
      f'   {count[0][1]}: {count[1][1]} pessoas que não pagaram o emprestimo.')
'''


'''
# Exibe o grafico em barras
sns.countplot(x = base_credit['default'])
plt.show()
'''

'''
# Exibe histograma usando a idade como parametro
plt.hist(x=base_credit['age'])
plt.show()
'''

'''
# Exibe histograma usando a remuneração como parametro
plt.hist(x=base_credit['income'])
plt.show()
'''

'''
# Exibe histograma usando a divida como parametro
plt.hist(x=base_credit['loan'])
plt.show()
'''

# Exibe graficos passando Idade, Renda e Divida como parametro, e classifica por cor os pagantes e não pagantes da divida
grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income','loan'], color= 'default')
grafico.show()