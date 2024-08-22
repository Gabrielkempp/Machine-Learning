import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

base_credit = pd.read_csv("Data/credit_data.csv")
'''
# Valores inconsistentes------------------------------------------------------------------------------------------------
print(base_credit.loc[base_credit['age'] < 0])
# Outra maneira de buscar os valores desejados:
# print(base_credit[base_credit['age'] < 0])
'''

'''
# Quando muitos dados estiverem inconsistentes (por exemplo 70%) é viavel apagar a coluna inteira.

# Copia base de dados e remove a coluna 'age'.
base_credit2 = base_credit.drop('age', axis= 1)
print(base_credit2)
'''

'''
# Apaga somente registros com valores inconsistentes.
base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)

# Verifica se os valores foram deletados corretamente
print(base_credit3.loc[base_credit3['age'] < 0])
'''

# O mais viavel a se fazer é preencher os valores inconsistentes com a média
age_mean = base_credit['age'][base_credit['age'] > 0].mean()
base_credit.loc[base_credit['age'] < 0, 'age'] = age_mean
# print(base_credit.loc[base_credit['age'] < 0])
# print(base_credit.head(27))

'''
# Exibe graficos passando Idade, Renda e Divida como parametro, e classifica por cor os pagantes e não pagantes 
da divida
grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income','loan'], color= 'default')
grafico.show()
'''

# Verifica se há valores nulos------------------------------------------------------------------------------------------
# print(base_credit.isnull().sum())

# Mostra quais os valores nulos
# print(base_credit.loc[pd.isnull(base_credit['age'])])

# Insere a média de idade onde a idade é nula. ('fill'-> preenche)('na' -> Valor nulo)
# CODIGO ANTIGO -> base_credit['age'].fillna(base_credit['age'].mean(),inplace = True)
base_credit['age'] = base_credit['age'].fillna(base_credit['age'].mean())

# print(base_credit.loc[pd.isnull(base_credit['age'])])
# print(base_credit.head(32))

# Visualizar linhas que tinham 'age' = 0
# print(base_credit.loc[(base_credit['clientid'] == 29) | (base_credit['clientid'] == 31) |
#                      (base_credit['clientid'] == 32)])
# print(base_credit.loc[base_credit['clientid'].isin([29, 31, 32])])

# Divisão entr previsores e classe -------------------------------------------------------------------------------------
X_credit = base_credit.iloc[:, 1:4].values # busca o atributo 1, 2 e 3 [todos os dados exceto a coluna 'default']
Y_credit = base_credit.iloc[:, 4].values # busca o atributo 4 [coluna 'default']
#type(X_credit) e type(X_credit) são "numpy.ndarray"
#print(Y_credit)

# Escalonamento dos valores---------------------------------------------------------------------------------------------
# "X_credit[:,0]" busca a primeira coluna inteira
minFormat = (f"A menor renda: ${X_credit[:,0].min():.2f} anualmente.\n"
      f"A menor idade: {X_credit[:,1].min():.0f} anos.\n"
      f"A menor divida: {X_credit[:,2].min():.2f}.")# "X_credit[:,0]" busca a primeira coluna inteira

maxFormat=(f"A maior renda: ${X_credit[:,0].max():,.2f} anualmente.\n"
      f"A maior idade: {X_credit[:,1].max():.0f} anos.\n"
      f"A maior divida: ${X_credit[:,2].max():.2f}")

# print(X_credit[:,0].min(), X_credit[:,1].min(), X_credit[:,2].min())
# print(X_credit[:,0].max(), X_credit[:,1].max(), X_credit[:,2].max())

'''
A padronização (Standardisation) e a normalização (normalization) 
são usadas em casos onde os valores são muito distantes 
(EX: idade 18 e divida 10.000) pois o algoritmo pode acabar 
se confundindo e achando que um desses valores é mais 
importante que outro.

Padronização é indicada quando há outliers na tabela, por exemplo idades negativas.
'''

from sklearn.preprocessing import StandardScaler
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)
# print(X_credit[:,0].min(), X_credit[:,1].min(), X_credit[:,2].min())
# print(X_credit[:,0].max(), X_credit[:,1].max(), X_credit[:,2].max())
# Valores Escalonados | Ultima etapa do preprocessamento de dados

# Salvando o DataFrame processado em um novo arquivo CSV
base_credit.to_csv("Data/credit_data_processed.csv", index=False)

# Divisão das bases em treinamento e teste -----------------------------------------------------------------------------
from sklearn.model_selection import train_test_split


# X_credit = Atributos previsores
# Y_credit = Classe
# test_sise > tamanho da base de dados de teste
X_credit_treinamento, X_credit_teste, Y_credit_treinamento, Y_credit_teste = train_test_split(X_credit, Y_credit, test_size = 0.25, random_state = 0)

'''print(X_credit_treinamento.shape)# 3 colunas Income - age - loan
print(Y_credit_treinamento.shape)# 1 coluna (classe 0 = pago | classe 1 = não pago)
print((X_credit_teste.shape,Y_credit_teste.shape))'''

# Salvar as variaveis --------------------------------------------------------------------------------------------------
import pickle
with open('Data/credit.pkl', mode ='wb') as f: # 'credit.pkl'-> nome do arquivo que sera criado, 'wb'-> w = write,, vamos escrever a variavel
      pickle.dump([X_credit_treinamento,Y_credit_treinamento, X_credit_teste, Y_credit_teste],f)