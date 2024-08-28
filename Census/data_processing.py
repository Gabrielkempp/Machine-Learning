import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Carregando base de dados para variavel 'base_census'
base_census = pd.read_csv('Data/census.csv')


# Configurações de exibição
'''pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(base_census.info())
print(base_census.describe())'''

# Verifica se há Valors nulos 
'''print(base_census.isnull().sum())'''

# Visualização dos dados -----------------------------------------------------------------------------------------------
# Contagem de quantos registos á em cada uma das classes
'''print(np.unique(base_census['income'], return_counts= True))# Há 2 valores "< 50k"[24.720] e "> 50k[7.841]"
sns.countplot(x = base_census['income'])
plt.show()# Exibe o grafico'''

# Grafico em barras exibindo a idade, maior parte das pessas tem entre 18 e 45 anos aproximadamente
'''plt.hist(x = base_census['age'])
plt.show()'''

# Grafico em barras exibindo tempo de estudo
'''plt.hist(x = base_census['education-num'])
plt.show()'''

#  Grafico em barras exibindo tempo de estudo semanal
'''plt.hist(x = base_census['hour-per-week'])
plt.show()'''

# Geração de graficos dinamicos que agrupa os dados
'''grafico = px.treemap(base_census, path=['occupation', 'relationship', 'age'])
grafico.show()'''

# Grafico de categorias paralelas *****
'''grafico = px.parallel_categories(base_census, dimensions=['workclass','occupation', 'relationship', 'income'])
grafico.show()'''

# Divisão entre Previsores e Classe-------------------------------------------------------------------------------------
'''Informações:
- X_census -> Representa os Previsores
- Y_census -> Representa Classe
-----------------------------------------
Primeiro deve ser trasnformado de string para numeros (LabelEncoder - Categorico para numero simples) e depois deve ser 
aplicado para OneHotEncoder   
'''
X_census = base_census.iloc[:,0:14].values# ':' -> Todas as linhas | 0:14 inclui do 0 ao 13 | ".values" -> numpy
Y_census = base_census.iloc[:,14].values# Somente o 14 -> ['income']

# print(X_census)

# Tratamento de atributos categóricos | LabelEncoder -> (desvantagem se houver muitas categorias, o algoritmo entende
# que os numeros maiores são mais importantes)
from sklearn.preprocessing import LabelEncoder # Transformar strings em numeros

# Criando variaveis
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital= LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_countr = LabelEncoder()

# Transformando Strings em Numeros na tabela
X_census[:,1] = label_encoder_workclass.fit_transform(X_census[:,1])
X_census[:,3] = label_encoder_education.fit_transform(X_census[:,3])
X_census[:,5] = label_encoder_marital.fit_transform(X_census[:,5])
X_census[:,6] = label_encoder_occupation.fit_transform(X_census[:,6])
X_census[:,7] = label_encoder_relationship.fit_transform(X_census[:,7])
X_census[:,8] = label_encoder_race.fit_transform(X_census[:,8])
X_census[:,9] = label_encoder_sex.fit_transform(X_census[:,9])
X_census[:,13] = label_encoder_countr.fit_transform(X_census[:,13])

# Agora a base de dados esta quase pronta para os algoritmos de aprendizagem de maquina
# print(X_census)

# Tratamento de atributos categóricos | OneHotEncoder------------------------------------

'''
Expllicação do porque usar o OneHotEncoder:
LabelEncoder:
Carro: gol-3 palio-2 uno1
coluna carro:
2
3
1
O algoritmo vai processar o gol (3 - maior numero) como mais importante

OneHotEncoder:
Carro: gol-0,0,1 palio-0,1,0 uno-1,0,0
coluna carro:
0,1,0
1,0,0
0,0,1 
'''

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Definição do ColumnTransformer para aplicar OneHotEncoding
onehotencoder_census = ColumnTransformer(transformers=[
    # Nome da transformação ('OneHot'), aplicação de OneHotEncoder, e colunas alvo para a transformação
    ('OneHot', OneHotEncoder(),[1,3,5,6,7,8,9,13])
    ],
    remainder='passthrough')# passthrough -> faz com que não sejam apagadas as colunas que não foram mencionadas ('2,4,10,11,12')

# Aplicação do ColumnTransformer no conjunto de dados X_census
# fit_transform ajusta o transformador aos dados e aplica a transformação
X_census = onehotencoder_census.fit_transform(X_census).toarray()

#print(X_census[0])

# Mostra o tamanho (Linhas x Colunas)
#print(X_census.shape)

# Escalonamento dos valores --------------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)

# print(X_census[0])

# Divisão das bases em treinamento e teste -----------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

# X_census = Atributos previsores
# Y_census = Classe
# test_sise > tamanho da base de dados de teste
X_census_treinamento, X_census_teste, Y_census_treinamento, Y_census_teste = train_test_split(X_census, Y_census, test_size = 0.15, random_state = 0)

'''print(X_census_treinamento.shape)# 108 colunas
print(Y_census_treinamento.shape)# classes -> maior que 50k ou menor que 50k
print((X_census_teste.shape,Y_census_teste.shape))'''

# Salvar as variaveis --------------------------------------------------------------------------------------------------
import pickle
with open('Data/census.pkl', mode ='wb') as f:
    pickle.dump([X_census_treinamento,Y_census_treinamento, X_census_teste, Y_census_teste],f)