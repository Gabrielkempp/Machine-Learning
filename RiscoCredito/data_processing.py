import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Variavel recebe o arquivo csv
base_risco_credito = pd.read_csv('Data/risco_credito.csv')

#print(base_risco_credito)

# Armazenar os atributos previsores
X_risco_credito = base_risco_credito.iloc[:,0:4].values # : -> todas as linhas / 0 ao 3 / Values -> numpy array

# Armazenar os atributos de classe
Y_risco_credito = base_risco_credito.iloc[:,4].values

#print(X_risco_credito)
#print(Y_risco_credito)

from sklearn.preprocessing import LabelEncoder

# Criando variaveis para serem usadas na transformação dos dados
label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

# Transformando strings em numeros que as representem
X_risco_credito[:,0] = label_encoder_historia.fit_transform(X_risco_credito[:,0])
X_risco_credito[:,1] = label_encoder_divida.fit_transform(X_risco_credito[:,1])
X_risco_credito[:,2] = label_encoder_garantia.fit_transform(X_risco_credito[:,2])
X_risco_credito[:,3] = label_encoder_renda.fit_transform(X_risco_credito[:,3])

# print(X_risco_credito)

# Base de dados muito pequena para ser aplicado o OneHotEncoder

# Criando o arquivo
import pickle
with open('Data/risco_credito.pkl', 'wb') as f:
    pickle.dump([X_risco_credito,Y_risco_credito], f)

