import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Carregando base de dados para variavel 'base_census'
base_census = pd.read_csv('Data/census.csv')

'''
# Configurações de exibição
pd.set_option('display.max_columns', None)
print(base_census.describe())

# Verifica se há Valors nulos 
print(base_census.isnull().sum())
'''