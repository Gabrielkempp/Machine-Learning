import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

base_credit = pd.read_csv("Data/credit_data.csv")

'''

print(base_credit) # defalted
print(base_credit.head(2)) # Busca os Primeiros registros na base de dados
print(base_credit.tail(2)) # Busca os Ultimos registro na base de dados
print(base_credit.describe()) # Faz uma descrição completa da tabela
print(base_credit[base_credit["income"] >= 69995.685578 ]) # Mostra quem é que possui essa receita
print(base_credit[base_credit["loan"] <= 1.40 ]) # Mostra quem que possui esta divida

'''
print(base_credit[base_credit["loan"] <= 1.40 ])