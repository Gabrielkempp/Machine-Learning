import Orange

# Abre o CSV adaptado para o Orange
base_risco_credito = Orange.data.Table('Data/risco_credito_regras.csv')

# Exibe informações
#print(base_risco_credito.domain)
#print(base_risco_credito)

# Treinamento do algoritmo
cn2 = Orange.classification.rules.CN2Learner()
regras_risco_credito = cn2(base_risco_credito)

# Exibe as regras
for regras in regras_risco_credito.rule_list:
    print(regras)

'''
Teste
hist: boa (0), divida: alta (0), garantia: nehuma (1), renda: >35 (2)
hist: ruim (2), divida: alta (0), garantia: adequada (0), renda: <15 (0)
'''
previsoes = regras_risco_credito([['boa', 'alta', 'nenhuma','acima_35'],['ruim', 'alta', 'adequada', '0_15']])
print(previsoes)

print(base_risco_credito.domain.class_var, base_risco_credito.domain.class_var.values)

for i in previsoes:
   print(base_risco_credito.domain.class_var.values[i])