import Orange

# Atribui os dados à variavel
base_credit = Orange.data.Table('Data/credit_data_regras.csv')

# Divide a base de dados
base_dividida = Orange.evaluation.testing.sample(base_credit, n = 0.25)

# Separa a base de dados
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

# Cria as regras
cn2 = Orange.classification.rules.CN2Learner()
regras_credit = cn2(base_treinamento)

# Exibe as regras
for regras in regras_credit.rule_list:
    print(regras)

previsoes = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [lambda testdata: regras_credit])
#print(previsoes)

print(Orange.evaluation.CA(previsoes))