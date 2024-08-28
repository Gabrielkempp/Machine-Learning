"""
Não se trata de um algoritmo que aprende com os dados, mas sim classifica
os regirstros com base na maioria dos dados existentes.

Se os algoritmos tiverem uma acurácia inferior à acuracia do "Algoritmo"
Majority Learner, então os registros novos deveraão ser classificados de
acordo com a classificação da maioria dos dados.
"""
import Orange

# Abre o arquivo e associa os dados à variavel [Dados para orange tem formato diferente, por isso o arquivo é outro]
base_credit = Orange.data.Table('Data/credit_data_regras.csv')

print(base_credit.domain)

# Associa o algoritmo à variavel
majority = Orange.classification.MajorityLearner()

# Testando os dados com os proprios dados de treinamento
previsoes = Orange.evaluation.testing.TestOnTestData(base_credit, base_credit, [majority])

# Exibe a acuracia
print(Orange.evaluation.CA(previsoes))

# Contagem de registros
from collections import Counter
print(Counter(str(registro.get_class()) for registro in base_credit))
