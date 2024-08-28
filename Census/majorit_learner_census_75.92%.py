"""
Não se trata de um algoritmo que aprende com os dados, mas sim classifica
os regirstros com base na maioria dos dados existentes.

Se os algoritmos tiverem uma acurácia inferior à acuracia do "Algoritmo"
Majority Learner, então os registros novos deveraão ser classificados de
acordo com a classificação da maioria dos dados.
"""
import Orange

# Associa os dados à variavel
base_census = Orange.data.Table('Data/census_regras.csv')

print(base_census.domain)

# Associa o algoritmo à variavel
majority = Orange.classification.MajorityLearner()

# Testando os dados com os proprios dados de treinamento
previsoes = Orange.evaluation.testing.TestOnTestData(base_census, base_census, [majority])

# Exibe a acuracia
print(Orange.evaluation.CA(previsoes))

# Contagem de registros
from collections import Counter
print(Counter(str(registro.get_class()) for registro in base_census))

