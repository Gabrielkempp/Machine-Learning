from sklearn.naive_bayes import GaussianNB
from data_processing import X_risco_credito, Y_risco_credito

naive_risco_credito = GaussianNB()
naive_risco_credito.fit(X_risco_credito, Y_risco_credito) # quando for exegutado sera feito o treiamento do algoritmo

'''
hist: boa (0), divida: alta (0), garantia: nehuma (1), renda: >35 (2)
hist: ruim (2), divida: alta (0), garantia: adequada (0), renda: <15 (0)
'''

previsao = naive_risco_credito.predict([[0, 0, 1, 2],[2, 0, 0, 0]])

print(previsao)
print(naive_risco_credito.classes_)
print(naive_risco_credito.class_count_)
print(naive_risco_credito.class_prior_)