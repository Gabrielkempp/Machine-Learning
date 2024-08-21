print("Bem-vindos à loja do Daniel Henrique Kern")

valorDoPedido = float(input("Entre com valor do pedido: "))
quantidadeParcelas = int(input("Entre com a quantidade de parcelas: "))

if quantidadeParcelas == 1:
    juros = 0
elif quantidadeParcelas == 2:
    juros = 0.03
elif quantidadeParcelas == 3:
    juros = 0.05
else:
    juros = 0.07

valorDaParcela = (valorDoPedido * (1 + juros)) / quantidadeParcelas
valorTotalParcelado = valorDaParcela * quantidadeParcelas

print("Nome completo do desenvolvedor: Daniel Henrique Kern")

if quantidadeParcelas >= 4:
    print(f"Parcelamento em {quantidadeParcelas} vezes com juros de {juros * 100}% ao mês:")
    print(f"Valor da parcela: R${valorDaParcela:.2f}")
    print(f"Valor total parcelado: R${valorTotalParcelado:.2f}")
else:
    print(f"Valor total a ser pago à vista: R${valorTotalParcelado:.2f}")
