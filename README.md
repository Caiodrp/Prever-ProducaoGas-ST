# PRODUÇÃO DE GÁS 

## Este é um tutorial em vídeo sobre como usar a aplicação


https://github.com/Caiodrp/Prever-ProducaoGas-ST/assets/99834159/6085971e-41c2-4eeb-8c7a-1f0c3995451d


## Coleta de Dados

Os dados foram capturados e transformados a partir de fontes confiáveis. Os arquivos brutos são encontrados no site de dados abertos do governo: https://dados.gov.br/dados/conjuntos-dados/producao-de-petroleo-e-gas-natural-por-poco.

Já a base trasnformada e agrupada por semestre e localidade se encontra qui no diretório: https://github.com/Caiodrp/Prever-ProducaoGas-ST/blob/main/df_todos.csv

## Modelagem

A modelagem é realizada por um algoritmo de regressão linear com regularização Ridge. O objetivo é encontrar a produção média diária de gás a partir de características físicas, químicas e de produção dos poços.

## Conclusões

Os resultados nos ajudam a estimar a produção média diária de gás natural de acordo com os Estados e Bacias dos poços, o Operador dos poços, o tipo do Grau API do poço, além da quantidade de Petróleo extraída, assim como o tempo de produção do poço e a quantidade de Água utilizada por dia na produção. Além disso, o relatório gerencial mostra o quanto cada variável aumenta ou diminui no resultado da produção de gás.
