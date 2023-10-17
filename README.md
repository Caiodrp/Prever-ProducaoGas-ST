# CREDIT SCORING

## Web aplicação que automatiza análises sobre dados de clientes que tomaram crédito, e faz previsão de novos dados usando a Regressão Logística para prever se novos clientes se tornarão inadimplentes ou não.

# Este é um tutorial em vídeo sobre como usar a aplicação




https://github.com/Caiodrp/Prever-Inadimplencia-ST/assets/99834159/1c979891-8f91-4a2e-9cef-1ba4ef91fcbf





# Modelos dos arquivos CSV
## Arquivo CSV clusterização
[Download Arquivo Online purchased](https://raw.githubusercontent.com/Caiodrp/Clusterizacao-Streamlit/main/CSV/online_shoppers_intention.csv)

## Arquivo CSV RFV
[Download Arquivo RFV](https://raw.githubusercontent.com/Caiodrp/Clusterizacao-Streamlit/main/CSV/exemplo_RFV.csv)

# Análise Clusterização

Na página "Análise Clusterização", após carregar o arquivo .CSV, você pode visualizar diferentes gráficos e informações sobre o conjunto de dados "online_shoppers_intention" que representa o comportamento de diversos acessos de usuários em diferentes tipos de sites, disponível em [https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset). 

**Info**

Na opção "Info", você encontrará informações sobre os dados, como o dicionário de dados, algumas linhas do dataframe e a opção de gerar um relatório completo sobre eles.

**Descritiva**

Na opção "Descritiva", você encontrará opções de visualizações gráficas entre as variáveis da base de dados e relações entre elas e o problema em questão.

## Clusterização

Após carregar o arquivo, essa página possibilita a realização da clusterização do comportamento de navegação dos acessos, utilizando o algoritmo de K-means ou algoritmos hierárquicos. É oferecido um filtro por variáveis para visualizar os grupos formados.

**K-Means**

Selecionando o "K-means", aparecerá a opção de visualizar, através do método do cotovelo ou da silhueta, sugestões de quantidades de grupos. Em seguida, você poderá definir quantos grupos deseja que o algoritmo divida. Será exibido um filtro para ver a distribuição das variáveis por grupo e a opção de baixar o Data Frame com a coluna de grupamento.

**Hierárquicos**

Selecionando "Hierárquicos", você poderá escolher o método de ligação de acordo com o estudo dos dados. Em seguida, digite a quantidade de clusters desejados (levando em consideração o poder computacional disponível). Será exibido um filtro e um botão de download semelhante ao do K-means.

# Análise Segmentação

Na página "Análise Segmentação", após carregar o arquivo .CSV, você pode visualizar diferentes gráficos e informações sobre o conjunto de dados "exemplo_RFV" que representa o comportamento de clientes em relação a compras, como tempo, quantidade e valor. Disponível em [https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset). 

# Segmentação

Após o carregamento do arquivo, será exibido um dataframe com cada cliente segmentado por Recência, Frequência e Valor em "A", "B", "C" e "D", sendo "A" o melhor nível e "D" o pior (conforme mostrado no notebook de dados). Também será exibido um botão para download do Data Frame segmentado.
