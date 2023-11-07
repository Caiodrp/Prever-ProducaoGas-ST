import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import plotly.express as px
import base64
import plotly.graph_objects as go
import chardet

from scipy import stats
from sklearn.ensemble import IsolationForest
from scipy.stats import kruskal
from ydata_profiling import ProfileReport
from scipy.stats import t
from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Definir o template
st.set_page_config(page_title='Análises',
                    page_icon='🏭',
                    layout='wide')

@st.cache_data
def carregar_dados(uploaded_file):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        try:
            if file_extension == "csv":
                # Se a extensão for .CSV, leia o arquivo como CSV
                return pd.read_csv(uploaded_file)
            else:
                st.warning("Extensão de arquivo não suportada. Por favor, faça upload de um arquivo .CSV")
                return None
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {str(e)}")
            return None

    st.warning("Por favor, faça upload de um arquivo .CSV")
    return None

def calcula_teste_h_agrupa_mediana(df, resposta='Gás Natural (Mm³/dia)', variavel='Operador', condicao=0.2):
    var_exp = variavel + '_Agrupado'
    # Cria um dicionário para armazenar os resultados
    resultados = {}

    # Loop através de cada categoria na variável
    for categoria in df[variavel].unique():
        # Cria uma nova coluna binária para a categoria
        df_temp = df.copy()
        df_temp[categoria] = np.where(df_temp[variavel] == categoria, 1, 0)

        # Seleciona os valores da variável resposta para esta categoria e para o resto
        grupo_categoria = df_temp[df_temp[categoria] == 1][resposta]
        grupo_resto = df_temp[df_temp[categoria] == 0][resposta]

        # Realiza o Teste H de Kruskal-Wallis
        h, p = kruskal(grupo_categoria, grupo_resto)

        # Calcula a mediana da variável resposta para esta categoria
        mediana = "{:.5f}".format(grupo_categoria.median())

        # Adiciona os resultados ao dicionário
        resultados[categoria] = (h, p, mediana)

    # Cria um DataFrame com as categorias agrupadas
    df_resultados = pd.DataFrame(resultados, index=['H', 'p-value', 'Mediana']).transpose()

    # Inicializa o DataFrame para armazenar os agrupamentos
    df_agrupamentos = pd.DataFrame(columns=['H', 'p-value', 'Mediana'])
    grupo_atual = []
    valor_Mediana_anterior = float(df_resultados.iloc[0]['Mediana'])

    # Loop através de todas as linhas do DataFrame
    for indice, linha in df_resultados.iterrows():
        # Converte a 'Mediana' de volta para float
        mediana_atual = float(linha['Mediana'])

        # Verifica se a diferença entre os valores atuais e anteriores de 'Mediana' é menor ou igual a condição
        if abs(mediana_atual - valor_Mediana_anterior) <= condicao:
            # Adiciona o índice ao grupo atual
            grupo_atual.append(indice)
        else:
            # Adiciona o grupo atual ao DataFrame de agrupamentos
            chave = '_'.join(map(str, grupo_atual))
            df_agrupamentos.loc[chave] = [df_resultados.loc[grupo_atual[0], 'H'], df_resultados.loc[grupo_atual[0], 'p-value'], valor_Mediana_anterior]

            # Inicia um novo grupo com o índice atual
            grupo_atual = [indice]

        # Atualiza o valor anterior de 'Mediana'
        valor_Mediana_anterior = mediana_atual

    # Adiciona o último grupo ao DataFrame de agrupamentos
    chave = '_'.join(map(str, grupo_atual))
    df_agrupamentos.loc[chave] = [df_resultados.loc[grupo_atual[0], 'H'], df_resultados.loc[grupo_atual[0], 'p-value'], valor_Mediana_anterior]

    # Cria um dicionário para mapear as categorias originais para as novas categorias agrupadas
    mapeamento = {indice: chave for chave in df_agrupamentos.index for indice in chave.split('_')}

    # Cria uma nova coluna no DataFrame original que mapeia a coluna var_exp para as novas categorias agrupadas
    df[var_exp] = df[variavel].map(mapeamento)

    return df

def remove_outliers(df):
    # Crie o modelo Isolation Forest
    clf = IsolationForest(contamination= 0.2, random_state=42)
    
    # vars de interesse
    vars_conti = ['Tempo de Produção (hs por mês)', 'Petróleo (bbl/dia)', 'Gás Natural (Mm³/dia)',
              'Água (bbl/dia)', 'Grau API']
    
    # Selecione apenas as colunas numéricas do DataFrame
    df_numeric = df[vars_conti]
    
    # Ajuste o modelo aos seus dados
    clf.fit(df_numeric)
    
    # Use o modelo para prever outliers
    outlier_predictions = clf.predict(df_numeric)
    
    # Remova os outliers do DataFrame
    df_no_outliers = df[outlier_predictions == 1]
    
    return df_no_outliers

def categoriza_grau_api(grau_api):
    if grau_api > 45:
        return 'Leve_Particular'
    elif 33 <= grau_api <= 45:
        return 'Leve'
    elif 22 <= grau_api < 33:
        return 'Medio'
    elif 10 <= grau_api < 22:
        return 'Pesado'
    else:
        return 'Extra_Pesado'

def transformacao_dados(df):
    # Cria um mapa dos valores de 'Nome Poço ANP' para 'Corrente' e 'Grau API'
    mapa_corrente = df.dropna(subset=['Corrente']).set_index('Nome Poço ANP')['Corrente'].to_dict()
    mapa_grau_api = df.dropna(subset=['Grau API']).set_index('Nome Poço ANP')['Grau API'].to_dict()

    # Preencha os valores faltantes nas colunas 'Corrente' e 'Grau API' com os valores correspondentes do mapa
    df['Corrente'] = df['Corrente'].fillna(df['Nome Poço ANP'].map(mapa_corrente))
    df['Grau API'] = df['Grau API'].fillna(df['Nome Poço ANP'].map(mapa_grau_api))

    # Exclua as linhas que ainda têm valores NaN
    df = df.dropna()

    # Retirando colunas que não tem haver com a produção, redundantes ou não significativas.
    cols_to_drop = ['Nome Poço Operador','Nome Poço ANP','Número do Contrato', 'Período','Condensado (bbl/dia)',
                    'Gás Natural (Mm³/dia) N Assoc', 'Gás Natural (Mm³/dia) Total', 'Volume Gás Royalties (m³/mês)',
                    'Instalação Destino', 'Tipo Instalação', 'Período da Carga','Óleo (bbl/dia)','Campo']
    df = df.drop(columns=cols_to_drop)

    df = df[~((df[['Tempo de Produção (hs por mês)', 'Petróleo (bbl/dia)', 'Água (bbl/dia)']] < 1).any(axis=1) |
               (df['Gás Natural (Mm³/dia)'] <= 0))]

    # Corrigindo nome dos estados
    df['Estado'] = df['Estado'].replace({
        'EspÃ­rito Santo': 'Espírito Santo',
        'CearÃ¡': 'Ceará',
        'SÃ£o Paulo': 'São Paulo',
        'MaranhÃ£o': 'Maranhão'
    })

    # Unir Estado e Bacia
    df['Estados_Bacias'] = df['Estado'] + "_" + df['Bacia']

    correcoes_operador = {
        '3R Fazenda BelÃ©m': '3R Fazenda Belém',
        'Seacrest SPE CricarÃ©': 'Seacrest SPE Cricaré',
        'NÃ­on Energia': 'Níon Energia',
        'Nova PetrÃ³leo': 'Nova Petróleo',
        'RecÃ´ncavo E&P': 'Recôncavo E&P',
        'Phoenix Ãleo & GÃ¡s': 'Phoenix Óleo & Gás',
        'Petro Rio O&G': 'Petro Rio O&G',
        'SPE TiÃªta': 'SPE Tiêta',
        'PetroRecÃ´ncavo': 'PetroRecôncavo'
    }

    # Use o método replace para corrigir os nomes das empresas
    df['Operador'] = df['Operador'].replace(correcoes_operador)

    # Removendo categorias com pouca representatividade
    df = df[~df['Operador'].isin(df['Operador'].value_counts()[df['Operador'].value_counts() < 10].index)]

    correcoes_corrente = {
        'EspÃ­rito Santo': 'Espírito Santo',
        'Fazenda BelÃ©m': 'Fazenda Belém',
        'Ãrea de Sul de Tupi': 'Área de Sul de Tupi',
        'SapinhoÃ¡': 'Sapinhoá',
        'SabiÃ¡ Bico de Osso': 'Sabiá Bico de Osso',
        'Fazenda Santo EstevÃ£o': 'Fazenda Santo Estevão',
        'SabiÃ¡ da Mata': 'Sabiá da Mata'
    }

    # Use o método replace para corrigir os nomes das correntes
    df['Corrente'] = df['Corrente'].replace(correcoes_corrente)

    # Removendo categorias com pouca representatividade
    df = df[~df['Corrente'].isin(df['Corrente'].value_counts()[df['Corrente'].value_counts() < 10].index)]

    # Agrupando variáveis categóricas
    df = calcula_teste_h_agrupa_mediana(df)

    df = remove_outliers(df)

    # Lista de colunas para transformar
    cols_to_transform = ['Petróleo (bbl/dia)', 'Água (bbl/dia)', 'Tempo de Produção (hs por mês)', 'Gás Natural (Mm³/dia)']

    # Crie novas colunas com o sufixo '_log' para as transformações
    for col in cols_to_transform:
        df[col + '_log'] = np.log(df[col])

    # Crie uma nova coluna com o sufixo '_sqrt' para a transformação de raiz quadrada
    df['Água (bbl/dia)_sqrt'] = np.sqrt(df['Água (bbl/dia)'])
    df['Tempo de Produção (hs por mês)_sqrt'] = np.sqrt(df['Tempo de Produção (hs por mês)'])

    # Funçãoq ue categoriza o API
    df['Grau_API_Cat'] = df['Grau API'].apply(categoriza_grau_api)

    return df

@st.cache_data
def describe_continuous(df):
    # Alterar a formatação global de exibição de float
    pd.options.display.float_format = '{:.2f}'.format

    # Selecionar variáveis contínuas
    continuous_vars = df.select_dtypes(include=['int64','int32','float64'])

    # Obter estatísticas descritivas
    desc = continuous_vars.describe()

    # Adicionar uma linha para a quantidade de valores únicos
    unique_counts = pd.DataFrame(continuous_vars.nunique(), columns=['unique']).transpose()
    desc = pd.concat([desc, unique_counts])

    return desc

@st.cache_data
def describe_categorical(df):
    categorical_vars = df.select_dtypes(include=['object', 'bool'])
    return categorical_vars.describe()

@st.cache_data
def plot_cont_bivariate(df, vars_cont):
    # Se nenhuma variável foi selecionada
    if not vars_cont:
        return

    # Se apenas uma variável foi selecionada
    if len(vars_cont) == 1:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        sns.scatterplot(data=df, x=vars_cont[0], y='Gás Natural (Mm³/dia)_log', ax=ax)
        st.pyplot(fig)
    else:
        # Criando uma figura com 2 subplots em cada linha e coluna
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Achatando o array axs para facilitar a iteração
        axs = axs.flatten()

        # Loop através das variáveis
        for i, var in enumerate(vars_cont):
            # Removendo os subplots vazios
            if i > len(vars_cont) - 1:
                fig.delaxes(axs[i])
            else:
                # Criando um gráfico de dispersão para cada variável usando Seaborn
                sns.scatterplot(data=df, x=var, y='Gás Natural (Mm³/dia)_log', ax=axs[i])

        plt.tight_layout()
        st.pyplot(fig)

@st.cache_data
def plot_cat_bivariate(df, vars_cat):
    # Loop através das variáveis
    for var in vars_cat:
        # Cria uma figura vazia
        fig = go.Figure()

        # Obtém a lista de categorias dentro da variável
        categorias = df[var].unique()

        # Gere cores aleatórias para cada categoria usando o NumPy
        paleta_cores = {categoria: f'rgb({int(np.random.rand() * 256)}, {int(np.random.rand() * 256)}, {int(np.random.rand() * 256)})' for categoria in categorias}

        # Inicializa uma lista para os rótulos do eixo x
        labels_x = []

        # Calcula a média da variável dependente para cada categoria
        for cat in categorias:
            media = df[df[var] == cat]['Gás Natural (Mm³/dia)'].mean()
            # Adiciona as barras ao gráfico com cores diferentes para cada categoria
            fig.add_trace(go.Bar(x=[cat], y=[media], name=cat, marker_color=paleta_cores[cat]))
            # Adiciona o rótulo da categoria ao eixo x
            labels_x.append(cat)

        # Atualiza o layout do gráfico para incluir os rótulos no eixo x
        fig.update_layout(title_text=f'Média de Gás por {var}', xaxis_title=var, yaxis_title='Média da Variável Dependente', xaxis={'type': 'category', 'categoryorder': 'array', 'categoryarray': labels_x})

        # Mostra o gráfico
        st.plotly_chart(fig)

@st.cache_data
def plot_correlation_matrix(df, vars):
    # Selecione apenas as colunas especificadas
    df_selected = df[vars]

    # Calculando a matriz de correlação de Spearman
    correlation_matrix = df_selected.corr(method='spearman')

    # Plotando a matriz de correlação como um mapa de calor
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Matriz de Correlação de Spearman')

    st.pyplot(fig)

@st.cache_data
def normalidade(df, vars):
    # Se nenhuma variável foi selecionada
    if not vars:
        return

    # Determinando o número de linhas e colunas para os subplots
    n = len(vars)
    ncols = 3
    nrows = n // ncols + (n % ncols > 0)

    # Criando uma figura com subplots suficientes para todas as variáveis
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, nrows*5))
    axs = axs.flatten()  # para facilitar a iteração

    # Loop através das variáveis
    for i, var in enumerate(vars):
        # para plotar
        data = df[var]

        # Criando um gráfico de distribuição para cada variável usando Seaborn
        sns.distplot(data, ax=axs[i])

        # Realizando o teste de Kolmogorov-Smirnov para a variável atual
        ks_test = stats.kstest(data, 'norm')

        # Adicionando o resultado do teste ao gráfico
        axs[i].annotate(f"D={ks_test.statistic:.4f}", xy=(0.6, 0.8), xycoords='axes fraction')

    # Removendo os gráficos extras
    for i in range(n, nrows*ncols):
        fig.delaxes(axs[i])

    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data
def calculate_vif(df, vars):
    # Selecione apenas as colunas especificadas
    df_numerics = df[vars]

    # Adicione uma constante ao dataframe
    df_numerics = add_constant(df_numerics)

    # Calcule o VIF para cada variável
    vars_vif = pd.DataFrame()
    vars_vif["VIF Factor"] = [variance_inflation_factor(df_numerics.values, i) for i in range(df_numerics.shape[1])]
    vars_vif["Feature"] = df_numerics.columns

    return vars_vif.round(2)

def main():  
    # Título centralizado
    st.markdown('<div style="display:flex; align-items:center; justify-content:center;"><h1 style="font-size:4.5rem;">Análises</h1></div>',
                unsafe_allow_html=True)

    # Divisão
    st.write("---")

    # Carregar dados
    uploaded_file = st.sidebar.file_uploader(
        "Faça upload do arquivo CSV",
        type=["csv"]
    )

    df = carregar_dados(uploaded_file)

    df = transformacao_dados(df)

    selecao_dados = st.sidebar.selectbox(
            "Selecione uma opção",
            ("Info", "Descritivas", "Suposições Modelo")
        )

    if selecao_dados == "Info":
            # Mostrar título
            st.header("Dicionário de dados:")

            # Mostrar imagem
            st.image("https://raw.githubusercontent.com/Caiodrp/Prever-Inadimplencia-St/main/dic_dados.png")

            # Mostrar cabeçalho do DataFrame
            st.dataframe(df.head())

            # Adicionar botão para gerar relatório HTML
            if st.button("Gerar relatório"):
                # Gerar relatório HTML usando o ydata-profiling
                profile = ProfileReport(df)
                html = profile.to_html()

                # Exibir relatório HTML na página do Streamlit
                components.html(html, width=900, height=500, scrolling=True)

    elif selecao_dados == "Descritivas":
        # Mostrar título
        st.header("Descritivas:")
        # Adicionar caixa de seleção na barra lateral
        selecao_desc = st.sidebar.selectbox(
            "Selecione uma opção",
            ("Contínuas", "Categóricas")
        )
        if selecao_desc== "Contínuas":
            # Filtrar as variáveis contínuas
            vars_cont = df.select_dtypes(['int64', 'float64']).columns.tolist()

            # Exibir o DataFrame descritivo
            st.dataframe(describe_continuous(df))

            # Adicionar um widget multiselect para selecionar as variáveis a serem observadas
            variaveis = st.multiselect("Selecione as variáveis", vars_cont)

            # Plotar os gráficos para cada variável contínua
            plot_cont_bivariate(df, variaveis)
        else:
            # Filtrar as variáveis categóricas
            vars_cat = df.select_dtypes(['object']).columns.tolist()

            # Exibir o DataFrame descritivo
            st.dataframe(describe_categorical(df))

            # Adicionar um widget multiselect para selecionar as variáveis a serem observadas
            variaveis = st.multiselect("Selecione as variáveis", vars_cat)

            # Plotar os gráficos para cada variável categórica
            plot_cat_bivariate(df, variaveis)

    else:
        # Mostrar título
        st.header("Suposições")
        # Adicionar caixa de seleção na barra lateral
        selecao_suposicoes = st.sidebar.selectbox(
            "Selecione uma opção",
            ("Normalidade/outliers", "Correlação/VIF")
        )
        # Adicionar um widget multiselect para selecionar as variáveis a serem observadas
        variaveis = st.multiselect("Selecione as variáveis", df.columns.tolist())

        if selecao_suposicoes == "Normalidade/outliers":
            normalidade(df, variaveis)
        else:  # Correlação/VIF
            st.dataframe(calculate_vif(df, variaveis))
            plot_correlation_matrix(df, variaveis)

if __name__ == "__main__":
    main()
