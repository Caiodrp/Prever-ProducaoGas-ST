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

from ydata_profiling import ProfileReport
from scipy.stats import t

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

@st.cache_data
def calcula_iv(df):
    """
    Calcula o IV (Information Value) com suavização para variáveis categóricas e contínuas.
    
    Parâmetros:
        df (DataFrame): DataFrame original com os dados.
    
    Retorna:
        metadados (DataFrame): DataFrame atualizado com os valores de IV calculados.
    """
    
    # Função para calcular o IV suavizado
    def IV(variavel, resposta):
        # Cria uma tabela de contingência entre a variável e a resposta
        tab = pd.crosstab(variavel, resposta, margins=True, margins_name='total')

        # Obtém os rótulos das colunas de evento e não evento
        rotulo_evento = tab.columns[0]
        rotulo_nao_evento = tab.columns[1]

        # Rótulos
        event = tab[rotulo_evento]
        non_event = tab[rotulo_nao_evento] 
        
        # Calcula as proporções de evento e não evento
        tab['pct_evento'] = event / tab.loc['total', rotulo_evento]
        tab['pct_nao_evento'] = non_event / tab.loc['total', rotulo_nao_evento]

        # Calcula o WoE (Weight of Evidence) e o IV parcial
        tab['woe'] = np.log(tab.pct_evento / tab.pct_nao_evento)
        tab['iv_parcial'] = (tab.pct_evento - tab.pct_nao_evento) * tab.woe

        # Retorna o IV parcial totalizado
        return tab['iv_parcial'].sum()

    # metadados para análise das variáveis
    metadados = pd.DataFrame(df.dtypes, columns=['dtype'])

    # Valores missing
    metadados['nmissing'] = df.isna().sum()

    # Categorias
    metadados['valores_unicos'] = df.nunique()

    # Adicionando a coluna 'papel' ao DataFrame 'metadados' e inicializando com o valor 'covariavel'
    metadados['papel'] = 'covariavel'

    # Alterando o valor da coluna 'papel' para 'resposta' na linha correspondente à coluna 'mau'
    metadados.loc['mau', 'papel'] = 'resposta'

    # Transformar a variável resposta em inteiro
    df['mau'] = df.mau.astype('int64')

    # Iterar sobre todas as variáveis categóricas e contínuas
    for var in metadados[metadados.papel == 'covariavel'].index:
        if metadados.loc[var, 'dtype'] in ['int', 'float']:
            # Se a variável for contínua, aplicar qcut se tiver mais de 5 categorias únicas
            if metadados.loc[var, 'valores_unicos'] > 5:
                variavel = pd.qcut(df[var], 5, duplicates='drop')
            else:
                variavel = df[var]
            iv = IV(variavel, df['mau'])
        else:
            # Se a variável for categórica, calcular o IV diretamente
            iv = IV(df[var], df['mau'])
        metadados.loc[var, 'IV'] = iv
    
    return metadados

@st.cache_data
def plot_cont_bivariate(df, column, q):
    # Criar uma cópia do dataframe
    df_copy = df.copy()

    # Criar bins usando qcut
    df_copy['bins'] = pd.qcut(df_copy[column], q=q, duplicates='drop')

    # Contar a frequência de cada bin para os valores True e False da variável 'mau'
    count_true = df_copy[df_copy['mau'] == True]['bins'].value_counts().sort_index()
    count_false = df_copy[df_copy['mau'] == False]['bins'].value_counts().sort_index()

    # Calcular a distribuição total e a proporção por True
    total = count_true + count_false
    proportion_true = count_true / total

    # Criar um DataFrame para o gráfico
    data = {
        'Bin': count_false.index.astype(str),
        'Distribuição': count_false.values,
        'Proporção mau': proportion_true.values
    }
    df_plot = pd.DataFrame(data)

    # Criar o gráfico de barras empilhadas usando Plotly Express
    fig = px.bar(df_plot, x='Bin', y=['Distribuição', 'Proporção mau'],
                title='Histograma de ' + column, labels={'value': 'Frequência'},
                height=600, color_discrete_sequence=['blue', 'red'])

    # Mostrar o gráfico no Streamlit
    st.plotly_chart(fig)

@st.cache_data
def plot_cat_bivariate(df, column):
    # Arredondar a coluna se os valores forem numéricos
    if np.issubdtype(df[column].dtype, np.number):
        df[column] = df[column].round()

    # Contar a frequência de cada categoria para os valores True e False da variável 'mau'
    count_true = df[df['mau'] == True][column].value_counts().sort_index()
    count_false = df[df['mau'] == False][column].value_counts().sort_index()

    # Garantir que ambos os índices tenham o mesmo conjunto de valores
    index = count_true.index.union(count_false.index)
    count_true = count_true.reindex(index, fill_value=0)
    count_false = count_false.reindex(index, fill_value=0)

    # Calcular a distribuição total e a proporção por True
    total = count_true + count_false
    proportion_true = count_true / total

    # Criar o gráfico de barras empilhadas
    fig = go.Figure(data=[
        go.Bar(name='Distribuição', x=total.index.astype(str), y=total.values, marker_color='blue'),
        go.Bar(name='Proporção mau', x=proportion_true.index.astype(str), y=proportion_true.values, marker_color='red')
    ])

    # Alterar o layout do gráfico
    fig.update_layout(
        barmode='stack',
        title_text='Histograma de ' + column,
        xaxis_title=column,
        yaxis_title='Frequência',
        autosize=False,
        width=1000,
        height=600,
    )

    # Mostrar o gráfico no Streamlit
    st.plotly_chart(fig)

@st.cache_data
def woe_discreta(var, df):
    
    """
    Calcula o Weight of Evidence (WOE) e outras métricas para uma variável categórica em relação à variável resposta.

    Parâmetros:
    - var: Nome da variável categórica no DataFrame 'df' para análise.
    - df: DataFrame pandas contendo as variáveis de interesse, incluindo a variável resposta 'mau'.

    Retorna:
    - biv: DataFrame contendo as métricas calculadas, incluindo WOE, limites de confiança e mais.
    
    Esta função calcula o Weight of Evidence (WOE) para uma variável categórica em relação à variável resposta 'mau' em um DataFrame 'df'. Ela também calcula outras métricas estatísticas, como proporção de 'mau', erros padrão, logit, limites de confiança e mais. Além disso, gera um gráfico mostrando o WOE para cada categoria da variável categórica.

    """
    
    # Cria uma nova coluna 'bom' no DataFrame 'df' que é igual a 1 menos o valor da coluna 'mau'
    df['bom'] = 1-df.mau
    
    # Agrupa o DataFrame 'df' pela variável categórica 'var'
    g = df.groupby(var)

    # Cria um novo DataFrame 'biv' que contém informações sobre a relação entre a variável categórica e a variável resposta
    biv = pd.DataFrame({'qt_bom': g['bom'].sum(),  # Soma dos valores "bom" para cada categoria
                        'qt_mau': g['mau'].sum(),  # Soma dos valores "mau" para cada categoria
                        'mau':g['mau'].mean(),  # Proporção de valores "mau" para cada categoria
                        var: g['mau'].mean().index,  # Nome das categorias
                        'cont':g[var].count()})  # Contagem de valores para cada categoria
    
    # Calcula o erro padrão da proporção de valores "mau" para cada categoria
    biv['ep'] = (biv.mau*(1-biv.mau)/biv.cont)**.5
    
    # Calcula os limites de confiança utilizando a distribuição t-Student para a proporção de valores "mau" para cada categoria
    biv['mau_sup'] = biv.mau+t.ppf([0.975], biv.cont-1)*biv.ep
    biv['mau_inf'] = biv.mau+t.ppf([0.025], biv.cont-1)*biv.ep
    
    # Calcula o logit da proporção de valores "mau" para cada categoria
    biv['logit'] = np.log(biv.mau/(1-biv.mau))
    
    # Calcula os limites de confiança para o logit da proporção de valores "mau" para cada categoria
    biv['logit_sup'] = np.log(biv.mau_sup/(1-biv.mau_sup))
    biv['logit_inf'] = np.log(biv.mau_inf/(1-biv.mau_inf))

    # Calcula o Weight of Evidence (WOE) geral
    tx_mau_geral = df.mau.mean()
    woe_geral = np.log(df.mau.mean() / (1 - df.mau.mean()))

    # Calcula o Weight of Evidence (WOE) para cada categoria da variável categórica
    biv['woe'] = biv.logit - woe_geral
    
    # Calcula os limites de confiança para o Weight of Evidence (WOE) para cada categoria da variável categórica
    biv['woe_sup'] = biv.logit_sup - woe_geral
    biv['woe_inf'] = biv.logit_inf - woe_geral

    # Cria um gráfico mostrando o Weight of Evidence (WOE) para cada categoria da variável categórica
    fig, ax = plt.subplots(2,1, figsize=(8,6))
    ax[0].plot(biv[var], biv.woe, ':bo', label='woe')
    ax[0].plot(biv[var], biv.woe_sup, 'o:r', label='limite superior')
    ax[0].plot(biv[var], biv.woe_inf, 'o:r', label='limite inferior')
    
    num_cat = biv.shape[0]
    ax[0].set_xlim([-.3, num_cat-.7])

    ax[0].set_ylabel("Weight of Evidence")
    ax[0].legend(bbox_to_anchor=(.83, 1.17), ncol=3)
    
    ax[0].set_xticks(list(range(num_cat)))
    ax[0].set_xticklabels(biv[var], rotation=15)
    
    ax[1] = biv.cont.plot.bar()

    # Mostrar a tabela no Streamlit
    st.dataframe(biv)
    
    # Mostrar o gráfico no Streamlit
    st.pyplot(fig)
    
    return biv

@st.cache_data
def woe_continua(var, ncat, df):
    
    """
    Calcula o Weight of Evidence (WOE) para uma variável contínua em relação à variável resposta "mau".
    
    Parâmetros:
    - var: Nome da variável contínua.
    - ncat: Número de categorias desejadas para a variável contínua.
    - df: DataFrame pandas contendo as variáveis de interesse.

    Retorna:
    - biv: DataFrame com WOE e proporção de eventos.

    A função realiza as seguintes etapas:
    1. Cria uma nova coluna 'bom' no DataFrame 'df' que é igual a 1 menos o valor da coluna 'mau'.
    2. Divide a variável contínua 'var' em 'ncat' categorias usando qcut, retendo os limites das categorias.
    3. Agrupa o DataFrame 'df' pelas categorias criadas.
    4. Calcula estatísticas como a quantidade de bons, quantidade de maus, proporção de maus e média da variável contínua para cada categoria.
    5. Calcula o erro padrão, limites de confiança e logit da proporção de maus para cada categoria.
    6. Calcula o Weight of Evidence (WOE) para cada categoria.
    7. Gera um gráfico de barras e um gráfico de linha mostrando o WOE para cada categoria e a contagem de observações em cada categoria.
    """
    
    # 1. Cria uma nova coluna 'bom' no DataFrame 'df' que é igual a 1 menos o valor da coluna 'mau'
    df['bom'] = 1 - df.mau
    
    # 2. Divide a variável contínua 'var' em 'ncat' categorias usando qcut, retendo os limites das categorias
    cat_srs, bins = pd.qcut(df[var], ncat, retbins=True, precision=0, duplicates='drop')
    
    # 3. Agrupa o DataFrame 'df' pelas categorias criadas
    g = df.groupby(cat_srs)

    # 4. Calcula estatísticas para cada categoria
    biv = pd.DataFrame({'qt_bom': g['bom'].sum(),  # Soma dos valores "bom" para cada categoria
                        'qt_mau': g['mau'].sum(),  # Soma dos valores "mau" para cada categoria
                        'mau': g['mau'].mean(),   # Proporção de valores "mau" para cada categoria
                        var: g[var].mean(),       # Média da variável contínua para cada categoria
                        'cont': g[var].count()})  # Contagem de valores para cada categoria
    
    # 5. Calcula o erro padrão da proporção de valores "mau" para cada categoria
    biv['ep'] = (biv.mau * (1 - biv.mau) / biv.cont) ** 0.5
    
    # Calcula os limites de confiança para a proporção de valores "mau" para cada categoria
    biv['mau_sup'] = biv.mau + t.ppf([0.975], biv.cont - 1) * biv.ep
    biv['mau_inf'] = biv.mau + t.ppf([0.025], biv.cont - 1) * biv.ep
    
    # Calcula o logit da proporção de valores "mau" para cada categoria
    biv['logit'] = np.log(biv.mau / (1 - biv.mau))
    
    # Calcula os limites de confiança para o logit da proporção de valores "mau" para cada categoria
    biv['logit_sup'] = np.log(biv.mau_sup / (1 - biv.mau_sup))
    biv['logit_inf'] = np.log(biv.mau_inf / (1 - biv.mau_inf))

    # 6. Calcula o Weight of Evidence (WOE) geral
    tx_mau_geral = df.mau.mean()
    woe_geral = np.log(tx_mau_geral / (1 - tx_mau_geral))

    # Calcula o Weight of Evidence (WOE) para cada categoria da variável contínua
    biv['woe'] = biv.logit - woe_geral
    
    # Calcula os limites de confiança para o Weight of Evidence (WOE) para cada categoria da variável contínua
    biv['woe_sup'] = biv.logit_sup - woe_geral
    biv['woe_inf'] = biv.logit_inf - woe_geral

    # 7. Gera um gráfico mostrando o Weight of Evidence (WOE) para cada categoria da variável contínua
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(biv[var], biv.woe, ':bo', label='woe')
    ax[0].plot(biv[var], biv.woe_sup, 'o:r', label='limite superior')
    ax[0].plot(biv[var], biv.woe_inf, 'o:r', label='limite inferior')
    
    num_cat = biv.shape[0]

    ax[0].set_ylabel("Weight of Evidence")
    ax[0].legend(bbox_to_anchor=(.83, 1.17), ncol=3)
    
    ax[1] = biv.cont.plot.bar()

    # Mostrar a tabela no Streamlit
    st.dataframe(biv)
    
    # Mostrar o gráfico no Streamlit
    st.pyplot(fig)
    
    return biv

def main():  
    # Definir o template
    st.set_page_config(page_title='Análises',
                       page_icon='💲',
                       layout='wide')

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

    if df is not None:
        # Adicionar caixa de seleção na barra lateral
        selecao_dados = st.sidebar.selectbox(
            "Selecione uma opção",
            ("Info", "Descritiva")
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

        else:
            # Adicionar caixa de seleção na barra lateral
            selecao_compras_acessos = st.sidebar.selectbox(
                "Selecione uma opção",
                ("Bivariada", "IV/WOE")
            )
            if selecao_compras_acessos == "Bivariada":
                # Adicionar um widget multiselect para selecionar as variáveis a serem observadas
                variaveis = st.multiselect("Selecione as variáveis", df.columns.tolist())

                # Criar duas colunas
                col1, col2 = st.columns(2)

                # Adicionar um slider para selecionar o número de categorias na primeira coluna
                ncat = col1.slider("Selecione o número de categorias", 2, 10, step=1)

                for var in variaveis:
                    if len(df[var].unique()) > 10:
                        plot_cont_bivariate(df, var, ncat)
                    else:
                        plot_cat_bivariate(df, var)
            else:  # IV/WOE
                # Adicionar outro selectbox para IV/WOE
                selecao_iv_woe = st.selectbox(
                    "Selecione uma opção",
                    ("IV", "WOE")
                )
                if selecao_iv_woe == "IV":
                    # Excluindo as colunas que não precisam ser calculadas
                    df = df.drop(columns=['Unnamed: 0', 'data_ref', 'index'])
                    # Chama a função calcula_iv
                    st.dataframe(calcula_iv(df))
                else:  # WOE
                    # Adicionar um widget multiselect para selecionar as variáveis a serem observadas
                    variaveis = st.multiselect("Selecione as variáveis", df.columns.tolist())
                    # Adicionar um slider para selecionar o número de categorias
                    ncat = st.slider("Selecione o número de categorias", 1, 10, step=1)
                    for var in variaveis:
                        if len(df[var].unique()) > 10:  # Se a variável tem mais de 10 valores únicos, é contínua
                            woe_continua(var, ncat, df)
                        else:  # Se a variável tem 10 ou menos valores únicos, é categórica
                            woe_discreta(var, df)

if __name__ == "__main__":
    main()
