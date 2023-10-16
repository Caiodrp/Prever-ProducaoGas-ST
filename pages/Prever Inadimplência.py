import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import base64
import plotly.graph_objects as go
import plotly.express as px

from pycaret.classification import load_model, predict_model

@st.cache_data
def carregar_dados(uploaded_file):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        try:
            if file_extension == "csv":
                # Se a extensão for .CSV, leia o arquivo como CSV
                return pd.read_csv(uploaded_file)
            elif file_extension == "ftr":
                # Se a extensão for .ftr, leia o arquivo como Feather
                return pd.read_feather(uploaded_file)
            else:
                st.warning("Extensão de arquivo não suportada. Por favor, faça upload de um arquivo .CSV ou .ftr.")
                return None
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {str(e)}")
            return None

@st.cache_resource
def carregar_modelo(path):
    return load_model(path)

@st.cache_data
def transform_dataframe(df):
    # Certifique-se de que o input é um DataFrame
    df = pd.DataFrame(df)
    
    # Aplicar as transformações
    df['qtd_filhos'] = df['qtd_filhos'].apply(lambda x: 5 if x > 4 else x).astype('object')
    df['tipo_renda'] = df['tipo_renda'].apply(lambda x: 'Servidor público/Bolsista' if x in ['Bolsista', 'Servidor público'] else x)
    df['educacao'] = df['educacao'].replace('Pós graduação', 'Superior completo')
    df['qt_pessoas_residencia'] = df['qt_pessoas_residencia'].apply(lambda x: 6 if x > 5 else x).astype('object')
    labels_te = ['0a2', '2a5', '5a6', '6a7','7a12','12mais']
    df['tempo_emprego_cat'] = pd.qcut(df['tempo_emprego'], q=6, labels=labels_te, duplicates='drop').astype('object')
    labels_idade = ['18a33', '33a40', '40a47', '47a55','55mais'] 
    df['idade_cat'] = pd.qcut(df['idade'], q=5, labels=labels_idade, duplicates='drop').astype('object')
    return df

@st.cache_data
def separar_dados(df):
    # Data de corte como "2016-01-01" para o OOT
    data_final = pd.to_datetime('2016-01-01')

    # Converte a coluna 'data_ref' para datetime
    df['data_ref'] = pd.to_datetime(df['data_ref'])

    # Separa os dados para treinamento e validação OOT
    df_train = df[df['data_ref'] < data_final]

    # Retira as colunas 'data_ref' e 'index'
    df_train.drop(['data_ref', 'index'], axis=1, inplace=True)

    return df_train

@st.cache_resource
def gerar_score(df):
    # Carregando o modelo
    url = 'https://github.com/Caiodrp/Prever-Inadimplencia-ST/raw/main/reg_logi.pkl'
    r = requests.get(url)
    with open('reg_logi.pkl', 'wb') as f:
        f.write(r.content)

    modelo = load_model('reg_logi')

    # Gerando as previsões
    predictions = predict_model(modelo, data=df.drop('mau', axis=1))

    # Adicionando as previsões ao dataframe
    df['prediction_score'] = predictions['prediction_score']

    return df

@st.cache_data
def plotar_graficos(df):
    tab_gan = df.sort_values(by='prediction_score').reset_index().copy()
    
    tab_gan['tx_mau_acum'] = tab_gan.mau.cumsum()/tab_gan.shape[0]
    
    tab_gan['pct_mau_acum'] = tab_gan.mau.cumsum()/tab_gan.mau.sum()
    
    tab_gan['red_mau_acum'] = 1-tab_gan.pct_mau_acum
    
    tab_gan['pct_aprovacao'] = np.array(range(tab_gan.shape[0]))/tab_gan.shape[0]
    
    # Criação do primeiro gráfico para a taxa de maus acumulados
    fig1 = go.Figure()
    
    # Adicionando a linha de maus acumulados
    fig1.add_trace(go.Scatter(x=tab_gan['tx_mau_acum'], y=tab_gan['pct_aprovacao'], mode='lines', name='Maus Acumulados'))
    
    # Layouts e os eixos do gráfico
    fig1.update_layout(
        title='Gráfico de Aprovação vs Maus Acumulados',
        xaxis=dict(title='Taxa de Maus Acumulados', range=[0, max(tab_gan['tx_mau_acum'])], tick0=0, dtick=0.005),
        yaxis=dict(title='% de Aprovação', range=[0, 1], tick0=0, dtick=0.1)
    )
    
    # Mostrando o primeiro gráfico
    st.plotly_chart(fig1)
    
    # Criação do segundo gráfico para a redução de maus acumulados
    fig2 = go.Figure()
    
    # Adicionando a linha de redução de maus acumulados com cor vermelha
    fig2.add_trace(go.Scatter(x=tab_gan['red_mau_acum'], y=tab_gan['pct_aprovacao'], mode='lines', name='Redução de Maus Acumulados', line=dict(color='red')))
    
    # Atualizando os layouts e os eixos do gráfico
    fig2.update_layout(
        title='Gráfico de Aprovação vs Redução de Maus Acumulados',
        xaxis=dict(title='Redução de Maus Acumulados', range=[0, max(tab_gan['red_mau_acum'])], tick0=0, dtick=0.05),
        yaxis=dict(title='% de Aprovação', range=[0, 1], tick0=0, dtick=0.05)
    )
    
    # Mostrando o segundo gráfico
    st.plotly_chart(fig2)

@st.cache_data
def plot_graph(df, var):
    # Cria um novo DataFrame
    df_new = df.copy()

    # Categorize the variable if it has more than 10 unique values
    if df_new[var].nunique() > 10:
        df_new[var+'_cat'] = pd.qcut(df_new[var], q=10)
    else:
        df_new[var+'_cat'] = df_new[var]

    # Sort the dataframe and calculate cumulative sums
    tmp = df_new.sort_values(by=[var+'_cat','prediction_score'], ascending=True).copy()
    tmp['freq']=1
    tmp[['freq_ac', 'maus_ac']] = tmp.groupby([var+'_cat'])[['freq', 'mau']].cumsum()
    tmp['freq_fx_'+var] = tmp.groupby([var+'_cat'])['freq'].transform('sum')
    
    # Calculate percentages
    tmp['pct_aprovados'] = tmp.freq_ac/tmp['freq_fx_'+var]
    tmp['tx_maus_pto_corte'] = tmp['maus_ac']/tmp['freq_ac']

    # Create a new figure
    fig = go.Figure()

    # For each category, add a trace to the figure
    for cat in tmp[var+'_cat'].unique():
        df_cat = tmp[tmp[var+'_cat'] == cat]
        fig.add_trace(go.Scatter(x=df_cat["prediction_score"], y=df_cat["pct_aprovados"], mode='lines', name=str(cat)))

    # Update layout
    fig.update_layout(title='Taxa de maus por %aprovação', xaxis_title='prediction_score', yaxis_title='pct_aprovados')

    # Display the graph in Streamlit
    st.plotly_chart(fig)

def main():
    # Definir o template
    st.set_page_config(page_title='Prever Inadimplência',
                       page_icon='💲',
                       layout='wide')

    # Título centralizado
    st.markdown('<div style="display:flex; align-items:center; justify-content:center;"><h1 style="font-size:4.5rem;">Prever Inadimplência</h1></div>',
                unsafe_allow_html=True)

    # Adicionar caixa de seleção na barra lateral
    selecao_dados = st.sidebar.selectbox(
        "Selecione uma opção",
        ("Prever Novos Clientes", "Relatório Gerencial")
    )

    if selecao_dados == "Prever Novos Clientes":
        opcao_dados = st.sidebar.radio(
            "Como você gostaria de fornecer os dados?",
            ("Carregar Dados", "Inserir Dados")
        )

        df = None

        if opcao_dados == "Carregar Dados":
            uploaded_file = st.sidebar.file_uploader(
                "Faça upload do arquivo CSV ou FTR",
                type=["csv", "ftr"]
            )

            df = carregar_dados(uploaded_file)
            if df is not None:
                # Obter o número de linhas em df
                num_linhas = df.shape[0]

                # limitar a 250000 linhas
                num_amostras = min(num_linhas, 250000)

                df = df.sample(n=num_amostras, random_state=1)

                df_transformed = transform_dataframe(df)

                model_file_path = 'C:\\Users\\user\\OneDrive\\Documentos\\Data Science\\Cientista de Dados\\Reg_Log_Inadimplencia\\reg_logi'

                # Carregar o modelo usando a função carregar_modelo
                modelo = carregar_modelo(model_file_path)

                # Fazer previsões com o modelo
                resultados = predict_model(modelo, data = df_transformed)

                # Adicionar previsões ao DataFrame original
                df_transformed['Previsão'] = np.where(resultados['prediction_label'] == 0, "Adimplente", "Inadimplente")

                csv = df_transformed.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="previsoes.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)


        elif opcao_dados == "Inserir Dados":
            st.write("Por favor, insira os dados:")
            entradas = {}

            # Carregando o dataframe
            df = pd.read_csv('https://raw.githubusercontent.com/Caiodrp/Prever-Inadimplencia-St/main/credit_scoring.csv')

            # Transformando o dataframe de entrada
            df = transform_dataframe(df)

            if df is not None:
                with st.form(key='my_form'):
                    for coluna in df.columns:
                        if coluna not in ['data_ref', 'index', 'mau', 'Unnamed: 0']:
                            if np.issubdtype(df[coluna].dtype, np.number):
                                entradas[coluna] = st.number_input(f'Insira um valor para {coluna}')
                            else:
                                categorias = df[coluna].unique().tolist()
                                entradas[coluna] = st.selectbox(f'Selecione uma opção para {coluna}', categorias)

                    submit_button = st.form_submit_button(label='Prever')

                    if submit_button:
                        df_entrada = pd.DataFrame([entradas])
                        st.dataframe(df_entrada)

                        # Carregando o modelo
                        url = 'https://github.com/Caiodrp/Prever-Inadimplencia-St/raw/main/reg_logi.pkl'
                        r = requests.get(url)
                        with open('reg_logi.pkl', 'wb') as f:
                            f.write(r.content)

                        modelo = carregar_modelo('reg_logi')

                        # Fazendo a previsão com o modelo carregado
                        previsao = predict_model(modelo, data=df_entrada)

                        # Escrevendo o resultado na página do Streamlit
                        if previsao['prediction_label'][0] == 0:
                            st.markdown("<h2 style='text-align: center; color: green;'>ADIMPLENTE</h2>", unsafe_allow_html=True)
                        elif previsao['prediction_label'][0] == 1:
                            st.markdown("<h2 style='text-align: center; color: red;'>INADIMPLENTE</h2>", unsafe_allow_html=True)

    elif selecao_dados == "Relatório Gerencial":
        # Adicione o widget de upload de arquivo na barra lateral
        uploaded_file = st.sidebar.file_uploader("Escolha um arquivo .csv", type=['csv'])
        if uploaded_file is not None:
            # Carregue os dados usando a função carregar_dados
            df = carregar_dados(uploaded_file)

            # Verifique se um arquivo foi carregado
            df = separar_dados(df)
            df = transform_dataframe(df)
            df = gerar_score(df)

            # Adicione o widget de rádio na página do Streamlit
            report_type = st.radio("Escolha o tipo de relatório", ("Relatório Geral", "Relatório por Características"))

            if report_type == "Relatório Geral":
                plotar_graficos(df)
            elif report_type == "Relatório por Características":
                var = st.selectbox('Selecione a variável:', [col for col in df.columns if col not in ['Unnamed: 0', 'prediction_score']])
                plot_graph(df, var)

if __name__ == "__main__":
    main()
