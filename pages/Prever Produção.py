import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import io
import plotly.graph_objects as go
import plotly.express as px
import tempfile

from io import BytesIO
from pycaret.regression import load_model, predict_model

# Definir o template
st.set_page_config(page_title='Prever Produção',
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
def importa_dados(url):
    df_imported = pd.read_csv(url)
    return df_imported

def agrupar_operadores(df):
    # Definindo os grupos de operadores
    grupos = ['Perenco Brasil', 'Petro Rio Jaguar', 'Petro Rio O&G', 'Prio Bravo', 'Shell Brasil']
    grupos_expandidos = ['3R Potiguar', '3R Fazenda Belém', 'Petrobras', 'Carmo', 'Mandacaru Energia', '3R Candeias', 'Seacrest SPE Cricaré', 'Níon Energia', 'Nova Petróleo', 'Petrosynergy', 'Seacrest', 'Recôncavo E&P', 'Phoenix Óleo & Gás', '3R Macau', 'Potiguar E&P SA', 'Capixaba Energia', 'SPE Miranga', 'Petrom', 'Origem Alagoas']

    # Criando a nova coluna 'Operador_Agrupado'
    df['Operador_Agrupado'] = df['Operador'].apply(lambda x: x if x in grupos else None)

    # Agrupando as categorias que estão separadas por '_'
    for grupo in grupos_expandidos:
        df.loc[df['Operador'].str.contains(grupo), 'Operador_Agrupado'] = '3R Potiguar_3R Fazenda Belém_Petrobras_Carmo_Mandacaru Energia_3R Candeias  SA_Seacrest SPE Cricaré_Níon Energia_Nova Petróleo_3R Candeias_Petrosynergy_Seacrest_Recôncavo E&P_Phoenix Óleo & Gás_3R Macau_Potiguar E&P SA_Capixaba Energia_SPE Miranga_Petrom_Origem Alagoas'

    # Removendo as linhas onde 'Operador_Agrupado' é None
    df = df.dropna(subset=['Operador_Agrupado'])

    return df

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

def transformacao_dados_prev(df):
    # Cria um mapa dos valores de 'Nome Poço ANP' para 'Corrente' e 'Grau API'
    mapa_corrente = df.dropna(subset=['Corrente']).set_index('Nome Poço ANP')['Corrente'].to_dict()
    mapa_grau_api = df.dropna(subset=['Grau API']).set_index('Nome Poço ANP')['Grau API'].to_dict()

    # Preencha os valores faltantes nas colunas 'Corrente' e 'Grau API' com os valores correspondentes do mapa
    df['Corrente'] = df['Corrente'].fillna(df['Nome Poço ANP'].map(mapa_corrente))
    df['Grau API'] = df['Grau API'].fillna(df['Nome Poço ANP'].map(mapa_grau_api))

    # Exclua as linhas que ainda têm valores NaN
    df = df.dropna()

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
    df = agrupar_operadores(df)

    # Lista de colunas para transformar
    cols_to_transform = ['Petróleo (bbl/dia)', 'Água (bbl/dia)', 'Tempo de Produção (hs por mês)', 'Gás Natural (Mm³/dia)']

    # Crie novas colunas com o sufixo '_log' para as transformações
    for col in cols_to_transform:
        df[col + '_log'] = np.log(df[col])

    # Crie uma nova coluna com o sufixo '_sqrt' para a transformação de raiz quadrada
    df['Água (bbl/dia)_sqrt'] = np.sqrt(df['Água (bbl/dia)'])
    df['Tempo de Produção (hs por mês)_sqrt'] = np.sqrt(df['Tempo de Produção (hs por mês)'])

    # Funçãoq que categoriza o API
    df['Grau_API_Cat'] = df['Grau API'].apply(categoriza_grau_api)

    return df

def transformacao_dados_prev_antes(df):
    # Cria um mapa dos valores de 'Nome Poço ANP' para 'Corrente' e 'Grau API'
    mapa_corrente = df.dropna(subset=['Corrente']).set_index('Nome Poço ANP')['Corrente'].to_dict()
    mapa_grau_api = df.dropna(subset=['Grau API']).set_index('Nome Poço ANP')['Grau API'].to_dict()

    # Preencha os valores faltantes nas colunas 'Corrente' e 'Grau API' com os valores correspondentes do mapa
    df['Corrente'] = df['Corrente'].fillna(df['Nome Poço ANP'].map(mapa_corrente))
    df['Grau API'] = df['Grau API'].fillna(df['Nome Poço ANP'].map(mapa_grau_api))

    # Exclua as linhas que ainda têm valores NaN
    df = df.dropna()

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
    df = agrupar_operadores(df)

    # Funçãoq que categoriza o API
    df['Grau_API_Cat'] = df['Grau API'].apply(categoriza_grau_api)

    return df

@st.cache_resource
def carregar_modelo(url):
    response = requests.get(url)
    modelo = joblib.load(io.BytesIO(response.content))
    return modelo

@st.cache_data
def interpret_coefficients(model, df_imported_encoded):
    # Função para interpretar os coeficientes do modelo
    coef = pd.DataFrame(model.coef_, df_imported_encoded.columns, columns=['Coefficient'])
    return coef

@st.cache_data
def plot_continuous(coef_df, variable):
    # Calcule a mudança na variável de resposta para um aumento de 10% na variável explicativa
    changes = [(np.exp(coef_df.loc[variable, 'Coefficient'] * increase / 100) - 1) * 100 for increase in np.arange(0, 110, 10)]
    
    # Crie o gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, 110, 10), changes, marker='o')
    plt.title('Impacto de um aumento em ' + variable + ' na variável de resposta')
    plt.xlabel('Aumento na ' + variable + ' (%)')
    plt.ylabel('Mudança na variável de resposta (%)')
    plt.xticks(np.arange(0, 110, 10))
    plt.yticks(np.arange(min(changes), max(changes), 10))
    plt.grid(True)

    # Exibir o gráfico no Streamlit
    st.pyplot(plt)

@st.cache_data
def plot_categorical(coef_df, variable):
    # Filtrar as colunas que pertencem à variável categórica selecionada
    categories = [col for col in coef_df.index if col.startswith(variable)]
    
    # Calcule a mudança na variável de resposta para cada categoria
    changes = [np.exp(coef_df.loc[category, 'Coefficient']) - 1 for category in categories]
    
    # Crie o gráfico
    plt.figure(figsize=(12, 6))
    bar_colors = ['#{:06x}'.format(int(np.random.rand() * 0xFFFFFF)) for _ in range(len(categories))]
    plt.bar(categories, changes, color=bar_colors)
    plt.title('Impacto das categorias de ' + variable + ' na variável de resposta')
    plt.xlabel('Categoria')
    plt.ylabel('Mudança na média de produção de Gás por categoria (%)')
    plt.xticks(rotation=90)
    plt.grid(True)

    # Exibir o gráfico no Streamlit
    st.pyplot(plt)

def main():  
    # Título centralizado
    st.markdown('<div style="display:flex; align-items:center; justify-content:center;"><h1 style="font-size:4.5rem;">Prever Produção</h1></div>',
                unsafe_allow_html=True)

    # Divisão
    st.write("---")

    # .CSV usado no modelo
    url_dados = 'https://github.com/Caiodrp/Prever-ProducaoGas-ST/raw/main/df_todos.csv'

    # dfs para serem usados para inserir ou carregar os novos dados
    df_imported_inserir = pd.read_csv(url_dados)
    df_imported_inserir = transformacao_dados_prev_antes(df_imported_inserir)

    # Retirando colunas que não tem haver com a produção, redundantes ou não significativas.
    cols_to_drop = ['Nome Poço Operador','Nome Poço ANP','Número do Contrato', 'Período','Condensado (bbl/dia)',
                    'Gás Natural (Mm³/dia) N Assoc', 'Gás Natural (Mm³/dia) Total', 'Volume Gás Royalties (m³/mês)',
                    'Instalação Destino', 'Tipo Instalação', 'Período da Carga','Óleo (bbl/dia)','Campo']
    df_imported_inserir = df_imported_inserir.drop(columns=cols_to_drop)

    if df_imported_inserir is not None:
        # Adicionar caixa de seleção na barra lateral
        selecao_prever = st.sidebar.selectbox(
            "Selecione uma opção",
            ("Previsões", "Relatório Gerencial")
        )

        if selecao_prever == "Previsões":
            # Adicionar um widget de rádio para escolher entre inserir dados ou carregar dados
            data_input = st.radio(
                "Escolha uma opção",
                ("Inserir dados", "Carregar dados")
            )

            if data_input == "Inserir dados":
                # Remova as colunas especificadas
                df_imported_inserir = df_imported_inserir.drop(columns=['Estado', 'Bacia', 'Operador', 'Grau API'])

                # Para cada coluna no DataFrame, adicione um widget apropriado para inserir dados
                input_data = {}
                for col in df_imported_inserir.columns:
                    if df_imported_inserir[col].dtype in ['int64', 'float64']:
                        input_data[col] = st.number_input(f'Insira o valor para {col}', min_value=1e-10)
                    else:  # coluna é categórica
                        # Obter a resposta do usuário
                        resposta = st.selectbox(f'Selecione o valor para {col}', options=df_imported_inserir[col].unique().tolist())

                        # Para cada categoria única, crie uma coluna binária
                        for categoria in df_imported_inserir[col].unique():
                            if categoria == resposta:
                                input_data[col + '_' + categoria] = 1
                            else:
                                input_data[col + '_' + categoria] = 0

                # Transformar as respostas do usuário em um dataframe
                df_input = pd.DataFrame([input_data])

                # Lista de colunas para transformar
                cols_to_transform = ['Petróleo (bbl/dia)', 'Água (bbl/dia)', 'Tempo de Produção (hs por mês)', 'Gás Natural (Mm³/dia)']

                # Crie novas colunas com o sufixo '_log' para as transformações
                for col in cols_to_transform:
                    df_input[col + '_log'] = np.log(df_input[col])

                url = 'https://github.com/Caiodrp/Prever-ProducaoGas-ST/raw/main/reg_final.pkl'
                modelo = carregar_modelo(url)

                st.write(df_input)

                # Lista de todas as colunas que serão usadas pelo modelo
                todas_as_colunas = ['Estados_Bacias','Local','Operador_Agrupado','Grau_API_Cat',
                'Petróleo (bbl/dia)_log','Água (bbl/dia)_log',
                'Tempo de Produção (hs por mês)_log']

                # Para cada coluna no DataFrame, verifique se ela começa com o nome de uma das variáveis categóricas
                colunas_para_usar = []
                for col in df_input.columns:
                    for var in todas_as_colunas:
                        if col.startswith(var):
                            colunas_para_usar.append(col)

                # Crie um novo DataFrame que contém apenas as colunas para usar
                df_input = df_input[colunas_para_usar]

                # Remover as colunas especificadas
                df_input = df_input.drop(columns=['Estados_Bacias_Amazonas_SolimÃµes', 'Grau_API_Cat_Leve_Particular'])

                if st.button('PREVER'):
                    # Fazer previsões com o modelo
                    pred_log = modelo.predict(df_input)

                    # Retransformar a previsão para a escala original
                    pred = np.exp(pred_log)

                    st.write(pred)

            else:
                # Carregar dados
                uploaded_file = st.sidebar.file_uploader("Faça upload do arquivo CSV", type=["csv"])
                df_carregar = carregar_dados(uploaded_file)

                # Limitar o DataFrame a 1000 linhas
                if len(df_carregar) > 1000:
                    df_carregar = df_carregar.sample(n=1000)

                # Aplicar a função transformacao_dados_prev
                df_carregar = transformacao_dados_prev(df_carregar)

                # Retirando colunas que não tem haver com a produção, redundantes ou não significativas.
                cols_to_drop = ['Nome Poço Operador','Nome Poço ANP','Número do Contrato', 'Período','Condensado (bbl/dia)',
                    'Gás Natural (Mm³/dia) N Assoc', 'Gás Natural (Mm³/dia) Total', 'Volume Gás Royalties (m³/mês)',
                    'Instalação Destino', 'Tipo Instalação', 'Período da Carga','Óleo (bbl/dia)','Campo']
                df_carregar = df_carregar.drop(columns=cols_to_drop)

                # Lista de colunas para transformar
                cols_to_transform = ['Petróleo (bbl/dia)', 'Água (bbl/dia)', 'Tempo de Produção (hs por mês)', 'Gás Natural (Mm³/dia)']

                # Crie novas colunas com o sufixo '_log' para as transformações
                for col in cols_to_transform:
                    df_carregar[col + '_log'] = np.log(df_carregar[col])

                # Carregar o modelo
                url = 'https://github.com/Caiodrp/Prever-ProducaoGas-ST/raw/main/reg_final.pkl'
                modelo = carregar_modelo(url)

                # Suponha que 'modelo' é o seu modelo treinado e 'colunas_modelo' são as colunas usadas para treinar o modelo
                colunas_modelo = modelo.get_booster().feature_names
                st.write(colunas_modelo)

                # # Lista de todas as colunas que serão usadas pelo modelo
                # vars_usar = ['Estados_Bacias','Local','Operador_Agrupado','Grau_API_Cat',
                # 'Petróleo (bbl/dia)_log','Água (bbl/dia)_log',
                # 'Tempo de Produção (hs por mês)_log']

                # # Aplicar get_dummies para preparar os dados para o modelo
                # df_carregar = pd.get_dummies(df_carregar)
                # st.write(df_carregar)

                # if st.button('PREVER'):
                #     # Fazer previsões com o modelo
                #     pred_log = modelo.predict(df_carregar)

                #     # Retransformar a previsão para a escala original
                #     pred = np.exp(pred_log)

                # # Fazer previsões com o modelo
                # pred = predict_model(modelo, data=df)

                # # Transformar a coluna 'prediction_label' com a função exponencial
                # pred['prediction_label'] = np.exp(pred['prediction_label'])

                # st.write(pred)

        # else:
        #     # Variáveis selecionadas
        #     vars_encoded = ['Estados_Bacias','Local','Operador_Agrupado','Grau_API_Cat',
        #     'Gás Natural (Mm³/dia)_log','Petróleo (bbl/dia)_log','Água (bbl/dia)_log',
        #                     'Tempo de Produção (hs por mês)_log']

        #     # Aplicar get_dummies nas variáveis selecionadas
        #     df_imported_encoded = pd.get_dummies(df_imported[vars_encoded], columns=df[vars_encoded].select_dtypes(include=['object']).columns)

        #     # Pegando o df das variáveis do modelo
        #     coef_df = interpret_coefficients(modelo, df_imported_encoded)

        #     # Variáveis contínuas e categóricas
        #     vars_continuas = ['Gás Natural (Mm³/dia)_log', 'Petróleo (bbl/dia)_log', 'Água (bbl/dia)_log', 'Tempo de Produção (hs por mês)_log']
        #     vars_categoricas = ['Estados_Bacias', 'Local', 'Operador_Agrupado', 'Grau_API_Cat']

        #     # Widget de rádio para escolher entre variáveis contínuas e categóricas
        #     tipo_variavel = st.radio("Escolha uma opção", ("Variáveis Contínuas", "Variáveis Categóricas"))

        #     # Caixa de seleção para selecionar a variável a ser plotada
        #     if tipo_variavel == "Variáveis Contínuas":
        #         variavel_selecionada = st.selectbox("Selecione a variável", options=vars_continuas)
        #         plot_continuous(coef_df, variavel_selecionada)
        #     else:
        #         variavel_selecionada = st.selectbox("Selecione a variável", options=vars_categoricas)
        #         plot_categorical(coef_df, variavel_selecionada)

if __name__ == "__main__":
    main()
