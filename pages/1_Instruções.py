import pandas as pd
import numpy as np
import streamlit as st
import base64
import requests

from io import BytesIO

# Definir o template
st.set_page_config(page_title='Instruções',
                page_icon='💲',
                layout='wide')

# Título centralizado
st.write(
    '<div style="display:flex; align-items:center; justify-content:center;">'
    '<h1 style="font-size:4.5rem;">Instruções</h1>'
    '</div>',
    unsafe_allow_html=True
)

# Divisão
st.write("---")

# Adicionando texto antes do vídeo
st.write("Este é um tutorial em vídeo sobre como usar a aplicação")

# URL do seu vídeo do GitHub
video_url = "https://github.com/Caiodrp/Prever-Inadimplencia-ST/blob/1417e8473c07a42ebcc76f207a3efb2865b72761/Tutorial.webm?raw=true"

# Exibindo o vídeo na página do Streamlit
st.video(video_url)

@st.cache_data()
def get_data(url):
    return BytesIO(requests.get(url).content)

url = 'https://github.com/Caiodrp/Prever-Inadimplencia-ST/blob/main/csv/credit_scoring.csv'
data = get_data(url)
st.download_button(label='Download CSV', data=data, file_name='credit_scoring.csv', mime='text/csv')

# Adicionando texto
st.write(
    """
    # Análises

    Na página Análises, se encontram as principais características da base de dados, tais como informações estatísticas e de realção com a variável reposta 

    ### Info

    A subseção "Info" exibe informações principais sobre a estrutura e o dicionário dos dados.

    ### Descritiva

    A subseção "Descritiva" contém duas partes:

    - **Bivariada**: Esta parte exibe a relação entre a variável de interesse e as demais variáveis.
    
    - **WOE/IV**: Esta parte exibe a importância da variável para o modelo.

    # Prever Inadimplência

    Na página "Prever Inadimplência", você pode fazer previsões de inadimplência para novos dados.

    ### Prever Novos Dados

    Para fazer previsões para novos dados, você precisa preencher os campos na barra lateral ou carregar um arquivo CSV com novos dados e clicar em "Prever Inadimplência". 

    ### Relatório Gerencial

    A subseção "Relatório Gerencial" contém relatórios estatísticos levando em conta o problema de negócio.
    
    """
)
