import pandas as pd
import numpy as np
import streamlit as st
import base64
import requests

# Definir o template
st.set_page_config(page_title='Instruções',
                page_icon='🏭',
                layout='wide')

# Função para baixar o arquivo
def download_file(url):
    response = requests.get(url)
    b64 = base64.b64encode(response.content).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="arquivo.xlsx">Baixar arquivo</a>'

# Função principal para o corpo do script
def main():
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

    # Adicionando vídeo
    st.write()
    st.write(
        '<div style="display:flex; align-items:center; justify-content:center;">'
        '<iframe width="560" height="315" src="https://www.youtube.com/embed/dQw4w9WgXcQ?start=40" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>'
        '</div>',
        unsafe_allow_html=True
    )
    # Arquivo csv modelo 
    st.write('# Modelos dos arquivos CSV')
    st.write('Os arquivos a serem usados devem ser com o nome e ordem das colunas identicos a do modelo')
    url = "https://raw.githubusercontent.com/Caiodrp/Prever-ProducaoGas-ST/main/df_todos.csv"

    # Adicionando botão para download
    st.markdown(download_file(url), unsafe_allow_html=True)

    # Adicionando texto
    st.write(
        """
        # Análise

        Na página "Análise", você pode carregar e visualizar diferentes informações sobre o conjunto de dados "online_shoppers_intention", que são o comportamento de diversos acessos de usuários em diferentes tipos de sites, disponível em https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset.

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

# Chamar a função principal
if __name__ == '__main__':
    main()
