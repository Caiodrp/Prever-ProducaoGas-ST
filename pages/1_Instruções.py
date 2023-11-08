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
    # Informações sobre os dados
    st.write('Os dados foram obtidos a partir do link: https://dados.gov.br/dados/conjuntos-dados/producao-de-petroleo-e-gas-natural-por-poco')
    st.write('Eles foram transformados conforme explicado no Notebook no diretório do GitHub.')
    st.write('O segundo arquivo contém as variáveis utilizadas no treinamento do modelo.')

    # Modelos dos arquivos CSV
    st.write('# Modelos dos arquivos CSV')
    st.write('Os arquivos a serem usados devem ter o nome e a ordem das colunas idênticos ao do modelo.')

    # Adicionando botão para download do arquivo df_todos.csv
    url = "https://github.com/Caiodrp/Prever-ProducaoGas-ST/raw/main/df_todos.csv"
    st.markdown(download_file(url), unsafe_allow_html=True)

    # Adicionando botão para download do arquivo X_train_transformado2.csv
    url = "https://github.com/Caiodrp/Prever-ProducaoGas-ST/raw/main/X_train_transformado2.csv"
    st.markdown(download_file(url), unsafe_allow_html=True)


    # Adicionando texto
    st.write(
        """
        # Análise

        Na página "Análise", você pode carregar e visualizar diferentes informações sobre os dados. 

        ### Info

        A subseção "Info" exibe informações sobre os dados.

        ### Descritiva

        A subseção "Descritiva" permite o carregamento de dados para análises descritivas das variáveis.

        ### Suposições do Modelo

        Esta parte faz análises estatísticas visando a criação do modelo.

        # Prever Produção

        Na página "Prever Produção", você pode fazer previsões usando novos dados inseridos ou carregados.

        ### Prever Novos Dados

        Para fazer previsões para novos dados, você precisa preencher os campos na barra lateral ou carregar um arquivo CSV com novos dados e clicar em "Prever Produção". 

        ### Relatório Gerencial

        A subseção "Relatório Gerencial" contém um relatório do modelo, mostrando o quanto cada variável usada para o modelo interfere quantificamente no resultado da média diária de gás natural.
        
        """
    )

# Chamar a função principal
if __name__ == '__main__':
    main()
