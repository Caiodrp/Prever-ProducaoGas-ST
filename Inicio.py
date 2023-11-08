import streamlit as st

# Definir o template
st.set_page_config(page_title='Início',
                page_icon='🏭',
                layout='wide')

def main():
    # Apresenta a imagem na barra lateral da aplicação
    url = "https://raw.githubusercontent.com/Caiodrp/Previsao-Renda-Streamlit/main/img/Design%20sem%20nome.jpg"
    st.sidebar.image(url,use_column_width=True)

    # Título centralizado
    st.write(
        '<div style="display:flex; align-items:center; justify-content:center;">'
        '<h1 style="font-size:4.5rem;">Produção de Gás Natural ANP</h1>'
        '</div>',
        unsafe_allow_html=True
    )

    # Subtítulo
    st.write(
        '<div style="display:flex; align-items:center; justify-content:center;">'
        '<h2 style="font-size:2.5rem;">Prevendo a produção de gás natural em poços da ANP</h2>'
        '</div>',
        unsafe_allow_html=True
    )

    # Divisão
    st.write("---")

    # Imagem do lado da explicação
    col1, col2 = st.columns(2)

    col1.write(
        "<p style='font-size:1.5rem;'> Esta aplicação web é uma <b>ferramenta de gerenciamento</b> de um <b>modelo de regressão</b> que visa prever a <b>média da produção diária de gás Natural</b> nos poços da ANP a partir de características químicas, físicas e de produção."
        "<br>"
        "Utiliza de um modelo treinado de <b>Regressão Linear</b> para prever a produção média diária de gás Natural nos poços da ANP a partir de características do poço.</p>"
        "<p style='font-size:1.5rem;'> Permite o carregamento de novos dados, estimando a produção média diária de gás Natural nos poços da ANP. Além de demonstrar visualmente relações entre as variáveis explicativas e a variável resposta, e automatizar relatórios gerenciais dos conjuntos de dados.</p>",
        unsafe_allow_html=True
    )

    col2.write(
        '<div style="position:relative;"><iframe src="https://gifer.com/embed/Otf" width="480" height="480" frameBorder="0" allowFullScreen></iframe></div>',
        unsafe_allow_html=True
    )

    # Divisão
    st.write("---")

    st.write(
        '<h3 style="text-align:left;">Autor</h3>'
        '<ul style="list-style-type: disc; margin-left: 20px;">'
        '<li>Caio Douglas Rodrigues de Paula</li>'
        '<li><a href="https://github.com/Caiodrp/Prever-ProducaoGas-ST">GitHub</a></li>'
        '</ul>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

