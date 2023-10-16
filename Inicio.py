import streamlit as st

def main():
    # Definir o template
    st.set_page_config(page_title='Credit Scoring',
                    page_icon='💲',
                    layout='wide')

    # Apresenta a imagem na barra lateral da aplicação
    url = "https://raw.githubusercontent.com/Caiodrp/Previsao-Renda-Streamlit/main/img/Design%20sem%20nome.jpg"
    st.sidebar.image(url,use_column_width=True)

    # Título centralizado
    st.write(
        '<div style="display:flex; align-items:center; justify-content:center;">'
        '<h1 style="font-size:4.5rem;">CREDIT SCORING</h1>'
        '</div>',
        unsafe_allow_html=True
    )

    # Subtítulo
    st.write(
        '<div style="display:flex; align-items:center; justify-content:center;">'
        '<h2 style="font-size:2.5rem;">Prevendo possível inadimplência com Regressão Logística</h2>'
        '</div>',
        unsafe_allow_html=True
    )

    # Divisão
    st.write("---")

    # Imagem do lado da explicação
    col1, col2 = st.columns(2)

    col1.write(
        "<p style='font-size:1.5rem;'> Este aplicativo é uma ferramenta de algoritmo de Machine Learning que prevê se o cliente vai ser inadimplente com base nas suas características."
        "<br>"
        "Utiliza de um modelo de Regressão Logística para estimar o log da chance (logito) de uma pessoa se tornar inadimplentes a partir de características pessoais como posse e renda</p>"
        "<p style='font-size:1.5rem;'> Permite o carregamento de novos dados estimando se o possível cliente irá se tornar inadimplente. Além de demonstrar visualmente relações entre as variáveis pessoais e automatizar relatórios gerenciais dos"
        "conjuntos de dados.</p>",
    unsafe_allow_html=True
    )

    col2.write(
        '<div style="position:relative;"><iframe src="https://giphy.com/embed/U6LUBbHBdGHJnhIgvX" width="480" height="270" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></div>',
        unsafe_allow_html=True
    )

    # Divisão
    st.write("---")

    st.write(
        '<h3 style="text-align:left;">Autor</h3>'
        '<ul style="list-style-type: disc; margin-left: 20px;">'
        '<li>Caio Douglas Rodrigues de Paula</li>'
        '<li><a href="https://github.com/Caiodrp/Prever-Inadimplencia-ST">GitHub</a></li>'
        '</ul>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

