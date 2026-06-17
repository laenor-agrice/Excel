import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from io import BytesIO

# ==========================================
# CONFIGURAÇÃO DA PÁGINA E DESIGN FUTURISTA
# ==========================================
st.set_page_config(
    page_title="NexusData Analytics",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Injeção de CSS para um visual futurista (Cyberpunk/Neon Vibes)
st.markdown("""
    <style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #00ffcc !important;
        text-shadow: 0 0 5px #00ffcc;
        font-family: 'Courier New', Courier, monospace;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #58a6ff;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb;
        color: white !important;
        box-shadow: 0 0 10px #1f6feb;
    }
    .stButton>button {
        border: 1px solid #00ffcc;
        color: #00ffcc;
        background-color: transparent;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00ffcc;
        color: #000;
        box-shadow: 0 0 15px #00ffcc;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌌 NexusData Analytics")
st.markdown("Plataforma Avançada para Tratamento e Análise de Dados Tabulares.")

# ==========================================
# GERENCIAMENTO DE ESTADO (SESSION STATE)
# ==========================================
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None

# ==========================================
# ESTRUTURA DE ABAS
# ==========================================
tabs = st.tabs([
    "1. Cadastro", 
    "2. Upload", 
    "3. Tratamento", 
    "4. Estatística Temporal", 
    "5. Estatística Exp.", 
    "6. Gráficos", 
    "7. IA Analítica", 
    "8. Exportar"
])

# ------------------------------------------
# ABA 1: CADASTRO
# ------------------------------------------
with tabs[0]:
    st.header("👤 1. Identificação do Usuário")
    with st.form("cadastro_form"):
        col1, col2 = st.columns(2)
        with col1:
            nome = st.text_input("Nome Completo")
            cidade = st.text_input("Cidade")
            estado = st.text_input("Estado")
        with col2:
            perfil = st.selectbox("Perfil Profissional/Acadêmico", 
                                  ["Estudante", "Pesquisador (Mestrado/Doutorado)", "Professor", "Profissional da Indústria"])
        
        submit_cadastro = st.form_submit_button("Registrar no Sistema")
        
        if submit_cadastro:
            if nome:
                st.session_state.user_data = {'Nome': nome, 'Cidade': cidade, 'Estado': estado, 'Perfil': perfil}
                st.success(f"Bem-vindo(a) ao NexusData, {nome}! Siga para a aba de Upload.")
            else:
                st.error("Por favor, insira pelo menos o seu nome.")

# ------------------------------------------
# ABA 2: UPLOAD DE DADOS
# ------------------------------------------
with tabs[1]:
    st.header("📂 2. Carregamento de Dados")
    uploaded_file = st.file_uploader("Faça o upload do seu arquivo (CSV, TXT, XLSX)", type=['csv', 'txt', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.txt'):
                df = pd.read_csv(uploaded_file, sep='\t')
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df_raw = df.copy()
            st.session_state.df_clean = df.copy()
            st.success("Arquivo carregado com sucesso!")
            
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}")
            
    if st.session_state.df_raw is not None:
        st.subheader("Pré-visualização (10 primeiras linhas)")
        st.dataframe(st.session_state.df_raw.head(10), use_container_width=True)

# ------------------------------------------
# ABA 3: TRATAMENTO DE DADOS
# ------------------------------------------
with tabs[2]:
    st.header("⚙️ 3. Limpeza e Tratamento")
    if st.session_state.df_clean is not None:
        df_atual = st.session_state.df_clean
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Ações de Limpeza")
            if st.button("Remover células vazias (NaN)"):
                st.session_state.df_clean = df_atual.dropna()
                st.success("Linhas com NaN removidas.")
                st.rerun()
                
            if st.button("Remover colunas totalmente vazias"):
                st.session_state.df_clean = df_atual.dropna(axis=1, how='all')
                st.success("Colunas vazias removidas.")
                st.rerun()
                
            if st.button("Remover colunas duplicadas"):
                st.session_state.df_clean = df_atual.loc[:,~df_atual.columns.duplicated()]
                st.success("Colunas com nomes duplicados removidas.")
                st.rerun()
                
            if st.button("Remover linhas duplicadas"):
                st.session_state.df_clean = df_atual.drop_duplicates()
                st.success("Linhas idênticas removidas.")
                st.rerun()
                
            if st.button("Resetar Tratamento (Voltar ao original)"):
                st.session_state.df_clean = st.session_state.df_raw.copy()
                st.warning("Dados resetados para o formato original.")
                st.rerun()

        with col2:
            st.markdown("### Visualização Pós-Tratamento")
            st.markdown("Abaixo estão as 12 primeiras linhas formatadas para caber na tela.")
            # st.dataframe com use_container_width=True comprime/ajusta as colunas na tela
            st.dataframe(st.session_state.df_clean.head(12), use_container_width=True)
            st.info(f"Dimensões atuais: {st.session_state.df_clean.shape[0]} linhas x {st.session_state.df_clean.shape[1]} colunas")
    else:
        st.warning("Por favor, faça o upload dos dados na aba 2 primeiro.")

# ------------------------------------------
# ABA 4: ESTATÍSTICA TEMPORAL
# ------------------------------------------
with tabs[3]:
    st.header("📅 4. Agrupamentos e Médias Temporais")
    if st.session_state.df_clean is not None:
        df_temp = st.session_state.df_clean.copy()
        
        col_data = st.selectbox("Selecione a coluna que contém as Datas:", ["Nenhuma"] + list(df_temp.columns))
        
        if col_data != "Nenhuma":
            try:
                # Tenta converter para datetime
                df_temp[col_data] = pd.to_datetime(df_temp[col_data])
                df_temp = df_temp.set_index(col_data)
                
                # Seleciona apenas colunas numéricas para tirar média
                cols_numericas = df_temp.select_dtypes(include=[np.number]).columns
                
                if len(cols_numericas) > 0:
                    st.subheader("Médias Temporais Calculadas")
                    tab_w, tab_m, tab_y = st.tabs(["Semanal", "Mensal", "Anual"])
                    
                    with tab_w:
                        st.dataframe(df_temp[cols_numericas].resample('W').mean().dropna(), use_container_width=True)
                    with tab_m:
                        st.dataframe(df_temp[cols_numericas].resample('ME').mean().dropna(), use_container_width=True) # ME para Pandas 2.0+
                    with tab_y:
                        st.dataframe(df_temp[cols_numericas].resample('YE').mean().dropna(), use_container_width=True)
                else:
                    st.warning("Não há colunas numéricas para calcular médias temporais.")
            except Exception as e:
                st.error("A coluna selecionada não possui um formato de data válido.")
        else:
            st.info("Selecione uma coluna de data para ativar esta função.")
    else:
        st.warning("Dados não carregados.")

# ------------------------------------------
# ABA 5: ESTATÍSTICA EXPERIMENTAL
# ------------------------------------------
with tabs[4]:
    st.header("🔬 5. Estatística Descritiva Experimental")
    if st.session_state.df_clean is not None:
        df_num = st.session_state.df_clean.select_dtypes(include=[np.number])
        if not df_num.empty:
            st.markdown("Resumo estatístico padronizado na literatura científica (Média, Desvio Padrão, Mín/Máx, Quartis):")
            st.dataframe(df_num.describe().T, use_container_width=True)
            
            st.markdown("### Variância e Soma")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Variância (Var):**")
                st.dataframe(df_num.var(), use_container_width=True)
            with col2:
                st.write("**Soma Total:**")
                st.dataframe(df_num.sum(), use_container_width=True)
        else:
            st.warning("O dataset atual não possui colunas numéricas para análise experimental.")
    else:
        st.warning("Dados não carregados.")

# ------------------------------------------
# ABA 6: GRÁFICOS
# ------------------------------------------
with tabs[5]:
    st.header("📊 6. Visualização Gráfica Interativa")
    if st.session_state.df_clean is not None:
        df_plot = st.session_state.df_clean
        
        col1, col2, col3 = st.columns(3)
        with col1:
            tipo_grafico = st.selectbox("Tipo de Gráfico", ["Dispersão (Scatter)", "Linha", "Barras", "Boxplot", "Histograma"])
        with col2:
            eixo_x = st.selectbox("Eixo X", df_plot.columns)
        with col3:
            eixo_y = st.selectbox("Eixo Y", df_plot.columns)
            
        color_col = st.selectbox("Colorir por (Opcional):", ["Nenhum"] + list(df_plot.columns))
        color_param = None if color_col == "Nenhum" else color_col

        st.markdown("---")
        
        try:
            if tipo_grafico == "Dispersão (Scatter)":
                fig = px.scatter(df_plot, x=eixo_x, y=eixo_y, color=color_param, template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Cyan)
            elif tipo_grafico == "Linha":
                fig = px.line(df_plot, x=eixo_x, y=eixo_y, color=color_param, template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Cyan)
            elif tipo_grafico == "Barras":
                fig = px.bar(df_plot, x=eixo_x, y=eixo_y, color=color_param, template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Cyan)
            elif tipo_grafico == "Boxplot":
                fig = px.box(df_plot, x=eixo_x, y=eixo_y, color=color_param, template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Cyan)
            elif tipo_grafico == "Histograma":
                fig = px.histogram(df_plot, x=eixo_x, color=color_param, template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Cyan)
            
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Não foi possível gerar o gráfico com os parâmetros selecionados. Erro: {e}")
    else:
        st.warning("Dados não carregados.")

# ------------------------------------------
# ABA 7: IA ANALÍTICA
# ------------------------------------------
with tabs[6]:
    st.header("🧠 7. Assistente de IA para Interpretação")
    st.markdown("Neste módulo, a IA analisa o resumo dos seus dados e sugere insights.")
    
    if st.session_state.df_clean is not None:
        if st.button("Gerar Análise com IA"):
            # Aqui é um MOCK (Simulação) de como a IA funcionaria.
            # Para usar IA real, você integraria a API da OpenAI (import openai) ou Gemini aqui.
            with st.spinner("Conectando ao núcleo lógico da IA..."):
                df_num = st.session_state.df_clean.select_dtypes(include=[np.number])
                
                if not df_num.empty:
                    maior_media = df_num.mean().idxmax()
                    maior_desvio = df_num.std().idxmax()
                    
                    resposta_ia = f"""
                    **Análise Preliminar do NexusData (Simulação IA):**
                    
                    Olá! Analisei as {st.session_state.df_clean.shape[0]} linhas de dados. 
                    - Identifiquei que a coluna **'{maior_media}'** possui a maior média geral do seu dataset.
                    - A variável **'{maior_desvio}'** apresenta a maior dispersão (desvio padrão), indicando grande variabilidade nos seus experimentos ou medições nesta categoria.
                    
                    *Sugestão de Estudo:* Avalie no gráfico de Boxplot a coluna '{maior_desvio}' para verificar a presença de *outliers* (valores atípicos) que possam estar distorcendo seus resultados finais.
                    
                    *(Nota para o Desenvolvedor: Integre sua chave de API da OpenAI ou Google Gemini neste bloco de código para obter análises reais usando prompts dinâmicos baseados no `.describe()` do pandas).*
                    """
                    st.info(resposta_ia)
                else:
                    st.warning("A IA requer dados numéricos para gerar interpretações estatísticas.")
    else:
        st.warning("Faça o upload e trate os dados primeiro.")

# ------------------------------------------
# ABA 8: EXPORTAÇÃO
# ------------------------------------------
with tabs[7]:
    st.header("💾 8. Download de Dados Processados")
    if st.session_state.df_clean is not None:
        st.markdown("Exporte os dados que você tratou na plataforma para utilizá-los em seu artigo, tese ou relatório.")
        
        # Função para converter o dataframe em um arquivo Excel em memória
        def to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Dados_Tratados')
            processed_data = output.getvalue()
            return processed_data

        df_xlsx = to_excel(st.session_state.df_clean)
        
        st.download_button(
            label="📥 Baixar Dados Tratados (.xlsx)",
            data=df_xlsx,
            file_name='NexusData_Tratado.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
        st.markdown("---")
        if st.session_state.df_raw is not None:
            df_raw_xlsx = to_excel(st.session_state.df_raw)
            st.download_button(
                label="📥 Baixar Dados Originais (.xlsx)",
                data=df_raw_xlsx,
                file_name='NexusData_Original.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
    else:
        st.warning("Não há dados para exportar.")

# Rodapé
st.sidebar.markdown("### NexusData Analytics")
st.sidebar.caption("Versão 1.0 - Desenvolvido com Streamlit")
if st.session_state.user_data.get('Nome'):
    st.sidebar.success(f"Usuário ativo: {st.session_state.user_data.get('Nome')}")
