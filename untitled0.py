import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import requests
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA E DESIGN FUTURISTA
# ==========================================
st.set_page_config(
    page_title="NexusData Analytics",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeção de CSS para um visual futurista (Cyberpunk/Neon Vibes)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0d1117, #161b22, #0d1117);
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #00ffcc !important;
        text-shadow: 0 0 8px rgba(0, 255, 204, 0.5);
        font-family: 'Courier New', Courier, monospace;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.02);
        padding: 8px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1c2128;
        border-radius: 6px;
        padding: 10px 20px;
        color: #58a6ff;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white !important;
        box-shadow: 0 0 15px rgba(31, 111, 235, 0.5);
        border: 1px solid #58a6ff;
    }
    .stButton>button {
        background: transparent;
        border: 1px solid #00ffcc;
        color: #00ffcc;
        border-radius: 30px;
        padding: 10px 25px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00ffcc;
        color: #000;
        box-shadow: 0 0 15px #00ffcc;
        transform: translateY(-2px);
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(0, 255, 204, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNÇÕES DE SUPORTE (LEITURA E IA)
# ==========================================
def ler_arquivo_robusto(arquivo_bytes, nome_arquivo):
    """Lê CSV, TXT ou Excel lidando com diferentes separadores."""
    try:
        if nome_arquivo.endswith('.csv') or nome_arquivo.endswith('.txt'):
            # Tenta ler com separador vírgula, ponto-e-vírgula ou tab
            df = pd.read_csv(arquivo_bytes, sep=None, engine='python')
        else:
            df = pd.read_excel(arquivo_bytes)
        return df
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None

def consultar_gemini(api_key, prompt, contexto):
    """Consulta a API real do Gemini (baseada no seu código original)."""
    if not api_key:
        return "⚠️ Chave da API (API Key) não fornecida. Insira no menu lateral."
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
        prompt_completo = f"""Você é um cientista de dados e estatístico sênior. Analise o resumo do conjunto de dados abaixo.
        CONTEXTO DOS DADOS (Resumo Estatístico):
        {contexto}
        
        O QUE O USUÁRIO DESEJA SABER:
        {prompt}
        """
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt_completo}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048}
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            resultado = response.json()
            if "candidates" in resultado and resultado["candidates"]:
                return resultado["candidates"][0]["content"]["parts"][0]["text"]
        return f"Erro na análise. Status: {response.status_code}"
    except Exception as e:
        return f"Erro na requisição da IA: {e}"

# ==========================================
# 3. GERENCIAMENTO DE ESTADO
# ==========================================
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None

# ==========================================
# 4. INTERFACE PRINCIPAL
# ==========================================
st.title("🌌 NexusData Analytics")
st.markdown("Plataforma Interativa de Processamento Estatístico de Dados Tabulares")

# Sidebar para configurações
with st.sidebar:
    st.markdown("### ⚙️ Configurações do Sistema")
    api_key = st.text_input("Chave da API Gemini (Opcional para IA)", type="password", help="Insira sua chave para ativar a aba de IA.")
    st.markdown("---")
    if st.session_state.user_data.get('Nome'):
        st.success(f"👤 Ativo: {st.session_state.user_data.get('Nome')}")
        st.info(f"📍 Perfil: {st.session_state.user_data.get('Perfil')}")

# Estrutura de Abas
tabs = st.tabs([
    "1. Cadastro", 
    "2. Upload", 
    "3. Tratamento", 
    "4. Est. Temporal", 
    "5. Est. Experimental", 
    "6. Gráficos", 
    "7. IA Analítica", 
    "8. Exportar"
])

# --- ABA 1: CADASTRO ---
with tabs[0]:
    st.header("👤 1. Identificação do Pesquisador")
    with st.form("cadastro_form"):
        col1, col2 = st.columns(2)
        with col1:
            nome = st.text_input("Nome Completo", value=st.session_state.user_data.get('Nome', ''))
            cidade = st.text_input("Cidade", value=st.session_state.user_data.get('Cidade', ''))
            estado = st.text_input("Estado", value=st.session_state.user_data.get('Estado', ''))
        with col2:
            perfis = ["Estudante", "Pesquisador (Mestrado/Doutorado)", "Aluno", "Professor"]
            perfil = st.selectbox("Perfil Acadêmico/Profissional", perfis)
        
        if st.form_submit_button("Acessar Plataforma"):
            if nome:
                st.session_state.user_data = {'Nome': nome, 'Cidade': cidade, 'Estado': estado, 'Perfil': perfil}
                st.success("Cadastro confirmado! Siga para a aba de Upload.")
            else:
                st.error("Insira seu nome para continuar.")

# --- ABA 2: UPLOAD ---
with tabs[1]:
    st.header("📂 2. Carregamento de Dados (Excel, CSV, TXT)")
    uploaded_file = st.file_uploader("Arraste ou selecione seu arquivo", type=['csv', 'txt', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        with st.spinner("Lendo e detectando formato do arquivo..."):
            df = ler_arquivo_robusto(uploaded_file, uploaded_file.name)
            if df is not None:
                st.session_state.df_raw = df.copy()
                st.session_state.df_clean = df.copy()
                st.success("Arquivo processado com sucesso!")
                
    if st.session_state.df_raw is not None:
        st.subheader("Visualização Prévia (Top 10)")
        st.dataframe(st.session_state.df_raw.head(10), use_container_width=True)

# --- ABA 3: TRATAMENTO ---
with tabs[2]:
    st.header("⚙️ 3. Limpeza e Tratamento")
    if st.session_state.df_clean is not None:
        df_atual = st.session_state.df_clean
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Ações de Limpeza")
            if st.button("🗑️ Remover valores nulos (NaN)"):
                st.session_state.df_clean = df_atual.dropna()
                st.rerun()
            if st.button("🗑️ Remover colunas vazias"):
                st.session_state.df_clean = df_atual.dropna(axis=1, how='all')
                st.rerun()
            if st.button("👥 Remover linhas inconsistentes (Duplicadas)"):
                st.session_state.df_clean = df_atual.drop_duplicates()
                st.rerun()
            if st.button("🔄 Resetar para Original"):
                st.session_state.df_clean = st.session_state.df_raw.copy()
                st.rerun()

        with col2:
            st.markdown("### Visualização de Dados Tratados")
            st.caption(f"Status Atual: {st.session_state.df_clean.shape[0]} linhas | {st.session_state.df_clean.shape[1]} colunas")
            # Tabela espremida mostrando apenas 12 linhas conforme solicitado
            st.dataframe(st.session_state.df_clean.head(12), use_container_width=True, height=450)
    else:
        st.warning("Aguardando upload de dados...")

# --- ABA 4: ESTATÍSTICA TEMPORAL ---
with tabs[3]:
    st.header("📅 4. Estatística Temporal (Agrupamentos)")
    if st.session_state.df_clean is not None:
        df_temp = st.session_state.df_clean.copy()
        col_data = st.selectbox("Selecione a coluna que contém as Datas:", ["Nenhuma"] + list(df_temp.columns))
        
        if col_data != "Nenhuma":
            try:
                df_temp[col_data] = pd.to_datetime(df_temp[col_data])
                df_temp = df_temp.set_index(col_data)
                cols_numericas = df_temp.select_dtypes(include=[np.number]).columns
                
                if len(cols_numericas) > 0:
                    t1, t2, t3 = st.tabs(["Média Semanal", "Média Mensal", "Média Anual"])
                    with t1: st.dataframe(df_temp[cols_numericas].resample('W').mean().dropna(), use_container_width=True)
                    with t2: st.dataframe(df_temp[cols_numericas].resample('ME').mean().dropna(), use_container_width=True)
                    with t3: st.dataframe(df_temp[cols_numericas].resample('YE').mean().dropna(), use_container_width=True)
                else:
                    st.warning("Não há colunas numéricas para calcular médias.")
            except:
                st.error("A coluna selecionada não parece ser um formato de data válido.")
        else:
            st.info("Para ativar, indique qual coluna guarda os dados temporais.")
    else:
        st.warning("Aguardando upload de dados...")

# --- ABA 5: ESTATÍSTICA EXPERIMENTAL ---
with tabs[4]:
    st.header("🔬 5. Tratamento de Dados de Experimento")
    if st.session_state.df_clean is not None:
        df_num = st.session_state.df_clean.select_dtypes(include=[np.number])
        if not df_num.empty:
            st.markdown("### Resumo Estatístico da Literatura")
            st.dataframe(df_num.describe().T, use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Variância")
                st.dataframe(df_num.var(), use_container_width=True)
            with c2:
                st.markdown("#### Soma Total")
                st.dataframe(df_num.sum(), use_container_width=True)
        else:
            st.warning("Não foram encontradas colunas numéricas para análise estatística.")
    else:
        st.warning("Aguardando upload de dados...")

# --- ABA 6: GRÁFICOS INTERATIVOS ---
with tabs[5]:
    st.header("📊 6. Painel Gráfico Futurista")
    if st.session_state.df_clean is not None:
        df_plot = st.session_state.df_clean
        c1, c2, c3 = st.columns(3)
        with c1: tipo = st.selectbox("Formato", ["Dispersão", "Linha", "Barras", "Boxplot"])
        with c2: eixo_x = st.selectbox("Eixo X", df_plot.columns)
        with c3: eixo_y = st.selectbox("Eixo Y", df_plot.columns)
        
        try:
            if tipo == "Dispersão": fig = px.scatter(df_plot, x=eixo_x, y=eixo_y, template="plotly_dark", color_discrete_sequence=['#00ffcc'])
            elif tipo == "Linha": fig = px.line(df_plot, x=eixo_x, y=eixo_y, template="plotly_dark", color_discrete_sequence=['#ff00ff'])
            elif tipo == "Barras": fig = px.bar(df_plot, x=eixo_x, y=eixo_y, template="plotly_dark", color_discrete_sequence=['#1f6feb'])
            elif tipo == "Boxplot": fig = px.box(df_plot, x=eixo_x, y=eixo_y, template="plotly_dark", color_discrete_sequence=['#ff9900'])
            
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error("Não foi possível renderizar este gráfico. Verifique os tipos de dados dos eixos.")
    else:
        st.warning("Aguardando upload de dados...")

# --- ABA 7: IA ANALÍTICA (GEMINI) ---
with tabs[6]:
    st.header("🧠 7. Interpretação com Inteligência Artificial")
    if st.session_state.df_clean is not None:
        st.markdown("Descreva o que você gostaria que a IA interpretasse sobre a **estatística dos seus dados**.")
        prompt_user = st.text_area("Exemplo: 'Qual coluna apresenta maior variabilidade e o que isso significa para o experimento?'")
        
        if st.button("Gerar Interpretação"):
            if not api_key:
                st.error("Insira sua chave da API Gemini no menu lateral para usar esta função.")
            else:
                with st.spinner("Conectando aos servidores do Google Gemini..."):
                    df_num = st.session_state.df_clean.select_dtypes(include=[np.number])
                    contexto_estatistico = df_num.describe().to_string()
                    resposta = consultar_gemini(api_key, prompt_user, contexto_estatistico)
                    st.info(resposta)
    else:
        st.warning("Aguardando upload de dados...")

# --- ABA 8: EXPORTAR ---
with tabs[7]:
    st.header("💾 8. Download e Exportação")
    if st.session_state.df_clean is not None:
        st.markdown("Baixe seus dados finais já tratados ou recupere o histórico original.")
        
        def convert_df_to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='NexusData')
            return output.getvalue()
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Baixar Dados Tratados (.xlsx)",
                data=convert_df_to_excel(st.session_state.df_clean),
                file_name='NexusData_Tratados.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        with col2:
            st.download_button(
                label="📥 Baixar Dados Originais (.xlsx)",
                data=convert_df_to_excel(st.session_state.df_raw),
                file_name='NexusData_Originais.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
    else:
        st.warning("Aguardando upload de dados...")
