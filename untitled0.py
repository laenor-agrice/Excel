import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import datetime
import time
import re
import math
import warnings

# Ignorar avisos de depreciação de bibliotecas
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E CSS FUTURISTA (CYBERPUNK/NEON)
# ============================================================================
st.set_page_config(
    page_title="NexusData Analytics Pro",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    """Applies advanced CSS for a futuristic, neon UI."""
    st.markdown("""
    <style>
        /* Variáveis de Cores Futuristas */
        :root {
            --bg-color: #050510;
            --panel-bg: #0b0f19;
            --primary-neon: #00ffcc;
            --secondary-neon: #ff00ff;
            --text-main: #e0e6ed;
            --text-muted: #8b9bb4;
            --border-color: rgba(0, 255, 204, 0.3);
            --danger: #ff3366;
            --success: #33ff99;
        }

        /* Fundo Geral */
        .stApp {
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(0, 255, 204, 0.05), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(255, 0, 255, 0.05), transparent 25%);
            color: var(--text-main);
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }

        /* Cabeçalhos */
        h1, h2, h3, h4, h5, h6 {
            color: var(--primary-neon) !important;
            text-transform: uppercase;
            letter-spacing: 2px;
            text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
            font-weight: 700;
        }

        /* Abas (Tabs) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: var(--panel-bg);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid var(--border-color);
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.5);
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border: 1px solid var(--text-muted);
            border-radius: 5px;
            padding: 10px 20px;
            color: var(--text-muted);
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(0, 255, 204, 0.1);
            border: 1px solid var(--primary-neon) !important;
            color: var(--primary-neon) !important;
            box-shadow: 0 0 15px rgba(0, 255, 204, 0.3);
            text-shadow: 0 0 5px var(--primary-neon);
        }

        /* Botões */
        .stButton > button {
            background: linear-gradient(45deg, transparent, rgba(0, 255, 204, 0.1));
            border: 1px solid var(--primary-neon);
            color: var(--primary-neon);
            border-radius: 4px;
            padding: 10px 24px;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton > button:hover {
            background: var(--primary-neon);
            color: #000;
            box-shadow: 0 0 20px rgba(0, 255, 204, 0.6);
            transform: translateY(-2px);
        }

        /* Inputs e Selects */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div,
        .stNumberInput > div > div > input {
            background-color: rgba(11, 15, 25, 0.8) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--primary-neon) !important;
            border-radius: 4px;
        }
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > div:focus {
            box-shadow: 0 0 10px rgba(0, 255, 204, 0.5) !important;
            border-color: var(--primary-neon) !important;
        }

        /* Dataframes */
        .stDataFrame {
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 5px;
            background-color: var(--panel-bg);
            box-shadow: 0 0 15px rgba(0,0,0,0.5);
        }

        /* Cards de Métrica */
        [data-testid="stMetricValue"] {
            color: var(--secondary-neon) !important;
            text-shadow: 0 0 10px rgba(255, 0, 255, 0.4);
        }
        [data-testid="stMetricLabel"] {
            color: var(--text-muted) !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: var(--panel-bg);
            border-right: 1px solid var(--border-color);
        }
        
        /* Upload Area */
        .stFileUploader > div > div {
            background-color: rgba(0, 255, 204, 0.05);
            border: 2px dashed var(--border-color);
            border-radius: 10px;
        }
        .stFileUploader > div > div:hover {
            border-color: var(--primary-neon);
            background-color: rgba(0, 255, 204, 0.1);
        }

        /* Mensagens de Sucesso/Aviso */
        .stSuccess {
            background-color: rgba(51, 255, 153, 0.1) !important;
            color: var(--success) !important;
            border-left: 5px solid var(--success) !important;
        }
        .stWarning {
            background-color: rgba(255, 204, 0, 0.1) !important;
            color: #ffcc00 !important;
            border-left: 5px solid #ffcc00 !important;
        }
        .stError {
            background-color: rgba(255, 51, 102, 0.1) !important;
            color: var(--danger) !important;
            border-left: 5px solid var(--danger) !important;
        }
        .stInfo {
            background-color: rgba(0, 255, 204, 0.1) !important;
            color: var(--primary-neon) !important;
            border-left: 5px solid var(--primary-neon) !important;
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# ============================================================================
# 2. GERENCIAMENTO DE ESTADO (SESSION STATE)
# ============================================================================
def init_session_state():
    """Initializes all required session state variables to prevent KeyErrors."""
    state_vars = {
        'user_registered': False,
        'user_info': {},
        'df_raw': None,
        'df_clean': None,
        'df_stats_temporal': None,
        'logs': [],
        'filename': '',
        'file_ext': ''
    }
    for var, default in state_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

init_session_state()

def log_action(action_msg):
    """Adds an action to the internal log."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {action_msg}")

# ============================================================================
# 3. FUNÇÕES UTILITÁRIAS E DE PROCESSAMENTO DE DADOS
# ============================================================================
def load_data(uploaded_file):
    """Loads data from CSV, TXT, or Excel files."""
    try:
        filename = uploaded_file.name.lower()
        st.session_state.filename = uploaded_file.name
        
        if filename.endswith('.csv'):
            # Tenta ler com diferentes separadores
            try:
                df = pd.read_csv(uploaded_file)
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';')
            st.session_state.file_ext = 'csv'
            
        elif filename.endswith('.txt'):
            try:
                df = pd.read_csv(uploaded_file, sep='\t')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=None, engine='python')
            st.session_state.file_ext = 'txt'
            
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
            st.session_state.file_ext = 'xlsx'
        else:
            return None, "Formato de arquivo não suportado."
        
        return df, None
    except Exception as e:
        return None, str(e)

def clean_data_nan(df):
    """Removes rows with NaN values."""
    initial_shape = df.shape
    df_cleaned = df.dropna()
    final_shape = df_cleaned.shape
    return df_cleaned, initial_shape[0] - final_shape[0]

def clean_data_empty_cols(df):
    """Removes columns that are entirely empty."""
    initial_cols = df.shape[1]
    df_cleaned = df.dropna(axis=1, how='all')
    final_cols = df_cleaned.shape[1]
    return df_cleaned, initial_cols - final_cols

def clean_inconsistent_duplicates(df):
    """Removes exact duplicate rows."""
    initial_shape = df.shape
    df_cleaned = df.drop_duplicates()
    final_shape = df_cleaned.shape
    return df_cleaned, initial_shape[0] - final_shape[0]

def convert_to_numeric(df):
    """Attempts to convert columns to numeric, coercing errors."""
    df_num = df.copy()
    converted_cols = []
    for col in df_num.columns:
        if df_num[col].dtype == 'object':
            try:
                # Substitui vírgulas por pontos antes da conversão
                s = df_num[col].astype(str).str.replace(',', '.')
                df_num[col] = pd.to_numeric(s, errors='coerce')
                converted_cols.append(col)
            except:
                pass
    return df_num, converted_cols

# ============================================================================
# 4. FUNÇÕES DE ESTATÍSTICA
# ============================================================================
def calculate_temporal_stats(df, date_col, numeric_cols, freq):
    """Calculates mean stats based on time frequency."""
    try:
        df_temp = df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
        df_temp = df_temp.dropna(subset=[date_col])
        df_temp.set_index(date_col, inplace=True)
        
        # 'W' para Semanal, 'ME' para Mensal, 'YE' para Anual
        df_resampled = df_temp[numeric_cols].resample(freq).mean()
        return df_resampled.reset_index(), None
    except Exception as e:
        return None, str(e)

def experimental_statistics(df, numeric_cols):
    """Calculates conventional experimental statistics."""
    stats_dict = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 0:
            stats_dict[col] = {
                'Média': series.mean(),
                'Mediana': series.median(),
                'Desvio Padrão': series.std(),
                'Variância': series.var(),
                'Mínimo': series.min(),
                'Máximo': series.max(),
                'Assimetria (Skew)': series.skew(),
                'Curtose (Kurt)': series.kurtosis(),
                'Erro Padrão': series.sem(),
                'Contagem Valida': len(series)
            }
    return pd.DataFrame(stats_dict).T

# ============================================================================
# 5. MÓDULO DE GRÁFICOS INTERATIVOS (USANDO PLOTLY GRAPH_OBJECTS)
# ============================================================================
def create_scatter_plot(df, x_col, y_col, color_col=None):
    fig = go.Figure()
    
    if color_col and color_col != "Nenhum":
        categories = df[color_col].unique()
        for cat in categories:
            df_sub = df[df[color_col] == cat]
            fig.add_trace(go.Scatter(
                x=df_sub[x_col], y=df_sub[y_col],
                mode='markers', name=str(cat),
                marker=dict(size=8, opacity=0.7, line=dict(width=1, color='white'))
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[y_col], mode='markers',
            marker=dict(color='#00ffcc', size=8, opacity=0.7, line=dict(width=1, color='white')),
            name='Dados'
        ))

    fig.update_layout(
        title=f'Dispersão: {y_col} vs {x_col}',
        xaxis_title=x_col, yaxis_title=y_col,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e6ed')
    )
    return fig

def create_line_plot(df, x_col, y_col, color_col=None):
    fig = go.Figure()
    
    df_sorted = df.sort_values(by=x_col)
    
    if color_col and color_col != "Nenhum":
        categories = df_sorted[color_col].unique()
        for cat in categories:
            df_sub = df_sorted[df_sorted[color_col] == cat]
            fig.add_trace(go.Scatter(
                x=df_sub[x_col], y=df_sub[y_col],
                mode='lines+markers', name=str(cat),
                line=dict(width=2)
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df_sorted[x_col], y=df_sorted[y_col], 
            mode='lines+markers', name='Série',
            line=dict(color='#ff00ff', width=2),
            marker=dict(size=6, color='#00ffcc')
        ))

    fig.update_layout(
        title=f'Evolução de {y_col} por {x_col}',
        xaxis_title=x_col, yaxis_title=y_col,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_bar_plot(df, x_col, y_col, color_col=None):
    fig = go.Figure()
    
    if color_col and color_col != "Nenhum":
        # Group data for stacked/grouped bar
        grouped = df.groupby([x_col, color_col])[y_col].mean().reset_index()
        categories = grouped[color_col].unique()
        for cat in categories:
            df_sub = grouped[grouped[color_col] == cat]
            fig.add_trace(go.Bar(
                x=df_sub[x_col], y=df_sub[y_col], name=str(cat)
            ))
        fig.update_layout(barmode='group')
    else:
        # Agrupa média por X
        grouped = df.groupby(x_col)[y_col].mean().reset_index()
        fig.add_trace(go.Bar(
            x=grouped[x_col], y=grouped[y_col],
            marker_color='#00ffcc',
            marker_line_color='white',
            marker_line_width=1.5,
            opacity=0.8
        ))

    fig.update_layout(
        title=f'Média de {y_col} por {x_col}',
        xaxis_title=x_col, yaxis_title=y_col,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_boxplot(df, y_col, x_col=None):
    fig = go.Figure()
    
    if x_col and x_col != "Nenhum":
        categories = df[x_col].unique()
        for cat in categories:
            df_sub = df[df[x_col] == cat]
            fig.add_trace(go.Box(
                y=df_sub[y_col], name=str(cat),
                boxpoints='outliers',
                line=dict(width=2)
            ))
    else:
        fig.add_trace(go.Box(
            y=df[y_col], name=y_col,
            marker_color='#ff00ff',
            boxpoints='outliers'
        ))

    fig.update_layout(
        title=f'Distribuição de {y_col}',
        yaxis_title=y_col,
        xaxis_title=x_col if x_col != "Nenhum" else '',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_correlation_heatmap(df, numeric_cols):
    fig = go.Figure()
    if len(numeric_cols) < 2:
        return fig
    
    corr_matrix = df[numeric_cols].corr().round(2)
    
    fig.add_trace(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',
        text=corr_matrix.values,
        texttemplate='%{text}',
        hoverinfo='z'
    ))
    
    fig.update_layout(
        title='Matriz de Correlação Global',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=600
    )
    return fig

# ============================================================================
# 6. ESTRUTURA PRINCIPAL E ABAS DA INTERFACE
# ============================================================================
def main():
    st.markdown('<h1 style="text-align: center; margin-bottom: 30px;">🌌 NexusData Analytics Pro</h1>', unsafe_allow_html=True)
    
    # Menu lateral
    with st.sidebar:
        st.markdown('### 📡 Sistema de Conexão')
        if st.session_state.user_registered:
            st.success(f"Logado como: {st.session_state.user_info.get('Nome', 'Usuário')}")
            st.info(f"Perfil: {st.session_state.user_info.get('Perfil', 'N/A')}")
            if st.button("Sair do Sistema", key="logout_btn"):
                st.session_state.user_registered = False
                st.rerun()
        else:
            st.warning("Usuário não identificado. Realize o cadastro na Aba 1.")
            
        st.markdown("---")
        st.markdown("### 📜 Logs do Sistema")
        if st.session_state.logs:
            with st.expander("Ver Logs"):
                for log in st.session_state.logs[-10:]: # Mostra os 10 últimos
                    st.caption(log)
        else:
            st.caption("Nenhum registro ainda.")

    # Container das Abas
    tabs = st.tabs([
        "1️⃣ Cadastro", 
        "2️⃣ Upload", 
        "3️⃣ Tratamento", 
        "4️⃣ Est. Temporal", 
        "5️⃣ Est. Experimental", 
        "6️⃣ Gráficos", 
        "7️⃣ IA Intérprete", 
        "8️⃣ Exportação"
    ])

    # ------------------------------------------------------------------------
    # ABA 1: CADASTRO DO USUÁRIO
    # ------------------------------------------------------------------------
    with tabs[0]:
        st.markdown("### 👤 Identificação do Pesquisador")
        st.markdown("Registre-se no terminal Nexus para acessar os módulos de processamento profundo.")
        
        if not st.session_state.user_registered:
            with st.form("form_cadastro"):
                col1, col2 = st.columns(2)
                with col1:
                    nome = st.text_input("Nome Completo *")
                    email = st.text_input("E-mail de Contato")
                    cidade = st.text_input("Cidade")
                    estado = st.text_input("Estado (UF)", max_chars=2)
                with col2:
                    perfil = st.selectbox("Selecione seu Perfil *", 
                        ["", "Estudante", "Pesquisador (Mestrado/Doutorado)", "Aluno", "Professor", "Cientista de Dados"])
                    instituicao = st.text_input("Instituição de Origem")
                    
                st.markdown("*( * ) Campos Obrigatórios*")
                submit = st.form_submit_button("Inicializar Sessão")
                
                if submit:
                    if nome and perfil:
                        st.session_state.user_info = {
                            "Nome": nome, "Email": email, "Cidade": cidade, 
                            "Estado": estado.upper(), "Perfil": perfil, "Instituicao": instituicao
                        }
                        st.session_state.user_registered = True
                        log_action(f"Usuário '{nome}' registrado com sucesso.")
                        st.success("Sessão inicializada com sucesso! Proceda para a Aba 2 (Upload).")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Falha na inicialização: Nome e Perfil são obrigatórios.")
        else:
            st.info("Sessão ativa. Você já possui acesso liberado aos módulos.")
            col_info1, col_info2 = st.columns(2)
            for key, val in st.session_state.user_info.items():
                col_info1.write(f"**{key}:**")
                col_info2.write(f"{val}")

    # ------------------------------------------------------------------------
    # ABA 2: UPLOAD DE DADOS
    # ------------------------------------------------------------------------
    with tabs[1]:
        st.markdown("### 📂 Módulo de Ingestão de Dados")
        
        if not st.session_state.user_registered:
            st.warning("⚠️ Acesso Restrito: Conclua o cadastro na Aba 1 para enviar arquivos.")
        else:
            st.markdown("Faça o upload do seu dataset estruturado. Formatos suportados: **CSV, TXT, XLSX, XLS**.")
            
            uploaded_file = st.file_uploader("", type=['csv', 'txt', 'xlsx', 'xls'])
            
            if uploaded_file is not None:
                # Botão para carregar para evitar recarregamento automático a cada clique
                if st.button("Processar Arquivo"):
                    with st.spinner("Decodificando matriz de dados..."):
                        df, error = load_data(uploaded_file)
                        
                        if error:
                            st.error(f"Erro na decodificação: {error}")
                            log_action(f"Erro ao ler arquivo: {error}")
                        else:
                            st.session_state.df_raw = df.copy()
                            st.session_state.df_clean = df.copy()
                            log_action(f"Arquivo '{uploaded_file.name}' carregado. Matriz: {df.shape}")
                            st.success("Matriz de dados carregada na memória com sucesso.")
            
            if st.session_state.df_raw is not None:
                st.markdown("#### Pré-visualização da Matriz Original (10 primeiras linhas)")
                st.dataframe(st.session_state.df_raw.head(10), use_container_width=True)
                
                col1, col2 = st.columns(2)
                col1.metric("Linhas Detectadas", st.session_state.df_raw.shape[0])
                col2.metric("Colunas Detectadas", st.session_state.df_raw.shape[1])

    # ------------------------------------------------------------------------
    # ABA 3: TRATAMENTO DE DADOS
    # ------------------------------------------------------------------------
    with tabs[2]:
        st.markdown("### ⚙️ Engine de Limpeza e Estruturação")
        
        if st.session_state.df_clean is None:
            st.warning("Nenhum dado detectado na memória principal. Retorne à Aba 2.")
        else:
            df_atual = st.session_state.df_clean
            
            col_tools, col_view = st.columns([1, 2])
            
            with col_tools:
                st.markdown("#### Ferramentas de Purificação")
                
                if st.button("🗑️ Remover Células Vazias (NaN)"):
                    novo_df, excluidas = clean_data_nan(df_atual)
                    st.session_state.df_clean = novo_df
                    log_action(f"Removidas {excluidas} linhas com NaN.")
                    st.success(f"{excluidas} anomalias removidas.")
                    time.sleep(0.5); st.rerun()
                
                if st.button("📐 Remover Colunas Vazias"):
                    novo_df, excluidas = clean_data_empty_cols(df_atual)
                    st.session_state.df_clean = novo_df
                    log_action(f"Removidas {excluidas} colunas totalmente vazias.")
                    st.success(f"{excluidas} colunas eliminadas.")
                    time.sleep(0.5); st.rerun()
                    
                if st.button("🪞 Remover Linhas Duplicadas"):
                    novo_df, excluidas = clean_inconsistent_duplicates(df_atual)
                    st.session_state.df_clean = novo_df
                    log_action(f"Removidas {excluidas} linhas duplicadas (inconsistentes).")
                    st.success(f"{excluidas} redundâncias apagadas.")
                    time.sleep(0.5); st.rerun()
                    
                if st.button("🔢 Forçar Numérico (Auto)"):
                    novo_df, colunas_conv = convert_to_numeric(df_atual)
                    st.session_state.df_clean = novo_df
                    if colunas_conv:
                        msg = f"Colunas convertidas para numérico: {', '.join(colunas_conv)}"
                        log_action(msg)
                        st.success(msg)
                    else:
                        st.info("Nenhuma conversão necessária ou possível.")
                    time.sleep(1); st.rerun()

                st.markdown("---")
                if st.button("♻️ Resetar Matriz ao Original"):
                    st.session_state.df_clean = st.session_state.df_raw.copy()
                    log_action("Matriz limpa resetada para o estado original.")
                    st.warning("Revertido ao estado primordial.")
                    time.sleep(0.5); st.rerun()

            with col_view:
                st.markdown("#### Visualização da Matriz Processada (12 linhas)")
                st.markdown("O sistema auto-ajusta as colunas para caber na dimensão do terminal.")
                # use_container_width=True resolve o requisito de manter e reduzir para caber no campo
                st.dataframe(st.session_state.df_clean.head(12), use_container_width=True)
                
                st.info(f"Estado Atual: {st.session_state.df_clean.shape[0]} instâncias | {st.session_state.df_clean.shape[1]} atributos")

    # ------------------------------------------------------------------------
    # ABA 4: ESTATÍSTICA TEMPORAL
    # ------------------------------------------------------------------------
    with tabs[3]:
        st.markdown("### ⏳ Agrupamento e Média Temporal")
        
        if st.session_state.df_clean is None:
            st.warning("Inicie a ingestão de dados primeiro.")
        else:
            df = st.session_state.df_clean
            colunas = df.columns.tolist()
            colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not colunas_numericas:
                st.error("A matriz não possui colunas numéricas para calcular médias. Use a ferramenta 'Forçar Numérico' na Aba 3.")
            else:
                st.markdown("Selecione a coluna que atua como vetor temporal (Data/Hora):")
                col_data = st.selectbox("Vetor Temporal", ["Selecione..."] + colunas)
                
                if col_data != "Selecione...":
                    if st.button("Calcular Agregações Temporais"):
                        with st.spinner("Processando séries temporais..."):
                            df_sem, erro_sem = calculate_temporal_stats(df, col_data, colunas_numericas, 'W')
                            df_men, erro_men = calculate_temporal_stats(df, col_data, colunas_numericas, 'ME')
                            df_anu, erro_anu = calculate_temporal_stats(df, col_data, colunas_numericas, 'YE')
                            
                            if erro_men:
                                st.error(f"Falha ao interpretar a coluna '{col_data}' como Data. Detalhes: {erro_men}")
                            else:
                                log_action(f"Estatísticas temporais geradas baseadas em '{col_data}'.")
                                st.session_state.df_stats_temporal = df_men # Salva mensal em memória opcional
                                
                                tab_w, tab_m, tab_y = st.tabs(["Média Semanal", "Média Mensal", "Média Anual"])
                                
                                with tab_w:
                                    st.dataframe(df_sem.dropna(how='all', subset=colunas_numericas).head(50), use_container_width=True)
                                with tab_m:
                                    st.dataframe(df_men.dropna(how='all', subset=colunas_numericas).head(50), use_container_width=True)
                                with tab_y:
                                    st.dataframe(df_anu.dropna(how='all', subset=colunas_numericas).head(50), use_container_width=True)

    # ------------------------------------------------------------------------
    # ABA 5: ESTATÍSTICA CONVENCIONAL DE EXPERIMENTO
    # ------------------------------------------------------------------------
    with tabs[4]:
        st.markdown("### 🔬 Análise Descritiva Experimental (Literatura)")
        
        if st.session_state.df_clean is None:
            st.warning("Base de dados inexistente.")
        else:
            df = st.session_state.df_clean
            colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(colunas_numericas) > 0:
                st.markdown("Quadro de resumo estatístico profundo para laudos e artigos científicos.")
                
                df_exp = experimental_statistics(df, colunas_numericas)
                st.dataframe(df_exp, use_container_width=True)
                
                st.markdown("#### Matriz de Correlação Global (Métrica de Pearson)")
                # Exibindo como dataframe para visão tabelada antes do gráfico
                st.dataframe(df[colunas_numericas].corr().round(4), use_container_width=True)
                
            else:
                st.error("Operação abortada: Ausência de escalares numéricos na matriz limpa.")

    # ------------------------------------------------------------------------
    # ABA 6: GRÁFICOS INTERATIVOS
    # ------------------------------------------------------------------------
    with tabs[5]:
        st.markdown("### 📈 Visualização Interativa do Tratamento")
        
        if st.session_state.df_clean is None:
            st.warning("Sem dados para plotagem.")
        else:
            df = st.session_state.df_clean
            colunas = df.columns.tolist()
            colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
            colunas_categoricas = df.select_dtypes(exclude=[np.number]).columns.tolist()
            
            col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
            with col_cfg1:
                tipo_grafico = st.selectbox("Motor Gráfico", ["Dispersão (Scatter)", "Linhas", "Barras", "Boxplot", "Mapa de Calor (Corr)"])
            with col_cfg2:
                eixo_x = st.selectbox("Eixo X", colunas)
            with col_cfg3:
                eixo_y = st.selectbox("Eixo Y", colunas_numericas if colunas_numericas else colunas)
            with col_cfg4:
                cor_agrup = st.selectbox("Agrupar por (Cor)", ["Nenhum"] + colunas_categoricas + colunas_numericas)
                
            st.markdown("---")
            
            try:
                if st.button("Gerar Renderização"):
                    with st.spinner("Renderizando visualização avançada com Plotly Graph Objects..."):
                        if tipo_grafico == "Dispersão (Scatter)":
                            fig = create_scatter_plot(df, eixo_x, eixo_y, cor_agrup)
                        elif tipo_grafico == "Linhas":
                            fig = create_line_plot(df, eixo_x, eixo_y, cor_agrup)
                        elif tipo_grafico == "Barras":
                            fig = create_bar_plot(df, eixo_x, eixo_y, cor_agrup)
                        elif tipo_grafico == "Boxplot":
                            fig = create_boxplot(df, eixo_y, cor_agrup if cor_agrup != "Nenhum" else None)
                        elif tipo_grafico == "Mapa de Calor (Corr)":
                            fig = create_correlation_heatmap(df, colunas_numericas)
                            
                        st.plotly_chart(fig, use_container_width=True)
                        log_action(f"Gráfico '{tipo_grafico}' renderizado com eixos X:{eixo_x}, Y:{eixo_y}.")
            except Exception as e:
                st.error(f"Erro no motor de renderização. Verifique a compatibilidade dos eixos selecionados. Detalhe técnico: {e}")

    # ------------------------------------------------------------------------
    # ABA 7: I.A. AJUDANDO A INTERPRETAR DADOS
    # ------------------------------------------------------------------------
    with tabs[6]:
        st.markdown("### 🧠 Inteligência Artificial Analítica")
        st.markdown("Módulo de interpretação neural simulado para auxiliar pesquisadores na compreensão estatística.")
        
        if st.session_state.df_clean is None:
            st.warning("Forneça a base de dados tratada para a rede neural analisar.")
        else:
            df = st.session_state.df_clean
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if st.button("Iniciar Interpretação com IA (Engine Interno)"):
                with st.spinner("Analisando métricas de correlação, dispersão e tendências centrais..."):
                    time.sleep(2) # Simula o processamento de rede
                    
                    if not num_cols:
                        st.error("A I.A. não detectou colunas numéricas para gerar relatórios experimentais.")
                    else:
                        # Extrai informações reais dos dados para montar um texto estilo IA
                        estatisticas = df[num_cols].describe()
                        var_max_media = estatisticas.loc['mean'].idxmax()
                        val_max_media = estatisticas.loc['mean'].max()
                        var_maior_desvio = estatisticas.loc['std'].idxmax()
                        val_maior_desvio = estatisticas.loc['std'].max()
                        
                        relatorio_ia = f"""
                        **Relatório Neural do NexusData:**
                        
                        Olá, {st.session_state.user_info.get('Nome', 'Pesquisador')}. Concluí a varredura da sua matriz tratada com **{df.shape[0]} registros**.
                        
                        **📌 Principais Descobertas:**
                        1. A variável com a maior concentração de valor médio é a **`{var_max_media}`** (Média: {val_max_media:.2f}).
                        2. Identifiquei que a variável **`{var_maior_desvio}`** apresenta a maior volatilidade no seu experimento, com um Desvio Padrão de {val_maior_desvio:.2f}. Isso indica que os dados desta coluna sofrem grande flutuação em relação à média.
                        
                        **💡 Recomendações de Aprofundamento Analítico:**
                        - Sugiro que você navegue até a Aba de Gráficos e gere um **Boxplot** para a coluna `{var_maior_desvio}`. É altamente provável a presença de *Outliers* (valores atípicos) que podem distorcer o resultado da sua pesquisa.
                        - Se existirem datas (timestamps), gere a *Estatística Temporal Mensal* para verificar se a flutuação obedece a um ciclo sazonal.
                        
                        *Status: Análise concluída. Aguardando novos parâmetros.*
                        """
                        
                        st.info(relatorio_ia)
                        log_action("Módulo de I.A. acionado.")

    # ------------------------------------------------------------------------
    # ABA 8: EXPORTAÇÃO E DOWNLOAD
    # ------------------------------------------------------------------------
    with tabs[7]:
        st.markdown("### 💾 Exportação de Base de Dados (Output)")
        
        if st.session_state.df_clean is None:
            st.warning("Não há pacote de dados preparado para exportação.")
        else:
            st.markdown("Baixe a planilha contendo a configuração atualizada ou retorne ao original.")
            
            def to_excel(df):
                output = BytesIO()
                # Usa openpyxl que é nativo nas instalações de dados modernas em python
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Nexus_Processed')
                processed_data = output.getvalue()
                return processed_data

            col_down1, col_down2 = st.columns(2)
            
            with col_down1:
                st.markdown("#### Matriz Tratada")
                st.markdown("Baixe os dados limpos, sem NaN, duplicatas ou inconsistências que você tratou na Aba 3.")
                try:
                    excel_clean = to_excel(st.session_state.df_clean)
                    st.download_button(
                        label="📥 Baixar Dados Tratados (Excel)",
                        data=excel_clean,
                        file_name='Nexus_Dados_Tratados.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                except Exception as e:
                    st.error(f"Erro ao compilar arquivo Excel: {e}")

            with col_down2:
                st.markdown("#### Matriz Original")
                st.markdown("Baixe o arquivo exato que foi feito o upload, caso tenha perdido o documento base original.")
                if st.session_state.df_raw is not None:
                    try:
                        excel_raw = to_excel(st.session_state.df_raw)
                        st.download_button(
                            label="📥 Baixar Dados Originais (Excel)",
                            data=excel_raw,
                            file_name='Nexus_Dados_Originais.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                    except Exception as e:
                        st.error(f"Erro ao compilar arquivo Excel original: {e}")

if __name__ == "__main__":
    main()
