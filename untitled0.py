# ============================================================
# NOME DO APLICATIVO: AgroDataLab - Análise de Dados Meteorológicos
# DESENVOLVIDO PARA: Tratamento de dados INMET e análise estatística
# ============================================================
import subprocess
import sys
import pkg_resources

# Lista de pacotes necessários
required = {'scipy', 'numpy', 'pandas', 'streamlit', 'altair', 'scikit-learn', 'openpyxl', 'google-generativeai'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

# Instalar pacotes faltantes
if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import altair as alt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import io
import base64
import google.generativeai as genai
from scipy.stats import pearsonr, spearmanr, f_oneway, levene
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÃO DA PÁGINA E ESTILO
# ============================================================
st.set_page_config(
    page_title="AgroDataLab",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS CUSTOMIZADO - DESIGN MINIMALISTA INSPIRADO NO MAPBIOMAS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a5632 0%, #2d8a4e 50%, #4caf50 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2d8a4e;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1a5632 0%, #2d8a4e 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 25px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(26,86,50,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(26,86,50,0.3);
        background: linear-gradient(135deg, #2d8a4e 0%, #1a5632 100%);
    }
    
    .tab-header {
        color: #1a5632;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .success-message {
        background: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background: white;
        border-radius: 8px;
        color: #1a5632;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a5632 0%, #2d8a4e 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def process_inmet_data(df):
    """
    Processa dados no formato INMET
    """
    try:
        # Identificar colunas de data
        date_columns = [col for col in df.columns if 'data' in col.lower() or 'hora' in col.lower()]
        
        # Converter colunas numéricas
        for col in df.columns:
            if col not in date_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Identificar coluna de data principal
        if date_columns:
            date_col = date_columns[0]
            try:
                df[date_col] = pd.to_datetime(df[date_col], format='mixed')
            except:
                pass
        
        return df
    except Exception as e:
        st.error(f"Erro no processamento: {str(e)}")
        return df

def detect_missing_values(df):
    """
    Detecta valores ausentes e retorna estatísticas
    """
    missing_info = pd.DataFrame({
        'Coluna': df.columns,
        'Valores Ausentes': df.isnull().sum().values,
        'Porcentagem (%)': (df.isnull().sum() / len(df) * 100).values
    })
    return missing_info[missing_info['Valores Ausentes'] > 0]

def fill_missing_values(df, method='mean'):
    """
    Preenche valores ausentes
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if method == 'mean':
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    elif method == 'median':
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    elif method == 'interpolate':
        df[numeric_columns] = df[numeric_columns].interpolate(method='linear')
    
    return df

def calculate_statistics(df, group_col=None, freq='D'):
    """
    Calcula estatísticas com diferentes agrupamentos
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if group_col and freq:
        df_temp = df.copy()
        df_temp['date_group'] = pd.to_datetime(df[group_col]).dt.to_period(freq)
        
        stats_df = df_temp.groupby('date_group')[numeric_cols].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(3)
        
        return stats_df
    else:
        stats_df = df[numeric_cols].describe().round(3)
        return stats_df

# ============================================================
# FUNÇÃO PARA CORRELAÇÃO
# ============================================================

def calculate_correlations(df, method='pearson'):
    """
    Calcula correlações entre variáveis numéricas
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return None, None
    
    corr_matrix = pd.DataFrame(index=numeric_cols, columns=numeric_cols)
    p_values = pd.DataFrame(index=numeric_cols, columns=numeric_cols)
    
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0
                p_values.iloc[i, j] = 0.0
            else:
                col1, col2 = numeric_cols[i], numeric_cols[j]
                valid_data = df[[col1, col2]].dropna()
                
                if len(valid_data) > 2:
                    if method == 'pearson':
                        corr, p_val = pearsonr(valid_data[col1], valid_data[col2])
                    else:  # spearman
                        corr, p_val = spearmanr(valid_data[col1], valid_data[col2])
                    
                    corr_matrix.iloc[i, j] = round(corr, 3)
                    p_values.iloc[i, j] = round(p_val, 4)
                else:
                    corr_matrix.iloc[i, j] = np.nan
                    p_values.iloc[i, j] = np.nan
    
    return corr_matrix, p_values

# ============================================================
# FUNÇÃO PARA ANOVA
# ============================================================

def perform_anova(df, target_col, group_col):
    """
    Realiza ANOVA entre grupos
    """
    if df[target_col].dtype not in [np.float64, np.int64]:
        return None
    
    groups = df[group_col].dropna().unique()
    if len(groups) < 2:
        return None
    
    group_data = [df[df[group_col] == group][target_col].dropna() for group in groups]
    
    try:
        # Teste de Levene para homogeneidade de variâncias
        stat_lev, p_lev = levene(*group_data)
        
        # ANOVA
        stat_anova, p_anova = f_oneway(*group_data)
        
        results = {
            'Levene_Statistic': round(stat_lev, 3),
            'Levene_P_Value': round(p_lev, 4),
            'ANOVA_F_Statistic': round(stat_anova, 3),
            'ANOVA_P_Value': round(p_anova, 4),
            'N_Groups': len(groups),
            'Groups': groups
        }
        
        return results
    except:
        return None

# ============================================================
# FUNÇÃO PARA COEFICIENTE DE VARIAÇÃO
# ============================================================

def calculate_cv(df):
    """
    Calcula Coeficiente de Variação
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    cv_data = []
    for col in numeric_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        cv = (std_val / mean_val * 100) if mean_val != 0 else np.nan
        
        cv_data.append({
            'Variável': col,
            'Média': round(mean_val, 3),
            'Desvio Padrão': round(std_val, 3),
            'CV (%)': round(cv, 2),
            'Classificação': classify_cv(cv)
        })
    
    return pd.DataFrame(cv_data)

def classify_cv(cv):
    """
    Classifica o Coeficiente de Variação
    """
    if pd.isna(cv):
        return "Não classificado"
    elif cv <= 10:
        return "Baixo"
    elif cv <= 20:
        return "Médio"
    elif cv <= 30:
        return "Alto"
    else:
        return "Muito Alto"

# ============================================================
# FUNÇÃO PARA CRIAR GRÁFICOS COM ALTAIR
# ============================================================

def create_line_chart(df, x_col, y_cols, title="Gráfico de Linhas"):
    """
    Cria gráfico de linhas com Altair
    """
    # Melt para transformar em formato long
    df_melted = pd.melt(
        df, 
        id_vars=[x_col], 
        value_vars=y_cols,
        var_name='Variável', 
        value_name='Valor'
    )
    
    chart = alt.Chart(df_melted).mark_line(point=True).encode(
        x=alt.X(f'{x_col}:T', title='Data'),
        y=alt.Y('Valor:Q', title='Valor'),
        color=alt.Color('Variável:N', legend=alt.Legend(title="Variáveis")),
        tooltip=[x_col, 'Variável', 'Valor']
    ).properties(
        title=title,
        width=800,
        height=400
    ).interactive()
    
    return chart

def create_histogram(df, cols, bins=30):
    """
    Cria histograma com Altair
    """
    df_melted = pd.melt(
        df, 
        value_vars=cols,
        var_name='Variável', 
        value_name='Valor'
    )
    
    chart = alt.Chart(df_melted).mark_bar(opacity=0.7).encode(
        alt.X('Valor:Q', bin=alt.Bin(maxbins=bins), title='Valor'),
        alt.Y('count()', title='Frequência'),
        alt.Color('Variável:N', legend=alt.Legend(title="Variáveis")),
        tooltip=['Variável', 'count()']
    ).properties(
        title="Histograma das Variáveis",
        width=800,
        height=400
    ).facet(
        'Variável:N',
        columns=2
    ).resolve_scale(
        y='independent'
    )
    
    return chart

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-weight:700;">🌱 AgroDataLab</h1>
    <p style="margin:0; opacity:0.9; font-size:1.1rem;">
        Análise Inteligente de Dados Meteorológicos
    </p>
    <p style="margin:0; opacity:0.8; font-size:0.9rem;">
        Desenvolvido para tratamento de dados INMET e análise estatística avançada
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar com configurações
with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    
    # API Key Gemini
    st.markdown("---")
    st.markdown("### 🤖 Inteligência Artificial")
    gemini_api_key = st.text_input(
        "Chave API Gemini",
        type="password",
        placeholder="Insira sua chave API do Gemini",
        help="Opcional: para análises avançadas com IA"
    )
    
    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-pro')
            st.success("✅ Gemini conectado!")
        except:
            st.error("❌ Erro na conexão com Gemini")
    
    st.markdown("---")
    st.markdown("### 📋 Sobre")
    st.info("""
    **AgroDataLab** v1.0
    
    Desenvolvido para análise e tratamento de dados meteorológicos do INMET.
    
    Funcionalidades:
    - Upload e processamento de dados
    - Tratamento de valores ausentes
    - Análises estatísticas
    - Visualizações gráficas
    - Exportação de dados
    """)

# ============================================================
# CRIAÇÃO DAS ABAS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1 - Upload",
    "2 - Tratamento", 
    "3 - Estatísticas",
    "4 - Análises Avançadas",
    "5 - Gráficos",
    "6 - Download",
    "7 - Referências"
])

# ============================================================
# ABA 1 - UPLOAD
# ============================================================

with tab1:
    st.markdown('<p class="tab-header">Upload de Dados</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4>Faça o upload da sua planilha</h4>
            <p>Formatos aceitos: CSV, Excel (.xlsx, .xls)</p>
            <p style="color: #666; font-size: 0.9rem;">
            Compatível com arquivos exportados pelo INMET
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Selecione o arquivo",
            type=['csv', 'xlsx', 'xls'],
            help="Arraste ou selecione o arquivo para upload"
        )
    
    if uploaded_file is not None:
        try:
            # Carregar dados
            if uploaded_file.name.endswith('.csv'):
                # Tentar diferentes encodings e delimitadores
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except:
                    try:
                        df = pd.read_csv(uploaded_file, encoding='latin1')
                    except:
                        df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
            else:
                df = pd.read_excel(uploaded_file)
            
            # Processar dados INMET
            df = process_inmet_data(df)
            
            # Armazenar no session_state
            st.session_state['df_original'] = df.copy()
            st.session_state['df_processed'] = df.copy()
            
            # Mensagem de sucesso
            st.markdown(f"""
            <div class="success-message">
                <h4>Upload realizado com sucesso!</h4>
                <p>Arquivo: {uploaded_file.name}</p>
                <p>Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Preview dos dados
            st.markdown("### Pré-visualização dos Dados")
            
            # Métricas rápidas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color:#1a5632; margin:0;">{}</h3>
                    <p style="margin:0; color:#666;">Registros</p>
                </div>
                """.format(len(df)), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color:#1a5632; margin:0;">{}</h3>
                    <p style="margin:0; color:#666;">Variáveis</p>
                </div>
                """.format(len(df.columns)), unsafe_allow_html=True)
            
            with col3:
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color:#1a5632; margin:0;">{:.1f}%</h3>
                    <p style="margin:0; color:#666;">Dados Ausentes</p>
                </div>
                """.format(missing_pct), unsafe_allow_html=True)
            
            with col4:
                memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color:#1a5632; margin:0;">{:.1f} MB</h3>
                    <p style="margin:0; color:#666;">Tamanho</p>
                </div>
                """.format(memory_usage), unsafe_allow_html=True)
            
            # Mostrar primeiras linhas
            st.markdown("### Primeiras 10 linhas do dataset:")
            st.dataframe(
                df.head(10).style.background_gradient(cmap='Greens', axis=0),
                use_container_width=True,
                height=400
            )
            
            # Informações das colunas
            with st.expander("Informações das Colunas"):
                col_info = pd.DataFrame({
                    'Coluna': df.columns,
                    'Tipo': df.dtypes.values,
                    'Valores Únicos': df.nunique().values,
                    'Valores Nulos': df.isnull().sum().values,
                    'Memória (bytes)': df.memory_usage(deep=True).values
                })
                st.dataframe(col_info, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {str(e)}")
            st.info("Verifique se o arquivo está no formato correto e tente novamente.")
    else:
        st.markdown("""
        <div class="warning-message">
            <h4>Nenhum arquivo selecionado</h4>
            <p>Faça o upload de um arquivo CSV ou Excel para começar a análise.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# ABA 2 - TRATAMENTO
# ============================================================

with tab2:
    st.markdown('<p class="tab-header">Tratamento de Dados Ausentes</p>', unsafe_allow_html=True)
    
    if 'df_processed' in st.session_state:
        df = st.session_state['df_processed']
        
        # Detectar valores ausentes
        missing_info = detect_missing_values(df)
        
        if len(missing_info) > 0:
            st.markdown("""
            <div class="warning-message">
                <h4>Valores Ausentes Detectados</h4>
                <p>As seguintes colunas possuem dados faltantes:</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Mostrar tabela de valores ausentes
            st.dataframe(
                missing_info.style.applymap(
                    lambda x: 'background-color: #ffebee' if x > 0 else '',
                    subset=['Valores Ausentes']
                ),
                use_container_width=True
            )
            
            # Opções de tratamento
            st.markdown("### Opções de Tratamento")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="card">
                    <h4>Preencher com Média</h4>
                    <p style="color:#666;">Substitui valores ausentes pela média da coluna</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Aplicar Média", key="btn_mean"):
                    df_filled = fill_missing_values(df.copy(), method='mean')
                    st.session_state['df_processed'] = df_filled
                    st.success("Valores preenchidos com média!")
                    st.rerun()
            
            with col2:
                st.markdown("""
                <div class="card">
                    <h4>Preencher com Mediana</h4>
                    <p style="color:#666;">Substitui valores ausentes pela mediana da coluna</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Aplicar Mediana", key="btn_median"):
                    df_filled = fill_missing_values(df.copy(), method='median')
                    st.session_state['df_processed'] = df_filled
                    st.success("Valores preenchidos com mediana!")
                    st.rerun()
            
            with col3:
                st.markdown("""
                <div class="card">
                    <h4>Interpolar</h4>
                    <p style="color:#666;">Preenche por interpolação linear entre valores</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Aplicar Interpolação", key="btn_interpolate"):
                    df_filled = fill_missing_values(df.copy(), method='interpolate')
                    st.session_state['df_processed'] = df_filled
                    st.success("Valores interpolados!")
                    st.rerun()
            
            # Opção de excluir
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="card">
                    <h4>Excluir Linhas com Dados Ausentes</h4>
                    <p style="color:#666;">Remove todas as linhas que contêm valores ausentes</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Excluir Linhas", key="btn_drop"):
                    df_dropped = df.dropna()
                    st.session_state['df_processed'] = df_dropped
                    st.success(f"{len(df) - len(df_dropped)} linhas removidas!")
                    st.rerun()
            
            with col2:
                st.markdown("""
                <div class="card">
                    <h4>Manter Dados Originais</h4>
                    <p style="color:#666;">Restaura os dados para o estado original</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Restaurar Original", key="btn_restore"):
                    if 'df_original' in st.session_state:
                        st.session_state['df_processed'] = st.session_state['df_original'].copy()
                        st.success("Dados restaurados!")
                        st.rerun()
            
            # Assistente IA para sugestões
            if gemini_api_key:
                st.markdown("---")
                st.markdown("### Sugestão da IA")
                
                if st.button("Analisar com Gemini", key="btn_ai_treatment"):
                    try:
                        missing_summary = missing_info.to_string()
                        prompt = f"""
                        Analise os dados meteorológicos com valores ausentes:
                        
                        {missing_summary}
                        
                        Sugira a melhor estratégia para tratar esses dados ausentes considerando:
                        1. Impacto na análise estatística
                        2. Preservação da série temporal
                        3. Recomendações científicas para dados meteorológicos
                        
                        Responda em português de forma objetiva.
                        """
                        
                        response = model.generate_content(prompt)
                        
                        st.markdown("""
                        <div class="success-message">
                            <h4>Recomendação da IA:</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.write(response.text)
                    except:
                        st.error("Erro ao consultar a IA. Verifique sua chave API.")
        else:
            st.markdown("""
            <div class="success-message">
                <h4>Nenhum valor ausente detectado!</h4>
                <p>Seus dados estão completos e prontos para análise.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Faça o upload dos dados na Aba 1 primeiro!")

# ============================================================
# ABA 3 - ESTATÍSTICAS BÁSICAS
# ============================================================

with tab3:
    st.markdown('<p class="tab-header">Estatísticas Descritivas</p>', unsafe_allow_html=True)
    
    if 'df_processed' in st.session_state:
        df = st.session_state['df_processed']
        
        # Selecionar coluna de data
        date_columns = [col for col in df.columns if 'data' in col.lower() or 'hora' in col.lower() or 'date' in col.lower()]
        
        if date_columns:
            date_col = st.selectbox("Selecione a coluna de data:", date_columns)
            
            # Converter para datetime se necessário
            if df[date_col].dtype == 'object':
                df[date_col] = pd.to_datetime(df[date_col], format='mixed')
            
            # Frequência de agregação
            st.markdown("### Frequência de Agregação")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="card">
                    <h4>Diário</h4>
                    <p style="color:#666;">Agrupa os dados por dia</p>
                </div>
                """, unsafe_allow_html=True)
                
                daily_selected = st.checkbox("Ativar Diário", key="daily")
            
            with col2:
                st.markdown("""
                <div class="card">
                    <h4>Semanal</h4>
                    <p style="color:#666;">Agrupa os dados por semana</p>
                </div>
                """, unsafe_allow_html=True)
                
                weekly_selected = st.checkbox("Ativar Semanal", key="weekly")
            
            with col3:
                st.markdown("""
                <div class="card">
                    <h4>Mensal</h4>
                    <p style="color:#666;">Agrupa os dados por mês</p>
                </div>
                """, unsafe_allow_html=True)
                
                monthly_selected = st.checkbox("Ativar Mensal", key="monthly")
            
            # Mostrar estatísticas conforme seleção
            if daily_selected:
                st.markdown("### Estatísticas Diárias")
                daily_stats = calculate_statistics(df, date_col, 'D')
                st.dataframe(daily_stats, use_container_width=True, height=400)
            
            if weekly_selected:
                st.markdown("### Estatísticas Semanais")
                weekly_stats = calculate_statistics(df, date_col, 'W')
                st.dataframe(weekly_stats, use_container_width=True, height=400)
            
            if monthly_selected:
                st.markdown("### Estatísticas Mensais")
                monthly_stats = calculate_statistics(df, date_col, 'M')
                st.dataframe(monthly_stats, use_container_width=True, height=400)
        else:
            st.info("Nenhuma coluna de data identificada. Mostrando estatísticas gerais:")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Estatísticas gerais
        st.markdown("---")
        st.markdown("### Resumo Estatístico Geral")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_df = df[numeric_cols].describe()
        
        # Adicionar métricas adicionais
        stats_df.loc['variância'] = df[numeric_cols].var()
        stats_df.loc['assimetria'] = df[numeric_cols].skew()
        stats_df.loc['curtose'] = df[numeric_cols].kurtosis()
        
        st.dataframe(stats_df.style.background_gradient(cmap='Greens', axis=1), use_container_width=True)
    else:
        st.warning("Faça o upload dos dados na Aba 1 primeiro!")

# ============================================================
# ABA 4 - ANÁLISES AVANÇADAS
# ============================================================

with tab4:
    st.markdown('<p class="tab-header">Análises Estatísticas Avançadas</p>', unsafe_allow_html=True)
    
    if 'df_processed' in st.session_state:
        df = st.session_state['df_processed']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Seção de correlação
        st.markdown("### Análise de Correlação")
        
        col1, col2 = st.columns(2)
        
        with col1:
            corr_method = st.radio(
                "Método de Correlação:",
                ['Pearson', 'Spearman'],
                help="Pearson: correlação linear | Spearman: correlação de postos"
            )
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>Interpretação</h4>
                <p style="font-size:0.9rem;">
                <strong>0.0 - 0.3:</strong> Fraca<br>
                <strong>0.3 - 0.7:</strong> Moderada<br>
                <strong>0.7 - 1.0:</strong> Forte
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        if len(numeric_cols) >= 2:
            corr_matrix, p_values = calculate_correlations(
                df, 
                method=corr_method.lower()
            )
            
            if corr_matrix is not None:
                st.markdown("#### Matriz de Correlação")
                st.dataframe(
                    corr_matrix.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1),
                    use_container_width=True
                )
                
                st.markdown("#### Valores P")
                st.dataframe(
                    p_values.style.applymap(
                        lambda x: 'background-color: #c8e6c9' if x < 0.05 else 'background-color: #ffcdd2'
                    ),
                    use_container_width=True
                )
                
                # Assistente IA para correlação
                if gemini_api_key:
                    if st.button("Interpretar Correlações com IA", key="btn_ai_corr"):
                        try:
                            prompt = f"""
                            Analise a matriz de correlação entre variáveis meteorológicas:
                            
                            Método: {corr_method}
                            
                            Matriz de correlação:
                            {corr_matrix.to_string()}
                            
                            Valores P:
                            {p_values.to_string()}
                            
                            Interprete os resultados, destacando:
                            1. Correlações significativas (p < 0.05)
                            2. Força das relações
                            3. Implicações práticas para agricultura/meteorologia
                            
                            Responda em português.
                            """
                            
                            response = model.generate_content(prompt)
                            st.markdown("#### Interpretação da IA:")
                            st.write(response.text)
                        except:
                            st.error("Erro ao consultar a IA.")
        else:
            st.warning("São necessárias pelo menos 2 variáveis numéricas para correlação.")
        
        st.markdown("---")
        
        # Seção ANOVA
        st.markdown("### Análise de Variância (ANOVA)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            anova_target = st.selectbox(
                "Variável Dependente:",
                numeric_cols,
                help="Variável numérica para análise"
            )
        
        with col2:
            # Identificar colunas categóricas ou de data
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            date_cols = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower()]
            group_cols = cat_cols + date_cols
            
            if group_cols:
                anova_group = st.selectbox(
                    "Variável de Agrupamento:",
                    group_cols,
                    help="Variável para formar os grupos"
                )
                
                if st.button("Executar ANOVA", key="btn_anova"):
                    results = perform_anova(df, anova_target, anova_group)
                    
                    if results:
                        st.markdown("### Resultados da ANOVA")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Teste de Levene</h4>
                                <p style="font-size:1.5rem; font-weight:bold;">
                                    F = {results['Levene_Statistic']}
                                </p>
                                <p>p = {results['Levene_P_Value']}</p>
                                <p style="font-size:0.8rem;">
                                    {'Homogeneidade confirmada' if results['Levene_P_Value'] > 0.05 else 'Variâncias diferentes'}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>ANOVA</h4>
                                <p style="font-size:1.5rem; font-weight:bold;">
                                    F = {results['ANOVA_F_Statistic']}
                                </p>
                                <p>p = {results['ANOVA_P_Value']}</p>
                                <p style="font-size:0.8rem;">
                                    {'Diferença significativa' if results['ANOVA_P_Value'] < 0.05 else 'Sem diferença significativa'}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Grupos Analisados</h4>
                                <p style="font-size:1.5rem; font-weight:bold;">
                                    {results['N_Groups']}
                                </p>
                                <p>grupos distintos</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("Não foi possível realizar a ANOVA. Verifique os dados.")
            else:
                st.warning("Nenhuma variável de agrupamento identificada.")
        
        st.markdown("---")
        
        # Coeficiente de Variação
        st.markdown("### Coeficiente de Variação")
        
        cv_df = calculate_cv(df)
        st.dataframe(
            cv_df.style.applymap(
                lambda x: 'background-color: #c8e6c9' if x == 'Baixo' 
                else 'background-color: #fff9c4' if x == 'Médio'
                else 'background-color: #ffccbc' if x == 'Alto'
                else 'background-color: #ffab91' if x == 'Muito Alto'
                else '',
                subset=['Classificação']
            ),
            use_container_width=True
        )
        
        st.info("""
        **Interpretação do CV:**
        - **Baixo (até 10%):** Alta precisão dos dados
        - **Médio (10-20%):** Precisão moderada
        - **Alto (20-30%):** Baixa precisão
        - **Muito Alto (acima de 30%):** Dados muito dispersos
        """)
    else:
        st.warning("Faça o upload dos dados na Aba 1 primeiro!")

# ============================================================
# ABA 5 - GRÁFICOS
# ============================================================

with tab5:
    st.markdown('<p class="tab-header">Visualização de Dados</p>', unsafe_allow_html=True)
    
    if 'df_processed' in st.session_state:
        df = st.session_state['df_processed']
        
        # Tipo de gráfico
        chart_type = st.radio(
            "Selecione o tipo de gráfico:",
            ['Gráfico de Linhas', 'Histograma'],
            horizontal=True
        )
        
        if chart_type == 'Gráfico de Linhas':
            st.markdown("### Gráfico de Linhas Temporais")
            
            # Selecionar eixo X
            date_columns = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower() or 'hora' in col.lower()]
            
            if date_columns:
                x_col = st.selectbox("Eixo X (Tempo):", date_columns)
                
                # Converter para datetime
                df_temp = df.copy()
                if df_temp[x_col].dtype == 'object':
                    df_temp[x_col] = pd.to_datetime(df_temp[x_col], format='mixed')
                
                # Selecionar variáveis para Y
                numeric_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
                y_cols = st.multiselect(
                    "Variáveis para o eixo Y:",
                    numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                )
                
                if y_cols:
                    # Criar gráfico
                    chart = create_line_chart(
                        df_temp,
                        x_col,
                        y_cols,
                        title="Série Temporal das Variáveis Meteorológicas"
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Estatísticas do gráfico
                    with st.expander("Estatísticas do Período"):
                        for col in y_cols:
                            st.markdown(f"**{col}:**")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Média", f"{df_temp[col].mean():.2f}")
                            with col2:
                                st.metric("Máximo", f"{df_temp[col].max():.2f}")
                            with col3:
                                st.metric("Mínimo", f"{df_temp[col].min():.2f}")
                            with col4:
                                st.metric("Desvio Padrão", f"{df_temp[col].std():.2f}")
                else:
                    st.info("Selecione pelo menos uma variável para o eixo Y.")
            else:
                st.warning("Nenhuma coluna de data identificada para o eixo X.")
        
        elif chart_type == 'Histograma':
            st.markdown("### Histograma das Variáveis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_cols = st.multiselect(
                "Selecione as variáveis:",
                numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
            )
            
            if selected_cols:
                n_bins = st.slider("Número de bins:", 10, 100, 30)
                
                # Criar histograma
                chart = create_histogram(df, selected_cols, bins=n_bins)
                st.altair_chart(chart, use_container_width=True)
                
                # Estatísticas descritivas
                with st.expander("Estatísticas Descritivas"):
                    st.dataframe(df[selected_cols].describe(), use_container_width=True)
            else:
                st.info("Selecione pelo menos uma variável para visualizar.")
        
        # Análise com IA
        if gemini_api_key and ('y_cols' in locals() or 'selected_cols' in locals()):
            if st.button("Analisar Gráfico com IA", key="btn_ai_chart"):
                try:
                    if chart_type == 'Gráfico de Linhas' and y_cols:
                        data_desc = df[y_cols].describe().to_string()
                        prompt = f"""
                        Analise a série temporal meteorológica:
                        
                        Variáveis: {', '.join(y_cols)}
                        
                        Estatísticas descritivas:
                        {data_desc}
                        
                        Forneça insights sobre:
                        1. Tendências observadas
                        2. Sazonalidade
                        3. Eventos extremos
                        4. Recomendações baseadas nos padrões
                        
                        Responda em português.
                        """
                    elif selected_cols:
                        data_desc = df[selected_cols].describe().to_string()
                        prompt = f"""
                        Analise a distribuição das variáveis meteorológicas:
                        
                        Variáveis: {', '.join(selected_cols)}
                        
                        Estatísticas descritivas:
                        {data_desc}
                        
                        Interprete:
                        1. Forma da distribuição
                        2. Assimetria e curtose
                        3. Valores atípicos
                        4. Implicações práticas
                        
                        Responda em português.
                        """
                    
                    response = model.generate_content(prompt)
                    
                    st.markdown("""
                    <div class="success-message">
                        <h4>Análise da IA:</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write(response.text)
                except:
                    st.error("Erro ao consultar a IA.")
    else:
        st.warning("Faça o upload dos dados na Aba 1 primeiro!")

# ============================================================
# ABA 6 - DOWNLOAD
# ============================================================

with tab6:
    st.markdown('<p class="tab-header">Download dos Dados Processados</p>', unsafe_allow_html=True)
    
    if 'df_processed' in st.session_state:
        df = st.session_state['df_processed']
        
        st.markdown("""
        <div class="success-message">
            <h4>Dados prontos para download!</h4>
            <p>Seus dados foram processados e estão prontos para exportação.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Informações do arquivo
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Registros</h4>
                <p style="font-size:2rem; font-weight:bold; color:#1a5632;">{len(df)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Colunas</h4>
                <p style="font-size:2rem; font-weight:bold; color:#1a5632;">{len(df.columns)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            missing = df.isnull().sum().sum()
            st.markdown(f"""
            <div class="metric-card">
                <h4>Dados Ausentes</h4>
                <p style="font-size:2rem; font-weight:bold; color:#1a5632;">{missing}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Opções de download
        st.markdown("### Selecione o formato de exportação:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4>CSV</h4>
                <p style="color:#666;">Formato texto com separador vírgula</p>
                <p style="color:#666;">Compatível com Excel, Google Sheets</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Criar CSV
            csv = df.to_csv(index=False)
            b64_csv = base64.b64encode(csv.encode()).decode()
            
            st.markdown(
                f'<a href="data:file/csv;base64,{b64_csv}" download="dados_processados.csv">'
                f'<button style="background: linear-gradient(135deg, #1a5632 0%, #2d8a4e 100%); '
                f'color: white; border: none; padding: 0.6rem 2rem; border-radius: 25px; '
                f'cursor: pointer; width: 100%;">Baixar CSV</button></a>',
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown("""
            <div class="card">
                <h4>Excel</h4>
                <p style="color:#666;">Formato nativo do Microsoft Excel</p>
                <p style="color:#666;">Com múltiplas abas e formatação</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Criar Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Dados_Processados', index=False)
                
                # Adicionar aba de estatísticas
                stats_df = df.describe()
                stats_df.to_excel(writer, sheet_name='Estatísticas')
            
            excel_data = output.getvalue()
            b64_excel = base64.b64encode(excel_data).decode()
            
            st.markdown(
                f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" '
                f'download="dados_processados.xlsx">'
                f'<button style="background: linear-gradient(135deg, #1a5632 0%, #2d8a4e 100%); '
                f'color: white; border: none; padding: 0.6rem 2rem; border-radius: 25px; '
                f'cursor: pointer; width: 100%;">Baixar Excel</button></a>',
                unsafe_allow_html=True
            )
        
        # Resumo do processamento
        st.markdown("---")
        st.markdown("### Resumo do Processamento")
        
        if 'df_original' in st.session_state:
            df_original = st.session_state['df_original']
            
            summary_data = {
                'Métrica': [
                    'Registros Originais',
                    'Registros Finais',
                    'Colunas',
                    'Dados Ausentes (Original)',
                    'Dados Ausentes (Final)',
                    'Tamanho em Memória'
                ],
                'Valor': [
                    len(df_original),
                    len(df),
                    len(df.columns),
                    df_original.isnull().sum().sum(),
                    df.isnull().sum().sum(),
                    f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.success("Seus dados estão prontos para uso!")
        
    else:
        st.warning("Faça o upload e processamento dos dados nas abas anteriores primeiro!")

# ============================================================
# ABA 7 - REFERÊNCIAS E MÉTODOS
# ============================================================

with tab7:
    st.markdown('<p class="tab-header">Referências e Métodos Científicos</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>Fundamentação Científica</h3>
        
        <p>Este aplicativo utiliza métodos estatísticos consolidados na literatura científica para análise de dados meteorológicos e agronômicos.</p>
        
        <h4>Métodos Implementados</h4>
        
        <h5>1. Tratamento de Dados Ausentes</h5>
        <ul>
            <li><strong>Média:</strong> Substituição pela média aritmética - Método simples e eficaz para dados com distribuição normal</li>
            <li><strong>Mediana:</strong> Substituição pela mediana - Robusto para dados com outliers</li>
            <li><strong>Interpolação Linear:</strong> Estimativa de valores baseada em pontos adjacentes - Ideal para séries temporais</li>
        </ul>
        
        <p><strong>Referência:</strong> Little, R.J.A. & Rubin, D.B. (2019). Statistical Analysis with Missing Data. 3rd Edition. Wiley.</p>
        
        <h5>2. Correlação de Pearson</h5>
        <p>Mede a relação linear entre duas variáveis. O coeficiente varia de -1 a 1:</p>
        <ul>
            <li>r = 1: Correlação positiva perfeita</li>
            <li>r = 0: Sem correlação linear</li>
            <li>r = -1: Correlação negativa perfeita</li>
        </ul>
        
        <p><strong>Fórmula:</strong> r = Σ((x - x̄)(y - ȳ)) / √(Σ(x - x̄)² · Σ(y - ȳ)²)</p>
        <p><strong>Referência:</strong> Pearson, K. (1895). Notes on regression and inheritance in the case of two parents. Proceedings of the Royal Society of London, 58, 240-242.</p>
        
        <h5>3. Correlação de Spearman</h5>
        <p>Versão não-paramétrica da correlação, baseada em postos. Não assume distribuição normal.</p>
        
        <p><strong>Referência:</strong> Spearman, C. (1904). The proof and measurement of association between two things. American Journal of Psychology, 15, 72-101.</p>
        
        <h5>4. ANOVA (Análise de Variância)</h5>
        <p>Testa diferenças entre médias de três ou mais grupos.</p>
        <p><strong>Pressupostos:</strong></p>
        <ul>
            <li>Independência das observações</li>
            <li>Normalidade dos resíduos</li>
            <li>Homogeneidade de variâncias (testada com Levene)</li>
        </ul>
        
        <p><strong>Fórmula:</strong> F = Variância entre grupos / Variância dentro dos grupos</p>
        <p><strong>Referência:</strong> Fisher, R.A. (1925). Statistical Methods for Research Workers. Oliver and Boyd, Edinburgh.</p>
        
        <h5>5. Coeficiente de Variação (CV)</h5>
        <p>Medida de dispersão relativa, expressa em porcentagem.</p>
        <p><strong>Fórmula:</strong> CV = (s / x̄) × 100</p>
        <p>Onde s = desvio padrão e x̄ = média</p>
        
        <p><strong>Classificação (Pimentel-Gomes, 2000):</strong></p>
        <ul>
            <li>CV até 10%: Baixo (alta precisão)</li>
            <li>10% a 20%: Médio</li>
            <li>20% a 30%: Alto</li>
            <li>CV acima de 30%: Muito Alto (baixa precisão)</li>
        </ul>
        
        <p><strong>Referência:</strong> Pimentel-Gomes, F. (2000). Curso de Estatística Experimental. 14ª ed. Piracicaba: ESALQ/USP.</p>
    </div>
    
    <div class="card">
        <h4>Estatísticas Descritivas</h4>
        <p>As estatísticas descritivas incluem:</p>
        <ul>
            <li><strong>Média (x̄):</strong> Soma dos valores dividida pelo número de observações</li>
            <li><strong>Mediana:</strong> Valor central quando os dados são ordenados</li>
            <li><strong>Desvio Padrão (s):</strong> Raiz quadrada da variância</li>
            <li><strong>Variância (s²):</strong> Média dos quadrados das diferenças em relação à média</li>
            <li><strong>Assimetria:</strong> Medida de simetria da distribuição</li>
            <li><strong>Curtose:</strong> Medida de achatamento da distribuição</li>
        </ul>
    </div>
    
    <div class="card">
        <h4>Créditos</h4>
        <p><strong>Desenvolvido por:</strong> Equipe AgroDataLab</p>
        <p><strong>Versão:</strong> 1.0</p>
        <p><strong>Bibliotecas Utilizadas:</strong></p>
        <ul>
            <li>Streamlit - Interface web interativa</li>
            <li>Pandas - Manipulação de dados</li>
            <li>NumPy - Computação numérica</li>
            <li>SciPy - Estatística avançada</li>
            <li>Altair - Visualização de dados</li>
            <li>Google Generative AI - Análise com IA</li>
        </ul>
        
        <p><strong>Licença:</strong> MIT License</p>
        <p>Este software é fornecido "como está", sem garantias de qualquer tipo.</p>
    </div>
    
    <div class="card">
        <h4>Contato e Suporte</h4>
        <p>Para dúvidas, sugestões ou reportar bugs:</p>
        <ul>
            <li>Email: suporte@agrodatalab.com</li>
            <li>GitHub: github.com/agrodatalab</li>
            <li>Documentação: docs.agrodatalab.com</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <p>AgroDataLab v1.0 | Desenvolvido com Streamlit e Python</p>
    <p>Análise Inteligente de Dados Meteorológicos | 2024</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INICIALIZAÇÃO DO SESSION STATE
# ============================================================

if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None
if 'df_processed' not in st.session_state:
    st.session_state['df_processed'] = None

# ============================================================
# FIM DO CÓDIGO
# ============================================================
