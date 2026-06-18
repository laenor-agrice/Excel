# ============================================================
# NOME DO APLICATIVO: AgroDataLab - Análise de Dados Meteorológicos
# DESENVOLVIDO PARA: Tratamento de dados INMET e análise estatística
# ============================================================

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
from scipy.stats import pearsonr, spearmanr, f_oneway, levene
import warnings
warnings.filterwarnings('ignore')

# Tentar importar google.generativeai (opcional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("Google Generative AI não instalado. Recursos de IA desabilitados.")

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
    if GEMINI_AVAILABLE:
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
                <p>Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas</p>
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
                df.head(10),
                use_container_width=True,
                height=400
            )
            
            # Informações das colunas
            with st.expander("Informações das Colunas"):
                col_info = pd.DataFrame({
                    'Coluna': df.columns,
                    'Tipo': df.dtypes.values,
                    'Valores Únicos': df.nunique().values,
                    'Valores Nulos': df.isnull().sum().values
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
            st.dataframe(missing_info, use_container_width=True)
            
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
            if GEMINI_AVAILABLE and gemini_api_key:
                st.markdown("---")
                st.markdown("### Sugestão da IA")
                
                if st.button("Analisar com Gemini", key="btn_ai_treatment"):
                    try:
                        missing_summary = missing_info.to_string()
                        prompt = f"""
                        Analise os dados meteorológicos com valores ausentes:
                        
                        {missing_summary}
                        
                        Sugira a melhor estratégia para tratar esses dados ausentes.
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
                daily_selected = st.checkbox("Ativar Diário", key="daily")
            
            with col2:
                weekly_selected = st.checkbox("Ativar Semanal", key="weekly")
            
            with col3:
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
        if len(numeric_cols) > 0:
            stats_df = df[numeric_cols].describe()
            st.dataframe(stats_df, use_container_width=True)
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
        
        if len(numeric_cols) > 0:
            # Seção de correlação
            st.markdown("### Análise de Correlação")
            
            col1, col2 = st.columns(2)
            
            with col1:
                corr_method = st.radio(
                    "Método de Correlação:",
                    ['Pearson', 'Spearman']
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
                corr_matrix, p_values = calculate_correlations(df, method=corr_method.lower())
                
                if corr_matrix is not None:
                    st.markdown("#### Matriz de Correlação")
                    st.dataframe(corr_matrix, use_container_width=True)
                    
                    st.markdown("#### Valores P")
                    st.dataframe(p_values, use_container_width=True)
            
            st.markdown("---")
            
            # Coeficiente de Variação
            st.markdown("### Coeficiente de Variação")
            
            cv_df = calculate_cv(df)
            st.dataframe(cv_df, use_container_width=True)
            
            st.info("""
            **Interpretação do CV:**
            - **Baixo (até 10%):** Alta precisão
            - **Médio (10-20%):** Precisão moderada
            - **Alto (20-30%):** Baixa precisão
            - **Muito Alto (acima de 30%):** Dados muito dispersos
            """)
        else:
            st.warning("Não foram encontradas variáveis numéricas para análise.")
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
            
            date_columns = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower() or 'hora' in col.lower()]
            
            if date_columns:
                x_col = st.selectbox("Eixo X (Tempo):", date_columns)
                
                df_temp = df.copy()
                if df_temp[x_col].dtype == 'object':
                    df_temp[x_col] = pd.to_datetime(df_temp[x_col], format='mixed')
                
                numeric_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
                y_cols = st.multiselect(
                    "Variáveis para o eixo Y:",
                    numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                )
                
                if y_cols:
                    chart = create_line_chart(df_temp, x_col, y_cols)
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("Nenhuma coluna de data identificada.")
        
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
                chart = create_histogram(df, selected_cols, bins=n_bins)
                st.altair_chart(chart, use_container_width=True)
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
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Download CSV")
            csv = df.to_csv(index=False)
            st.download_button(
                label="Baixar CSV",
                data=csv,
                file_name="dados_processados.csv",
                mime="text/csv"
            )
        
        with col2:
            st.markdown("### Download Excel")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Dados_Processados', index=False)
            excel_data = output.getvalue()
            
            st.download_button(
                label="Baixar Excel",
                data=excel_data,
                file_name="dados_processados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("Faça o upload dos dados na Aba 1 primeiro!")

# ============================================================
# ABA 7 - REFERÊNCIAS
# ============================================================

with tab7:
    st.markdown('<p class="tab-header">Referências e Métodos Científicos</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>Fundamentação Científica</h3>
        
        <h4>Métodos Implementados</h4>
        
        <h5>1. Tratamento de Dados Ausentes</h5>
        <ul>
            <li><strong>Média:</strong> Substituição pela média aritmética</li>
            <li><strong>Mediana:</strong> Substituição pela mediana</li>
            <li><strong>Interpolação Linear:</strong> Estimativa baseada em pontos adjacentes</li>
        </ul>
        <p><strong>Referência:</strong> Little, R.J.A. & Rubin, D.B. (2019). Statistical Analysis with Missing Data. 3rd Edition. Wiley.</p>
        
        <h5>2. Correlação</h5>
        <ul>
            <li><strong>Pearson:</strong> Mede relação linear entre variáveis</li>
            <li><strong>Spearman:</strong> Versão não-paramétrica baseada em postos</li>
        </ul>
        <p><strong>Referência:</strong> Pearson, K. (1895); Spearman, C. (1904)</p>
        
        <h5>3. Coeficiente de Variação</h5>
        <p><strong>Fórmula:</strong> CV = (s / x̄) × 100</p>
        <p><strong>Classificação (Pimentel-Gomes, 2000):</strong></p>
        <ul>
            <li>CV ≤ 10%: Baixo</li>
            <li>10% < CV ≤ 20%: Médio</li>
            <li>20% < CV ≤ 30%: Alto</li>
            <li>CV > 30%: Muito Alto</li>
        </ul>
    </div>
    
    <div class="card">
        <h4>Créditos</h4>
        <p><strong>AgroDataLab</strong> v1.0</p>
        <p>Desenvolvido com Streamlit, Pandas, NumPy, SciPy, Altair</p>
        <p>Licença MIT</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <p>AgroDataLab v1.0 | Análise Inteligente de Dados Meteorológicos | 2024</p>
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
