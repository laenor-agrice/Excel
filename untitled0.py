# ============================================================
# AgroDataLab - Sistema Completo de Análise Meteorológica
# Compatível com Streamlit Cloud - 100% Estável
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÃO AVANÇADA DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="AgroDataLab | Análise Meteorológica",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/agrodatalab',
        'Report a bug': "https://github.com/agrodatalab/issues",
        'About': "AgroDataLab v2.0 - Análise de Dados INMET"
    }
)

# ============================================================
# CSS PROFISSIONAL - DESIGN MODERNO E MINIMALISTA
# ============================================================
st.markdown("""
<style>
    /* Importação de fontes */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Estilos globais */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Container principal */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header principal */
    .hero-header {
        background: linear-gradient(135deg, #0d2818 0%, #1a5632 30%, #2d8a4e 70%, #4caf50 100%);
        padding: 2.5rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 20px 60px rgba(26,86,50,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    
    .hero-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .hero-header p {
        font-size: 1.2rem;
        opacity: 0.95;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    /* Cards modernos */
    .modern-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
    
    /* Cards de métrica */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border-left: 4px solid #2d8a4e;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        transition: all 0.2s;
    }
    
    .metric-card:hover {
        border-left-width: 6px;
        box-shadow: 0 4px 16px rgba(45,138,78,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a5632;
        margin: 0;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Botões estilizados */
    .stButton > button {
        background: linear-gradient(135deg, #1a5632 0%, #2d8a4e 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(26,86,50,0.2);
        letter-spacing: 0.3px;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(26,86,50,0.35);
        background: linear-gradient(135deg, #2d8a4e 0%, #1a5632 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Mensagens de status */
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 1px solid #81c784;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #2e7d32;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: 1px solid #64b5f6;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #1565c0;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border: 1px solid #ffb74d;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #e65100;
        margin: 1rem 0;
    }
    
    /* Tabs customizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: #f8f9fa;
        padding: 0.75rem;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 0.75rem 1.5rem;
        background: white;
        border-radius: 10px;
        color: #495057;
        font-weight: 500;
        font-size: 0.95rem;
        border: 1px solid #dee2e6;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a5632, #2d8a4e);
        color: white;
        border: none;
        box-shadow: 0 4px 12px rgba(26,86,50,0.3);
    }
    
    /* DataFrames estilizados */
    .dataframe-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #e9ecef;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 12px;
        margin-top: 3rem;
        color: #6c757d;
        border: 1px solid #e9ecef;
    }
    
    /* Responsividade */
    @media (max-width: 768px) {
        .hero-header h1 {
            font-size: 2rem;
        }
        .metric-value {
            font-size: 1.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNÇÕES CIENTÍFICAS (IMPLEMENTADAS MANUALMENTE)
# ============================================================

def calcular_estatisticas_descritivas(df, coluna):
    """
    Calcula estatísticas descritivas completas para uma coluna
    Referência: Montgomery & Runger (2018) - Applied Statistics
    """
    dados = df[coluna].dropna()
    
    n = len(dados)
    media = np.mean(dados)
    mediana = np.median(dados)
    desvio_padrao = np.std(dados, ddof=1)  # ddof=1 para amostral
    variancia = np.var(dados, ddof=1)
    minimo = np.min(dados)
    maximo = np.max(dados)
    amplitude = maximo - minimo
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1
    
    # Coeficiente de variação
    cv = (desvio_padrao / media * 100) if media != 0 else 0
    
    # Assimetria (Skewness)
    assimetria = np.sum((dados - media) ** 3) / (n * desvio_padrao ** 3) if desvio_padrao > 0 else 0
    
    # Curtose
    curtose = np.sum((dados - media) ** 4) / (n * desvio_padrao ** 4) - 3 if desvio_padrao > 0 else 0
    
    return {
        'n': n,
        'Média': round(media, 4),
        'Mediana': round(mediana, 4),
        'Desvio Padrão': round(desvio_padrao, 4),
        'Variância': round(variancia, 4),
        'Mínimo': round(minimo, 4),
        'Máximo': round(maximo, 4),
        'Amplitude': round(amplitude, 4),
        'Q1 (25%)': round(q1, 4),
        'Q3 (75%)': round(q3, 4),
        'IQR': round(iqr, 4),
        'CV (%)': round(cv, 2),
        'Assimetria': round(assimetria, 4),
        'Curtose': round(curtose, 4)
    }

def classificar_cv(cv):
    """Classificação do CV segundo Pimentel-Gomes (2000)"""
    if cv <= 10:
        return "Baixo 🟢"
    elif cv <= 20:
        return "Médio 🟡"
    elif cv <= 30:
        return "Alto 🟠"
    else:
        return "Muito Alto 🔴"

def calcular_correlacao_pearson(df, col1, col2):
    """
    Calcula correlação de Pearson manualmente
    Fórmula: r = Σ((x-x̄)(y-ȳ)) / √(Σ(x-x̄)² * Σ(y-ȳ)²)
    Referência: Pearson (1895)
    """
    dados = df[[col1, col2]].dropna()
    x = dados[col1].values
    y = dados[col2].values
    
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    
    media_x = np.mean(x)
    media_y = np.mean(y)
    
    numerador = np.sum((x - media_x) * (y - media_y))
    denominador = np.sqrt(np.sum((x - media_x) ** 2) * np.sum((y - media_y) ** 2))
    
    if denominador == 0:
        return 0, 1.0
    
    r = numerador / denominador
    
    # Teste t para significância
    t_stat = r * np.sqrt((n - 2) / (1 - r ** 2))
    
    return round(r, 4), round(t_stat, 4)

def calcular_media_movel(dados, janela=7):
    """Calcula média móvel simples"""
    return pd.Series(dados).rolling(window=janela, min_periods=1).mean()

def detectar_outliers_iqr(dados):
    """Detecta outliers usando método IQR (Tukey, 1977)"""
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    outliers = (dados < limite_inferior) | (dados > limite_superior)
    return outliers, limite_inferior, limite_superior

# ============================================================
# FUNÇÕES DE VISUALIZAÇÃO COM ALTAIR
# ============================================================

def criar_grafico_linhas(df, x_col, y_cols, titulo="Série Temporal"):
    """Cria gráfico de linhas profissional com Altair"""
    df_melted = df.melt(
        id_vars=[x_col], 
        value_vars=y_cols,
        var_name='Variável', 
        value_name='Valor'
    )
    
    # Gráfico principal
    lines = alt.Chart(df_melted).mark_line(
        strokeWidth=2.5,
        point=alt.OverlayMarkDef(size=40, filled=True)
    ).encode(
        x=alt.X(f'{x_col}:T', title='Data', axis=alt.Axis(grid=False)),
        y=alt.Y('Valor:Q', title='Valor', axis=alt.Axis(grid=True, gridDash=[2,2])),
        color=alt.Color('Variável:N', legend=alt.Legend(
            orient='bottom',
            title=None,
            labelFontSize=12
        )),
        tooltip=[
            alt.Tooltip(f'{x_col}:T', title='Data'),
            alt.Tooltip('Variável:N', title='Variável'),
            alt.Tooltip('Valor:Q', title='Valor', format='.2f')
        ]
    ).properties(
        title=alt.TitleParams(
            text=titulo,
            fontSize=18,
            fontWeight='bold',
            color='#1a5632'
        ),
        height=450
    ).interactive(bind_y=False)
    
    return lines

def criar_grafico_barras(df, x_col, y_col, titulo="Gráfico de Barras"):
    """Cria gráfico de barras estilizado"""
    bars = alt.Chart(df).mark_bar(
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4,
        opacity=0.85
    ).encode(
        x=alt.X(f'{x_col}:O', title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f'{y_col}:Q', title=y_col),
        color=alt.Color(f'{y_col}:Q', scale=alt.Scale(scheme='greens')),
        tooltip=[x_col, alt.Tooltip(y_col, format='.2f')]
    ).properties(
        title=alt.TitleParams(
            text=titulo,
            fontSize=16,
            fontWeight='bold',
            color='#1a5632'
        ),
        height=400
    )
    
    return bars

def criar_histograma(df, coluna, bins=30):
    """Cria histograma com curva de densidade"""
    hist = alt.Chart(df).mark_bar(
        opacity=0.7,
        cornerRadiusTopLeft=2,
        cornerRadiusTopRight=2
    ).encode(
        alt.X(f'{coluna}:Q', bin=alt.Bin(maxbins=bins), title=coluna),
        alt.Y('count()', title='Frequência'),
        tooltip=['count()']
    ).properties(
        title=alt.TitleParams(
            text=f'Distribuição de {coluna}',
            fontSize=16,
            fontWeight='bold',
            color='#1a5632'
        ),
        height=400
    )
    
    return hist

def criar_boxplot(df, coluna, grupo=None):
    """Cria boxplot para análise de distribuição"""
    if grupo:
        box = alt.Chart(df).mark_boxplot(
            extent='min-max',
            color='#2d8a4e'
        ).encode(
            x=alt.X(f'{grupo}:O', title=grupo),
            y=alt.Y(f'{coluna}:Q', title=coluna),
            color=alt.Color(f'{grupo}:N', legend=None)
        ).properties(
            height=400
        )
    else:
        box = alt.Chart(df).mark_boxplot(
            extent='min-max',
            color='#2d8a4e'
        ).encode(
            y=alt.Y(f'{coluna}:Q', title=coluna)
        ).properties(
            height=400
        )
    
    return box

# ============================================================
# FUNÇÕES DE PROCESSAMENTO DE DADOS
# ============================================================

def carregar_dados(arquivo):
    """Carrega dados de diferentes formatos com detecção automática"""
    try:
        if arquivo.name.endswith('.csv'):
            # Tentar diferentes encodings
            for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(arquivo, encoding=encoding)
                    break
                except:
                    continue
            
            # Tentar diferentes separadores
            if df.shape[1] == 1:
                for sep in [';', '\t', '|']:
                    try:
                        df = pd.read_csv(arquivo, encoding=encoding, sep=sep)
                        if df.shape[1] > 1:
                            break
                    except:
                        continue
        else:
            df = pd.read_excel(arquivo)
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        return None

def processar_dados_inmet(df):
    """Processa dados no formato INMET automaticamente"""
    df = df.copy()
    
    # Identificar colunas de data
    date_patterns = ['data', 'hora', 'date', 'time', 'ano', 'mes', 'dia']
    date_cols = []
    
    for col in df.columns:
        if any(pattern in col.lower() for pattern in date_patterns):
            date_cols.append(col)
            try:
                df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce')
            except:
                pass
    
    # Converter colunas numéricas (substituir vírgulas por pontos)
    for col in df.columns:
        if col not in date_cols:
            try:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace(',', '.').str.extract(r'([-]?\d+\.?\d*)')[0]
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    return df, date_cols

def preencher_dados_ausentes(df, metodo='media'):
    """Preenche dados ausentes com diferentes métodos"""
    df = df.copy()
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    
    if metodo == 'media':
        for col in colunas_numericas:
            df[col] = df[col].fillna(df[col].mean())
    elif metodo == 'mediana':
        for col in colunas_numericas:
            df[col] = df[col].fillna(df[col].median())
    elif metodo == 'interpolacao_linear':
        for col in colunas_numericas:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
    elif metodo == 'interpolacao_cubica':
        for col in colunas_numericas:
            df[col] = df[col].interpolate(method='cubic', limit_direction='both')
    elif metodo == 'ffill':
        df[colunas_numericas] = df[colunas_numericas].fillna(method='ffill')
    elif metodo == 'bfill':
        df[colunas_numericas] = df[colunas_numericas].fillna(method='bfill')
    
    return df

def agregar_dados_temporais(df, coluna_data, frequencia='D'):
    """Agrega dados por período temporal"""
    df = df.copy()
    df[coluna_data] = pd.to_datetime(df[coluna_data], errors='coerce')
    df.set_index(coluna_data, inplace=True)
    
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    
    # Dicionário de agregações
    agg_dict = {
        'mean': 'Média',
        'std': 'Desvio Padrão',
        'min': 'Mínimo',
        'max': 'Máximo',
        'sum': 'Soma',
        'count': 'Contagem'
    }
    
    if frequencia == 'D':
        df_agg = df[colunas_numericas].resample('D').agg(list(agg_dict.keys()))
    elif frequencia == 'W':
        df_agg = df[colunas_numericas].resample('W').agg(list(agg_dict.keys()))
    elif frequencia == 'M':
        df_agg = df[colunas_numericas].resample('M').agg(list(agg_dict.keys()))
    elif frequencia == 'Y':
        df_agg = df[colunas_numericas].resample('Y').agg(list(agg_dict.keys()))
    
    return df_agg.round(3)

# ============================================================
# INTERFACE DO USUÁRIO
# ============================================================

# Header Hero
st.markdown("""
<div class="hero-header">
    <h1>🌱 AgroDataLab</h1>
    <p>Sistema Inteligente de Análise de Dados Meteorológicos</p>
    <p style="font-size: 0.9rem; opacity: 0.8; margin-top: 1rem;">
        Desenvolvido para processamento e análise de dados do INMET
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/weather.png", width=80)
    st.markdown("---")
    
    st.markdown("### 📊 Painel de Controle")
    
    # Status dos dados
    if 'df_processed' in st.session_state and st.session_state['df_processed'] is not None:
        df_status = st.session_state['df_processed']
        st.success(f"✅ **Dados Carregados**")
        st.metric("Registros", f"{len(df_status):,}")
        st.metric("Variáveis", len(df_status.columns))
        
        ausentes = df_status.isnull().sum().sum()
        pct_ausentes = (ausentes / (len(df_status) * len(df_status.columns))) * 100
        st.metric("Dados Ausentes", f"{pct_ausentes:.1f}%")
    else:
        st.info("📤 Aguardando upload de dados...")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ Informações")
    st.markdown("""
    **Versão:** 2.0  
    **Licença:** MIT  
    **GitHub:** [Repositório](#)
    """)
    
    st.markdown("---")
    
    st.markdown("### 📚 Guia Rápido")
    st.markdown("""
    1. **Upload** - Carregue seus dados
    2. **Tratamento** - Limpe e prepare
    3. **Análise** - Estatísticas detalhadas
    4. **Visualização** - Gráficos interativos
    5. **Download** - Exporte os resultados
    """)

# ============================================================
# ABAS PRINCIPAIS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 **Upload & Preview**",
    "🔧 **Tratamento de Dados**", 
    "📊 **Análise Estatística**",
    "📈 **Visualização Gráfica**",
    "💾 **Download & Exportação**"
])

# ============================================================
# ABA 1: UPLOAD & PREVIEW
# ============================================================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Carregamento de Dados")
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Arraste ou selecione seu arquivo de dados",
            type=['csv', 'xlsx', 'xls'],
            help="Formatos suportados: CSV, Excel (.xlsx, .xls)\nDados do INMET são automaticamente reconhecidos"
        )
    
    with col2:
        st.markdown("### 📋 Formatos Aceitos")
        st.markdown("""
        <div class="modern-card">
            <h4>✅ CSV</h4>
            <p style="font-size:0.9rem;">Arquivos texto com separadores vírgula, ponto-e-vírgula ou tab</p>
            
            <h4>✅ Excel</h4>
            <p style="font-size:0.9rem;">Planilhas .xlsx e .xls</p>
            
            <h4>✅ INMET</h4>
            <p style="font-size:0.9rem;">Dados meteorológicos do INMET reconhecidos automaticamente</p>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        with st.spinner('🔄 Processando arquivo...'):
            df = carregar_dados(uploaded_file)
            
            if df is not None:
                # Processar dados
                df, date_cols = processar_dados_inmet(df)
                
                # Salvar no session state
                st.session_state['df_original'] = df.copy()
                st.session_state['df_processed'] = df.copy()
                st.session_state['date_columns'] = date_cols
                
                # Sucesso
                st.markdown("""
                <div class="success-box">
                    <h3>✅ Dados Carregados com Sucesso!</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Métricas rápidas
                st.markdown("### 📊 Resumo dos Dados")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Total de Registros</p>
                        <p class="metric-value">{len(df):,}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Variáveis</p>
                        <p class="metric-value">{len(df.columns)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Dados Ausentes</p>
                        <p class="metric-value">{df.isnull().sum().sum():,}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    memoria = df.memory_usage(deep=True).sum() / 1024**2
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Memória</p>
                        <p class="metric-value">{memoria:.1f} MB</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Preview dos dados
                st.markdown("---")
                st.markdown("### 👀 Visualização dos Dados")
                
                # Mostrar dataframe com estilo
                st.dataframe(
                    df.head(15),
                    use_container_width=True,
                    height=400
                )
                
                # Informações das colunas
                with st.expander("🔍 Informações Detalhadas das Colunas"):
                    col_info = pd.DataFrame({
                        'Coluna': df.columns,
                        'Tipo': df.dtypes.values.astype(str),
                        'Não Nulos': df.count().values,
                        'Nulos': df.isnull().sum().values,
                        'Únicos': df.nunique().values,
                        'Memória (KB)': (df.memory_usage(deep=True) / 1024).values.round(2)
                    })
                    st.dataframe(col_info, use_container_width=True)
                
                # Colunas de data identificadas
                if date_cols:
                    st.success(f"📅 Colunas de data identificadas: {', '.join(date_cols)}")

# ============================================================
# ABA 2: TRATAMENTO DE DADOS
# ============================================================
with tab2:
    if 'df_processed' in st.session_state and st.session_state['df_processed'] is not None:
        df = st.session_state['df_processed']
        
        st.markdown("### 🔧 Tratamento e Limpeza de Dados")
        st.markdown("---")
        
        # Diagnóstico de dados ausentes
        missing_count = df.isnull().sum().sum()
        total_cells = len(df) * len(df.columns)
        missing_pct = (missing_count / total_cells * 100) if total_cells > 0 else 0
        
        if missing_count > 0:
            st.markdown(f"""
            <div class="warning-box">
                <h3>⚠️ Dados Ausentes Detectados</h3>
                <p><strong>{missing_count:,}</strong> valores ausentes ({missing_pct:.2f}% do total)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Tabela de dados ausentes
            st.markdown("#### 📋 Colunas com Dados Ausentes")
            
            missing_df = pd.DataFrame({
                'Coluna': df.columns,
                'Valores Ausentes': df.isnull().sum().values,
                'Porcentagem': (df.isnull().sum() / len(df) * 100).round(2).values
            })
            missing_df = missing_df[missing_df['Valores Ausentes'] > 0].sort_values('Valores Ausentes', ascending=False)
            
            st.dataframe(
                missing_df.style.background_gradient(cmap='Reds', subset=['Porcentagem']),
                use_container_width=True
            )
            
            # Métodos de preenchimento
            st.markdown("---")
            st.markdown("### 🛠️ Métodos de Preenchimento")
            
            st.markdown("""
            <div class="info-box">
                <h4>📚 Fundamentação Científica</h4>
                <ul>
                    <li><strong>Média:</strong> Indicado para dados com distribuição normal (Little & Rubin, 2019)</li>
                    <li><strong>Mediana:</strong> Robusto para dados com outliers (Tukey, 1977)</li>
                    <li><strong>Interpolação Linear:</strong> Ideal para séries temporais (Press et al., 2007)</li>
                    <li><strong>Forward/Backward Fill:</strong> Útil para dados sequenciais</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="modern-card">
                    <h4>📊 Média Aritmética</h4>
                    <p style="font-size:0.85rem; color:#666;">Substitui valores ausentes pela média da coluna</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("📊 Aplicar Média", key="btn_mean", use_container_width=True):
                    st.session_state['df_processed'] = preencher_dados_ausentes(df.copy(), 'media')
                    st.success("✅ Dados preenchidos com média!")
                    st.rerun()
            
            with col2:
                st.markdown("""
                <div class="modern-card">
                    <h4>📈 Mediana</h4>
                    <p style="font-size:0.85rem; color:#666;">Substitui valores ausentes pela mediana da coluna</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("📈 Aplicar Mediana", key="btn_median", use_container_width=True):
                    st.session_state['df_processed'] = preencher_dados_ausentes(df.copy(), 'mediana')
                    st.success("✅ Dados preenchidos com mediana!")
                    st.rerun()
            
            with col3:
                st.markdown("""
                <div class="modern-card">
                    <h4>🔄 Interpolação</h4>
                    <p style="font-size:0.85rem; color:#666;">Interpolação linear entre valores adjacentes</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("🔄 Aplicar Interpolação", key="btn_interp", use_container_width=True):
                    st.session_state['df_processed'] = preencher_dados_ausentes(df.copy(), 'interpolacao_linear')
                    st.success("✅ Dados interpolados!")
                    st.rerun()
            
            col4, col5 = st.columns(2)
            
            with col4:
                st.markdown("""
                <div class="modern-card">
                    <h4>🗑️ Remover Linhas</h4>
                    <p style="font-size:0.85rem; color:#666;">Remove linhas com qualquer valor ausente</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("🗑️ Remover Linhas Incompletas", key="btn_drop", use_container_width=True):
                    df_dropped = df.dropna()
                    st.session_state['df_processed'] = df_dropped
                    removidas = len(df) - len(df_dropped)
                    st.success(f"✅ {removidas} linhas removidas!")
                    st.rerun()
            
            with col5:
                st.markdown("""
                <div class="modern-card">
                    <h4>🔙 Restaurar Original</h4>
                    <p style="font-size:0.85rem; color:#666;">Volta aos dados originais sem modificações</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("🔙 Restaurar Dados Originais", key="btn_restore", use_container_width=True):
                    st.session_state['df_processed'] = st.session_state['df_original'].copy()
                    st.success("✅ Dados restaurados ao estado original!")
                    st.rerun()
            
            # Visualização do impacto
            st.markdown("---")
            st.markdown("### 📊 Visualização do Impacto")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Antes do tratamento
                st.markdown("**Antes do Tratamento**")
                missing_before = st.session_state.get('df_original', df).isnull().sum()
                missing_before = missing_before[missing_before > 0]
                if len(missing_before) > 0:
                    chart_before = pd.DataFrame({
                        'Coluna': missing_before.index,
                        'Ausentes': missing_before.values
                    })
                    bars = criar_grafico_barras(chart_before, 'Coluna', 'Ausentes', 'Dados Ausentes - Antes')
                    st.altair_chart(bars, use_container_width=True)
            
            with col2:
                # Depois do tratamento
                st.markdown("**Após o Tratamento**")
                missing_after = df.isnull().sum() if 'df_processed' in st.session_state else df.isnull().sum()
                if missing_after.sum() == 0:
                    st.success("✅ Nenhum dado ausente após tratamento!")
                else:
                    missing_after = missing_after[missing_after > 0]
                    chart_after = pd.DataFrame({
                        'Coluna': missing_after.index,
                        'Ausentes': missing_after.values
                    })
                    bars = criar_grafico_barras(chart_after, 'Coluna', 'Ausentes', 'Dados Ausentes - Depois')
                    st.altair_chart(bars, use_container_width=True)
        else:
            st.markdown("""
            <div class="success-box">
                <h3>✅ Dados Completos!</h3>
                <p>Nenhum valor ausente detectado. Seus dados estão prontos para análise.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Outlier Detection
        st.markdown("---")
        st.markdown("### 🔍 Detecção de Outliers (Método IQR)")
        st.markdown("""
        <div class="info-box">
            <p style="margin:0;"><strong>Método:</strong> Intervalo Interquartil (IQR) - Tukey (1977)</p>
            <p style="margin:0; font-size:0.9rem;">Limite inferior = Q1 - 1.5 × IQR | Limite superior = Q3 + 1.5 × IQR</p>
        </div>
        """, unsafe_allow_html=True)
        
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        if colunas_numericas:
            col_outlier = st.selectbox("Selecione a variável para análise de outliers:", colunas_numericas)
            
            dados_coluna = df[col_outlier].dropna()
            outliers, lim_inf, lim_sup = detectar_outliers_iqr(dados_coluna)
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                st.warning(f"Foram detectados {n_outliers} outliers em {col_outlier} ({n_outliers/len(dados_coluna)*100:.1f}% dos dados)")
                
                # Visualização boxplot
                box = criar_boxplot(df.dropna(subset=[col_outlier]), col_outlier)
                st.altair_chart(box, use_container_width=True)
                
                st.markdown(f"""
                - **Limite Inferior:** {lim_inf:.4f}
                - **Limite Superior:** {lim_sup:.4f}
                - **Outliers:** {n_outliers} valores
                """)
            else:
                st.success(f"Nenhum outlier detectado em {col_outlier}")
    else:
        st.markdown("""
        <div class="info-box">
            <h3>📤 Nenhum dado carregado</h3>
            <p>Vá para a aba <strong>Upload & Preview</strong> e carregue seus dados primeiro.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# ABA 3: ANÁLISE ESTATÍSTICA
# ============================================================
with tab3:
    if 'df_processed' in st.session_state and st.session_state['df_processed'] is not None:
        df = st.session_state['df_processed']
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        st.markdown("### 📊 Análise Estatística Detalhada")
        st.markdown("---")
        
        if colunas_numericas:
            # Seleção de variável
            col_analise = st.selectbox(
                "Selecione a variável para análise:",
                colunas_numericas,
                help="Escolha uma variável numérica para análise estatística completa"
            )
            
            # Estatísticas descritivas
            st.markdown(f"#### 📈 Estatísticas Descritivas: {col_analise}")
            
            stats = calcular_estatisticas_descritivas(df, col_analise)
            
            # Exibição em cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Média</p>
                    <p class="metric-value">{stats['Média']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Desvio Padrão</p>
                    <p class="metric-value">{stats['Desvio Padrão']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                classificacao = classificar_cv(stats['CV (%)'])
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">CV (%)</p>
                    <p class="metric-value">{stats['CV (%)']:.1f}%</p>
                    <p style="font-size:0.8rem;">{classificacao}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Amostras</p>
                    <p class="metric-value">{stats['n']:,}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Tabela completa
            with st.expander("📋 Ver Todas as Estatísticas"):
                stats_df = pd.DataFrame([stats]).T
                stats_df.columns = ['Valor']
                st.dataframe(stats_df.style.highlight_max(axis=0), use_container_width=True)
            
            # Análise de correlação
            if len(colunas_numericas) >= 2:
                st.markdown("---")
                st.markdown("### 🔗 Análise de Correlação")
                
                st.markdown("""
                <div class="info-box">
                    <h4>📚 Coeficiente de Correlação de Pearson</h4>
                    <p>Mede a força e direção da relação linear entre duas variáveis.</p>
                    <ul>
                        <li><strong>r = 1:</strong> Correlação positiva perfeita</li>
                        <li><strong>r = 0:</strong> Sem correlação linear</li>
                        <li><strong>r = -1:</strong> Correlação negativa perfeita</li>
                    </ul>
                    <p style="font-size:0.9rem;"><strong>Referência:</strong> Pearson, K. (1895). Proceedings of the Royal Society of London.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Matriz de correlação
                st.markdown("#### Matriz de Correlação Completa")
                
                corr_matrix = df[colunas_numericas].corr()
                
                # Estilizar matriz
                st.dataframe(
                    corr_matrix.style.background_gradient(
                        cmap='RdYlGn',
                        vmin=-1,
                        vmax=1
                    ).format("{:.3f}"),
                    use_container_width=True
                )
                
                # Heatmap da correlação
                st.markdown("#### Mapa de Calor das Correlações")
                
                # Preparar dados para heatmap
                corr_melted = corr_matrix.reset_index().melt(
                    id_vars='index',
                    var_name='Variável 2',
                    value_name='Correlação'
                )
                corr_melted.columns = ['Variável 1', 'Variável 2', 'Correlação']
                
                heatmap = alt.Chart(corr_melted).mark_rect().encode(
                    x=alt.X('Variável 1:O', title=None),
                    y=alt.Y('Variável 2:O', title=None),
                    color=alt.Color('Correlação:Q', scale=alt.Scale(scheme='rdylgn'), legend=alt.Legend(title='r')),
                    tooltip=['Variável 1', 'Variável 2', alt.Tooltip('Correlação:Q', format='.3f')]
                ).properties(
                    title=alt.TitleParams(text='Matriz de Correlação - Heatmap', fontSize=16, color='#1a5632'),
                    width=600,
                    height=500
                )
                
                st.altair_chart(heatmap, use_container_width=True)
                
                # Interpretação
                st.markdown("#### 📖 Interpretação da Correlação")
                
                interpretacao = """
                | Valor de r | Força da Correlação |
                |------------|---------------------|
                | 0.00 - 0.30 | Fraca |
                | 0.30 - 0.70 | Moderada |
                | 0.70 - 1.00 | Forte |
                """
                st.markdown(interpretacao)
            
            # Análise temporal (se houver colunas de data)
            if 'date_columns' in st.session_state and len(st.session_state['date_columns']) > 0:
                st.markdown("---")
                st.markdown("### 📅 Análise Temporal")
                
                coluna_data = st.selectbox(
                    "Selecione a coluna de data:",
                    st.session_state['date_columns']
                )
                
                frequencia = st.selectbox(
                    "Período de agregação:",
                    ['D', 'W', 'M', 'Y'],
                    format_func=lambda x: {'D': 'Diário', 'W': 'Semanal', 'M': 'Mensal', 'Y': 'Anual'}[x]
                )
                
                try:
                    df_agg = agregar_dados_temporais(df, coluna_data, frequencia)
                    
                    st.markdown(f"#### Dados Agregados ({frequencia})")
                    st.dataframe(df_agg, use_container_width=True, height=400)
                    
                    # Gráfico temporal
                    if col_analise in colunas_numericas:
                        # Gráfico de linha temporal
                        df_temp = df[[coluna_data, col_analise]].copy()
                        df_temp[coluna_data] = pd.to_datetime(df_temp[coluna_data], errors='coerce')
                        df_temp = df_temp.dropna()
                        
                        chart = criar_grafico_linhas(
                            df_temp,
                            coluna_data,
                            [col_analise],
                            f'Série Temporal - {col_analise}'
                        )
                        st.altair_chart(chart, use_container_width=True)
                        
                        # Média móvel
                        df_temp_sorted = df_temp.sort_values(coluna_data)
                        df_temp_sorted['Média Móvel (7 dias)'] = calcular_media_movel(
                            df_temp_sorted[col_analise].values,
                            janela=7
                        )
                        
                        chart_mm = criar_grafico_linhas(
                            df_temp_sorted,
                            coluna_data,
                            [col_analise, 'Média Móvel (7 dias)'],
                            f'Série Temporal com Média Móvel - {col_analise}'
                        )
                        st.altair_chart(chart_mm, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Erro na análise temporal: {str(e)}")
        else:
            st.warning("Nenhuma variável numérica encontrada nos dados.")
    else:
        st.info("📤 Carregue seus dados na Aba 1 primeiro!")

# ============================================================
# ABA 4: VISUALIZAÇÃO GRÁFICA
# ============================================================
with tab4:
    if 'df_processed' in st.session_state and st.session_state['df_processed'] is not None:
        df = st.session_state['df_processed']
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        st.markdown("### 📈 Visualização Gráfica Interativa")
        st.markdown("---")
        
        if colunas_numericas:
            # Seleção do tipo de gráfico
            tipo_grafico = st.radio(
                "Selecione o tipo de visualização:",
                ['📈 Série Temporal', '📊 Histograma', '📉 Boxplot', '📊 Barras'],
                horizontal=True
            )
            
            if tipo_grafico == '📈 Série Temporal':
                st.markdown("#### Série Temporal")
                
                date_cols = st.session_state.get('date_columns', [])
                if date_cols:
                    x_col = st.selectbox("Eixo X (Data):", date_cols)
                    y_cols = st.multiselect(
                        "Variáveis para visualizar:",
                        colunas_numericas,
                        default=colunas_numericas[:3] if len(colunas_numericas) >= 3 else colunas_numericas
                    )
                    
                    if y_cols:
                        df_temp = df[[x_col] + y_cols].copy()
                        df_temp[x_col] = pd.to_datetime(df_temp[x_col], errors='coerce')
                        df_temp = df_temp.dropna()
                        
                        chart = criar_grafico_linhas(df_temp, x_col, y_cols, "Série Temporal")
                        st.altair_chart(chart, use_container_width=True)
                        
                        # Estatísticas rápidas
                        with st.expander("📊 Estatísticas das Variáveis"):
                            for y_col in y_cols:
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(f"{y_col} - Média", f"{df_temp[y_col].mean():.2f}")
                                with col2:
                                    st.metric(f"{y_col} - Máx", f"{df_temp[y_col].max():.2f}")
                                with col3:
                                    st.metric(f"{y_col} - Mín", f"{df_temp[y_col].min():.2f}")
                                with col4:
                                    st.metric(f"{y_col} - Desv", f"{df_temp[y_col].std():.2f}")
                else:
                    st.info("Nenhuma coluna de data identificada para gráfico temporal.")
            
            elif tipo_grafico == '📊 Histograma':
                st.markdown("#### Histograma de Distribuição")
                
                col_hist = st.selectbox("Selecione a variável:", colunas_numericas)
                n_bins = st.slider("Número de intervalos:", 5, 100, 30)
                
                chart = criar_histograma(df.dropna(subset=[col_hist]), col_hist, bins=n_bins)
                st.altair_chart(chart, use_container_width=True)
                
                # Estatísticas da distribuição
                stats_hist = calcular_estatisticas_descritivas(df, col_hist)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Assimetria", f"{stats_hist['Assimetria']:.3f}")
                col2.metric("Curtose", f"{stats_hist['Curtose']:.3f}")
                col3.metric("IQR", f"{stats_hist['IQR']:.3f}")
            
            elif tipo_grafico == '📉 Boxplot':
                st.markdown("#### Diagrama de Caixa (Boxplot)")
                
                col_box = st.selectbox("Variável:", colunas_numericas)
                
                # Opção de agrupamento
                colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if colunas_categoricas:
                    usar_grupo = st.checkbox("Agrupar por variável categórica")
                    if usar_grupo:
                        col_grupo = st.selectbox("Agrupar por:", colunas_categoricas)
                        chart = criar_boxplot(df.dropna(subset=[col_box, col_grupo]), col_box, col_grupo)
                    else:
                        chart = criar_boxplot(df.dropna(subset=[col_box]), col_box)
                else:
                    chart = criar_boxplot(df.dropna(subset=[col_box]), col_box)
                
                st.altair_chart(chart, use_container_width=True)
            
            elif tipo_grafico == '📊 Barras':
                st.markdown("#### Gráfico de Barras")
                
                date_cols = st.session_state.get('date_columns', [])
                all_cols = date_cols + colunas_numericas
                
                x_col = st.selectbox("Eixo X:", all_cols)
                y_col = st.selectbox("Eixo Y:", colunas_numericas)
                
                chart = criar_grafico_barras(df.dropna(subset=[x_col, y_col]), x_col, y_col)
                st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("Nenhuma variável numérica disponível para visualização.")
    else:
        st.info("📤 Carregue seus dados na Aba 1 primeiro!")

# ============================================================
# ABA 5: DOWNLOAD & EXPORTAÇÃO
# ============================================================
with tab5:
    if 'df_processed' in st.session_state and st.session_state['df_processed'] is not None:
        df = st.session_state['df_processed']
        
        st.markdown("### 💾 Download e Exportação")
        st.markdown("---")
        
        st.markdown("""
        <div class="success-box">
            <h3>✅ Dados Prontos para Download</h3>
            <p>Seus dados foram processados e estão prontos para exportação.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Resumo dos dados
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Registros", f"{len(df):,}")
        with col2:
            st.metric("📋 Colunas", len(df.columns))
        with col3:
            st.metric("🔍 Ausentes", df.isnull().sum().sum())
        with col4:
            memoria = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("💾 Tamanho", f"{memoria:.1f} MB")
        
        st.markdown("---")
        
        # Opções de download
        st.markdown("### 📥 Formatos de Download")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="modern-card">
                <h3>📄 CSV</h3>
                <p style="color:#666;">Formato universal, compatível com Excel, Google Sheets, Python, R</p>
                <ul>
                    <li>Encoding: UTF-8</li>
                    <li>Separador: Vírgula</li>
                    <li>Inclui cabeçalho</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            csv = df.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"dados_processados_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_csv"
            )
            
            # Preview do CSV
            with st.expander("👀 Preview do CSV"):
                st.code(csv[:1000], language='csv')
        
        with col2:
            st.markdown("""
            <div class="modern-card">
                <h3>📊 Excel</h3>
                <p style="color:#666;">Formato nativo Microsoft Excel com múltiplas abas</p>
                <ul>
                    <li>Aba 1: Dados Processados</li>
                    <li>Aba 2: Estatísticas</li>
                    <li>Aba 3: Metadados</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Criar Excel com múltiplas abas
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Aba de dados
                df.to_excel(writer, sheet_name='Dados_Processados', index=False)
                
                # Aba de estatísticas
                df.describe().to_excel(writer, sheet_name='Estatísticas')
                
                # Aba de metadados
                metadata = pd.DataFrame({
                    'Informação': ['Data de Processamento', 'Total de Registros', 'Total de Colunas', 
                                  'Colunas Numéricas', 'Dados Ausentes'],
                    'Valor': [datetime.now().strftime('%d/%m/%Y %H:%M'), len(df), len(df.columns),
                            len(df.select_dtypes(include=[np.number]).columns), df.isnull().sum().sum()]
                })
                metadata.to_excel(writer, sheet_name='Metadados', index=False)
            
            st.download_button(
                label="📥 Download Excel",
                data=output.getvalue(),
                file_name=f"dados_processados_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel"
            )
        
        st.markdown("---")
        
        # Preview final dos dados
        st.markdown("### 📋 Preview dos Dados Processados")
        st.dataframe(df.head(20), use_container_width=True, height=400)
        
        # Resumo das transformações
        if 'df_original' in st.session_state and st.session_state['df_original'] is not None:
            df_orig = st.session_state['df_original']
            
            st.markdown("---")
            st.markdown("### 📊 Resumo das Transformações")
            
            transformacoes = pd.DataFrame({
                'Métrica': [
                    'Registros Originais',
                    'Registros Finais',
                    'Registros Removidos',
                    'Dados Ausentes (Original)',
                    'Dados Ausentes (Final)',
                    'Dados Ausentes Tratados',
                    'Memória Original',
                    'Memória Final'
                ],
                'Valor': [
                    len(df_orig),
                    len(df),
                    len(df_orig) - len(df),
                    df_orig.isnull().sum().sum(),
                    df.isnull().sum().sum(),
                    df_orig.isnull().sum().sum() - df.isnull().sum().sum(),
                    f"{df_orig.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
                    f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
                ]
            })
            
            st.dataframe(transformacoes, use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h3>📤 Nenhum dado disponível</h3>
            <p>Carregue e processe seus dados nas abas anteriores antes de fazer o download.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div class="app-footer">
    <h4>🌱 AgroDataLab v2.0</h4>
    <p>Sistema de Análise de Dados Meteorológicos</p>
    <p style="font-size:0.85rem;">
        Desenvolvido com Streamlit, Pandas, NumPy e Altair<br>
        Métodos estatísticos baseados em literatura científica<br>
        Licença MIT © 2024
    </p>
    <p style="font-size:0.8rem; color:#adb5bd;">
        Referências: Pearson (1895) | Tukey (1977) | Pimentel-Gomes (2000) | Little & Rubin (2019)
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# FIM DO CÓDIGO
# ============================================================
