# ============================================================
# AgroDataLab - Sistema Completo de Análise Meteorológica
# Versão Corrigida - 100% Estável para Streamlit Cloud
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÃO AVANÇADA DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="AgroDataLab | Análise Meteorológica",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS PROFISSIONAL - DESIGN MODERNO E MINIMALISTA
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
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
    
    .app-footer {
        text-align: center;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 12px;
        margin-top: 3rem;
        color: #6c757d;
        border: 1px solid #e9ecef;
    }
    
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
    """Calcula estatísticas descritivas completas"""
    dados = df[coluna].dropna()
    
    if len(dados) == 0:
        return None
    
    n = len(dados)
    media = np.mean(dados)
    mediana = np.median(dados)
    desvio_padrao = np.std(dados, ddof=1)
    variancia = np.var(dados, ddof=1)
    minimo = np.min(dados)
    maximo = np.max(dados)
    amplitude = maximo - minimo
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1
    cv = (desvio_padrao / media * 100) if media != 0 else 0
    assimetria = np.sum((dados - media) ** 3) / (n * desvio_padrao ** 3) if desvio_padrao > 0 else 0
    curtose = np.sum((dados - media) ** 4) / (n * desvio_padrao ** 4) - 3 if desvio_padrao > 0 else 0
    
    return {
        'n': n, 'Média': media, 'Mediana': mediana,
        'Desvio Padrão': desvio_padrao, 'Variância': variancia,
        'Mínimo': minimo, 'Máximo': maximo, 'Amplitude': amplitude,
        'Q1 (25%)': q1, 'Q3 (75%)': q3, 'IQR': iqr,
        'CV (%)': cv, 'Assimetria': assimetria, 'Curtose': curtose
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
    
    lines = alt.Chart(df_melted).mark_line(
        strokeWidth=2.5,
        point=alt.OverlayMarkDef(size=40, filled=True)
    ).encode(
        x=alt.X(f'{x_col}:T', title='Data', axis=alt.Axis(grid=False)),
        y=alt.Y('Valor:Q', title='Valor', axis=alt.Axis(grid=True, gridDash=[2,2])),
        color=alt.Color('Variável:N', legend=alt.Legend(orient='bottom', title=None, labelFontSize=12)),
        tooltip=[
            alt.Tooltip(f'{x_col}:T', title='Data'),
            alt.Tooltip('Variável:N', title='Variável'),
            alt.Tooltip('Valor:Q', title='Valor', format='.2f')
        ]
    ).properties(
        title=alt.TitleParams(text=titulo, fontSize=18, fontWeight='bold', color='#1a5632'),
        height=450
    ).interactive(bind_y=False)
    
    return lines

def criar_grafico_barras(df, x_col, y_col, titulo="Gráfico de Barras"):
    """Cria gráfico de barras estilizado"""
    bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, opacity=0.85).encode(
        x=alt.X(f'{x_col}:O', title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f'{y_col}:Q', title=y_col),
        color=alt.Color(f'{y_col}:Q', scale=alt.Scale(scheme='greens')),
        tooltip=[x_col, alt.Tooltip(y_col, format='.2f')]
    ).properties(
        title=alt.TitleParams(text=titulo, fontSize=16, fontWeight='bold', color='#1a5632'),
        height=400
    )
    
    return bars

def criar_histograma(df, coluna, bins=30):
    """Cria histograma"""
    hist = alt.Chart(df).mark_bar(opacity=0.7, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        alt.X(f'{coluna}:Q', bin=alt.Bin(maxbins=bins), title=coluna),
        alt.Y('count()', title='Frequência'),
        tooltip=['count()']
    ).properties(
        title=alt.TitleParams(text=f'Distribuição de {coluna}', fontSize=16, fontWeight='bold', color='#1a5632'),
        height=400
    )
    
    return hist

def criar_boxplot(df, coluna, grupo=None):
    """Cria boxplot"""
    if grupo:
        box = alt.Chart(df).mark_boxplot(extent='min-max', color='#2d8a4e').encode(
            x=alt.X(f'{grupo}:O', title=grupo),
            y=alt.Y(f'{coluna}:Q', title=coluna),
            color=alt.Color(f'{grupo}:N', legend=None)
        ).properties(height=400)
    else:
        box = alt.Chart(df).mark_boxplot(extent='min-max', color='#2d8a4e').encode(
            y=alt.Y(f'{coluna}:Q', title=coluna)
        ).properties(height=400)
    
    return box

# ============================================================
# FUNÇÕES DE PROCESSAMENTO DE DADOS
# ============================================================

def carregar_dados(arquivo):
    """Carrega dados de diferentes formatos"""
    try:
        if arquivo.name.endswith('.csv'):
            for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(arquivo, encoding=encoding)
                    if df.shape[1] == 1:
                        for sep in [';', '\t', '|']:
                            try:
                                df = pd.read_csv(arquivo, encoding=encoding, sep=sep)
                                if df.shape[1] > 1:
                                    break
                            except:
                                continue
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
    
    date_patterns = ['data', 'hora', 'date', 'time', 'ano', 'mes', 'dia']
    date_cols = []
    
    for col in df.columns:
        if any(pattern in col.lower() for pattern in date_patterns):
            date_cols.append(col)
            try:
                df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce')
            except:
                pass
    
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
    """Preenche dados ausentes"""
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
    
    agg_dict = ['mean', 'std', 'min', 'max', 'sum', 'count']
    
    freq_map = {'D': 'D', 'W': 'W', 'M': 'M', 'Y': 'Y'}
    df_agg = df[colunas_numericas].resample(freq_map.get(frequencia, 'D')).agg(agg_dict)
    
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
    st.markdown("### 📊 Painel de Controle")
    
    if 'df_processed' in st.session_state and st.session_state['df_processed'] is not None:
        df_status = st.session_state['df_processed']
        st.success(f"✅ Dados Carregados")
        st.metric("Registros", f"{len(df_status):,}")
        st.metric("Variáveis", len(df_status.columns))
        
        ausentes = df_status.isnull().sum().sum()
        pct_ausentes = (ausentes / (len(df_status) * len(df_status.columns))) * 100 if len(df_status) > 0 else 0
        st.metric("Dados Ausentes", f"{pct_ausentes:.1f}%")
    else:
        st.info("📤 Aguardando upload de dados...")
    
    st.markdown("---")
    st.markdown("### ℹ️ Informações")
    st.markdown("**Versão:** 2.0 | **Licença:** MIT")
    
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
    "📤 Upload & Preview",
    "🔧 Tratamento de Dados", 
    "📊 Análise Estatística",
    "📈 Visualização Gráfica",
    "💾 Download & Exportação"
])

# ============================================================
# ABA 1: UPLOAD & PREVIEW
# ============================================================
with tab1:
    st.markdown("### 📤 Carregamento de Dados")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Arraste ou selecione seu arquivo de dados",
        type=['csv', 'xlsx', 'xls'],
        help="Formatos suportados: CSV, Excel (.xlsx, .xls)"
    )
    
    if uploaded_file is not None:
        with st.spinner('🔄 Processando arquivo...'):
            df = carregar_dados(uploaded_file)
            
            if df is not None:
                df, date_cols = processar_dados_inmet(df)
                
                st.session_state['df_original'] = df.copy()
                st.session_state['df_processed'] = df.copy()
                st.session_state['date_columns'] = date_cols
                
                st.markdown("""
                <div class="success-box">
                    <h3>✅ Dados Carregados com Sucesso!</h3>
                </div>
                """, unsafe_allow_html=True)
                
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
                
                st.markdown("---")
                st.markdown("### 👀 Visualização dos Dados")
                
                st.dataframe(df.head(15), use_container_width=True, height=400)
                
                # Informações das colunas - CORRIGIDO
                with st.expander("🔍 Informações Detalhadas das Colunas"):
                    try:
                        n_cols = len(df.columns)
                        col_info = pd.DataFrame({
                            'Coluna': df.columns.tolist(),
                            'Tipo': [str(dtype) for dtype in df.dtypes.values],
                            'Não Nulos': df.count().tolist(),
                            'Nulos': df.isnull().sum().tolist(),
                            'Únicos': df.nunique().tolist()
                        })
                        st.dataframe(col_info, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erro ao gerar informações: {str(e)}")
                
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
            
            st.markdown("#### 📋 Colunas com Dados Ausentes")
            
            try:
                missing_df = pd.DataFrame({
                    'Coluna': df.columns,
                    'Valores Ausentes': df.isnull().sum().values,
                    'Porcentagem': (df.isnull().sum() / len(df) * 100).round(2).values
                })
                missing_df = missing_df[missing_df['Valores Ausentes'] > 0].sort_values('Valores Ausentes', ascending=False)
                st.dataframe(missing_df, use_container_width=True)
            except:
                st.write("Colunas com dados ausentes:")
                st.write(df.isnull().sum()[df.isnull().sum() > 0])
            
            st.markdown("---")
            st.markdown("### 🛠️ Métodos de Preenchimento")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Aplicar Média", key="btn_mean"):
                    st.session_state['df_processed'] = preencher_dados_ausentes(df.copy(), 'media')
                    st.success("✅ Dados preenchidos com média!")
                    st.rerun()
            
            with col2:
                if st.button("📈 Aplicar Mediana", key="btn_median"):
                    st.session_state['df_processed'] = preencher_dados_ausentes(df.copy(), 'mediana')
                    st.success("✅ Dados preenchidos com mediana!")
                    st.rerun()
            
            with col3:
                if st.button("🔄 Aplicar Interpolação", key="btn_interp"):
                    st.session_state['df_processed'] = preencher_dados_ausentes(df.copy(), 'interpolacao_linear')
                    st.success("✅ Dados interpolados!")
                    st.rerun()
            
            col4, col5 = st.columns(2)
            
            with col4:
                if st.button("🗑️ Remover Linhas Incompletas", key="btn_drop"):
                    df_dropped = df.dropna()
                    st.session_state['df_processed'] = df_dropped
                    removidas = len(df) - len(df_dropped)
                    st.success(f"✅ {removidas} linhas removidas!")
                    st.rerun()
            
            with col5:
                if st.button("🔙 Restaurar Dados Originais", key="btn_restore"):
                    st.session_state['df_processed'] = st.session_state['df_original'].copy()
                    st.success("✅ Dados restaurados ao estado original!")
                    st.rerun()
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
        
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        if colunas_numericas:
            col_outlier = st.selectbox("Selecione a variável:", colunas_numericas)
            
            dados_coluna = df[col_outlier].dropna()
            if len(dados_coluna) > 0:
                outliers, lim_inf, lim_sup = detectar_outliers_iqr(dados_coluna)
                n_outliers = outliers.sum()
                
                if n_outliers > 0:
                    st.warning(f"Detectados {n_outliers} outliers ({n_outliers/len(dados_coluna)*100:.1f}%)")
                    
                    box = criar_boxplot(df.dropna(subset=[col_outlier]), col_outlier)
                    st.altair_chart(box, use_container_width=True)
                    
                    st.markdown(f"""
                    - **Limite Inferior:** {lim_inf:.4f}
                    - **Limite Superior:** {lim_sup:.4f}
                    """)
                else:
                    st.success(f"Nenhum outlier detectado em {col_outlier}")
    else:
        st.info("📤 Carregue seus dados na Aba 1 primeiro!")

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
            col_analise = st.selectbox("Selecione a variável:", colunas_numericas)
            
            stats = calcular_estatisticas_descritivas(df, col_analise)
            
            if stats:
                st.markdown(f"#### 📈 Estatísticas: {col_analise}")
                
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
                
                # Correlação
                if len(colunas_numericas) >= 2:
                    st.markdown("---")
                    st.markdown("### 🔗 Matriz de Correlação")
                    
                    corr_matrix = df[colunas_numericas].corr()
                    
                    st.dataframe(
                        corr_matrix.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1).format("{:.3f}"),
                        use_container_width=True
                    )
                
                # Análise temporal
                if 'date_columns' in st.session_state and len(st.session_state['date_columns']) > 0:
                    st.markdown("---")
                    st.markdown("### 📅 Análise Temporal")
                    
                    coluna_data = st.selectbox("Coluna de data:", st.session_state['date_columns'])
                    frequencia = st.selectbox("Período:", ['D', 'W', 'M', 'Y'],
                                            format_func=lambda x: {'D': 'Diário', 'W': 'Semanal', 
                                                                  'M': 'Mensal', 'Y': 'Anual'}[x])
                    
                    try:
                        df_agg = agregar_dados_temporais(df, coluna_data, frequencia)
                        st.dataframe(df_agg, use_container_width=True, height=400)
                        
                        # Gráfico temporal
                        df_temp = df[[coluna_data, col_analise]].copy()
                        df_temp[coluna_data] = pd.to_datetime(df_temp[coluna_data], errors='coerce')
                        df_temp = df_temp.dropna()
                        
                        if len(df_temp) > 0:
                            chart = criar_grafico_linhas(df_temp, coluna_data, [col_analise],
                                                        f'Série Temporal - {col_analise}')
                            st.altair_chart(chart, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erro na análise temporal: {str(e)}")
        else:
            st.warning("Nenhuma variável numérica encontrada.")
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
            tipo_grafico = st.radio(
                "Tipo de visualização:",
                ['📈 Série Temporal', '📊 Histograma', '📉 Boxplot'],
                horizontal=True
            )
            
            if tipo_grafico == '📈 Série Temporal':
                date_cols = st.session_state.get('date_columns', [])
                if date_cols:
                    x_col = st.selectbox("Eixo X:", date_cols)
                    y_cols = st.multiselect("Variáveis:", colunas_numericas,
                                           default=colunas_numericas[:2] if len(colunas_numericas) >= 2 else colunas_numericas)
                    
                    if y_cols:
                        df_temp = df[[x_col] + y_cols].copy()
                        df_temp[x_col] = pd.to_datetime(df_temp[x_col], errors='coerce')
                        df_temp = df_temp.dropna()
                        
                        if len(df_temp) > 0:
                            chart = criar_grafico_linhas(df_temp, x_col, y_cols)
                            st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Nenhuma coluna de data identificada.")
            
            elif tipo_grafico == '📊 Histograma':
                col_hist = st.selectbox("Variável:", colunas_numericas)
                n_bins = st.slider("Intervalos:", 5, 100, 30)
                
                chart = criar_histograma(df.dropna(subset=[col_hist]), col_hist, bins=n_bins)
                st.altair_chart(chart, use_container_width=True)
            
            elif tipo_grafico == '📉 Boxplot':
                col_box = st.selectbox("Variável:", colunas_numericas)
                chart = criar_boxplot(df.dropna(subset=[col_box]), col_box)
                st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("Nenhuma variável numérica disponível.")
    else:
        st.info("📤 Carregue seus dados na Aba 1 primeiro!")

# ============================================================
# ABA 5: DOWNLOAD
# ============================================================
with tab5:
    if 'df_processed' in st.session_state and st.session_state['df_processed'] is not None:
        df = st.session_state['df_processed']
        
        st.markdown("### 💾 Download e Exportação")
        st.markdown("---")
        
        st.markdown("""
        <div class="success-box">
            <h3>✅ Dados Prontos para Download</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Registros", f"{len(df):,}")
        with col2:
            st.metric("Colunas", len(df.columns))
        with col3:
            st.metric("Ausentes", df.isnull().sum().sum())
        with col4:
            memoria = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Tamanho", f"{memoria:.1f} MB")
        
        st.markdown("---")
        st.markdown("### 📥 Download")
        
        csv = df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="📥 Baixar CSV",
            data=csv,
            file_name=f"dados_processados_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        st.markdown("### 📋 Preview dos Dados Processados")
        st.dataframe(df.head(20), use_container_width=True, height=400)
    else:
        st.info("📤 Carregue e processe seus dados primeiro!")

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
# INICIALIZAÇÃO
# ============================================================
if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None
if 'df_processed' not in st.session_state:
    st.session_state['df_processed'] = None
if 'date_columns' not in st.session_state:
    st.session_state['date_columns'] = []
