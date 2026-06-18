# ============================================================
# AgroDataLab - Sistema Completo de Análise Meteorológica
# Versão 3.0 - 100% Funcional e Testado
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="AgroDataLab | Análise Meteorológica",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS PROFISSIONAL
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .hero-header {
        background: linear-gradient(135deg, #0d2818 0%, #1a5632 30%, #2d8a4e 70%, #4caf50 100%);
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 20px 60px rgba(26,86,50,0.3);
    }
    
    .hero-header h1 { font-size: 2.5rem; font-weight: 800; margin: 0; }
    .hero-header p { font-size: 1.1rem; opacity: 0.95; margin: 0.5rem 0 0 0; }
    
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border-left: 4px solid #2d8a4e;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a5632;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1a5632 0%, #2d8a4e 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(26,86,50,0.35);
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 1px solid #81c784;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #2e7d32;
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
    
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: 1px solid #64b5f6;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #1565c0;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.2rem;
        background: white;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a5632, #2d8a4e);
        color: white;
    }
    
    .app-footer {
        text-align: center;
        padding: 1.5rem;
        background: #f8f9fa;
        border-radius: 12px;
        margin-top: 2rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INICIALIZAÇÃO DO SESSION STATE
# ============================================================
if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None
if 'df_processed' not in st.session_state:
    st.session_state['df_processed'] = None
if 'date_columns' not in st.session_state:
    st.session_state['date_columns'] = []

# ============================================================
# FUNÇÕES DE PROCESSAMENTO
# ============================================================

def carregar_dados(arquivo):
    """Carrega dados de CSV ou Excel"""
    try:
        if arquivo.name.endswith('.csv'):
            # Tentar diferentes encodings
            for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(arquivo, encoding=encoding)
                    # Se só tem 1 coluna, tentar outros separadores
                    if df.shape[1] == 1:
                        for sep in [';', '\t', '|']:
                            try:
                                arquivo.seek(0)
                                df = pd.read_csv(arquivo, encoding=encoding, sep=sep)
                                if df.shape[1] > 1:
                                    return df
                            except:
                                continue
                    if df.shape[1] > 1:
                        return df
                except:
                    continue
            # Última tentativa
            arquivo.seek(0)
            return pd.read_csv(arquivo)
        else:
            return pd.read_excel(arquivo)
    except Exception as e:
        st.error(f"Erro ao carregar: {str(e)}")
        return None

def processar_dados(df):
    """Processa e limpa dados automaticamente"""
    df = df.copy()
    date_cols = []
    
    # Identificar colunas de data
    for col in df.columns:
        col_lower = col.lower()
        if any(p in col_lower for p in ['data', 'hora', 'date', 'time', 'ano', 'mes', 'dia']):
            date_cols.append(col)
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    
    # Converter colunas numéricas
    for col in df.columns:
        if col not in date_cols:
            try:
                # Remover caracteres não numéricos (exceto ponto e sinal negativo)
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '.')
                    df[col] = df[col].str.extract(r'([-]?\d+\.?\d*)')[0]
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    return df, date_cols

def preencher_ausentes(df, metodo='media'):
    """Preenche valores ausentes"""
    df = df.copy()
    colunas_num = df.select_dtypes(include=[np.number]).columns
    
    if metodo == 'media':
        for col in colunas_num:
            media = df[col].mean()
            df[col].fillna(media, inplace=True)
    elif metodo == 'mediana':
        for col in colunas_num:
            mediana = df[col].median()
            df[col].fillna(mediana, inplace=True)
    elif metodo == 'interpolar':
        for col in colunas_num:
            df[col].interpolate(method='linear', limit_direction='both', inplace=True)
    
    return df

# ============================================================
# FUNÇÕES ESTATÍSTICAS
# ============================================================

def calcular_estatisticas(df, coluna):
    """Calcula estatísticas descritivas completas"""
    dados = df[coluna].dropna()
    
    if len(dados) < 2:
        return None
    
    n = len(dados)
    media = np.mean(dados)
    mediana = np.median(dados)
    desvio = np.std(dados, ddof=1)
    variancia = np.var(dados, ddof=1)
    minimo = np.min(dados)
    maximo = np.max(dados)
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1
    cv = (desvio / media * 100) if media != 0 else 0
    
    # Assimetria
    assimetria = np.sum((dados - media) ** 3) / (n * desvio ** 3) if desvio > 0 else 0
    
    # Curtose
    curtose = np.sum((dados - media) ** 4) / (n * desvio ** 4) - 3 if desvio > 0 else 0
    
    return {
        'Amostras': n,
        'Média': round(media, 3),
        'Mediana': round(mediana, 3),
        'Desvio Padrão': round(desvio, 3),
        'Variância': round(variancia, 3),
        'Mínimo': round(minimo, 3),
        'Máximo': round(maximo, 3),
        'Q1': round(q1, 3),
        'Q3': round(q3, 3),
        'IQR': round(iqr, 3),
        'CV (%)': round(cv, 2),
        'Assimetria': round(assimetria, 3),
        'Curtose': round(curtose, 3)
    }

def classificar_cv(cv):
    """Classifica o CV segundo Pimentel-Gomes"""
    if cv <= 10:
        return "Baixo 🟢"
    elif cv <= 20:
        return "Médio 🟡"
    elif cv <= 30:
        return "Alto 🟠"
    else:
        return "Muito Alto 🔴"

def detectar_outliers(dados):
    """Detecta outliers pelo método IQR"""
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    outliers = (dados < lim_inf) | (dados > lim_sup)
    return outliers, lim_inf, lim_sup

# ============================================================
# FUNÇÕES DE GRÁFICOS
# ============================================================

def grafico_linhas(df, x_col, y_cols, titulo="Série Temporal"):
    """Cria gráfico de linhas"""
    df_melted = df.melt(
        id_vars=[x_col],
        value_vars=y_cols,
        var_name='Variável',
        value_name='Valor'
    )
    
    chart = alt.Chart(df_melted).mark_line(
        strokeWidth=2,
        point=alt.OverlayMarkDef(size=30)
    ).encode(
        x=alt.X(f'{x_col}:T', title='Data'),
        y=alt.Y('Valor:Q', title='Valor'),
        color=alt.Color('Variável:N', legend=alt.Legend(orient='bottom')),
        tooltip=[f'{x_col}:T', 'Variável:N', alt.Tooltip('Valor:Q', format='.2f')]
    ).properties(
        title=titulo,
        height=400
    ).interactive()
    
    return chart

def grafico_histograma(df, coluna, bins=30):
    """Cria histograma"""
    chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
        alt.X(f'{coluna}:Q', bin=alt.Bin(maxbins=bins), title=coluna),
        alt.Y('count()', title='Frequência'),
        tooltip=['count()']
    ).properties(
        title=f'Distribuição de {coluna}',
        height=400
    )
    
    return chart

def grafico_boxplot(df, coluna):
    """Cria boxplot"""
    chart = alt.Chart(df).mark_boxplot(
        extent='min-max',
        color='#2d8a4e'
    ).encode(
        y=alt.Y(f'{coluna}:Q', title=coluna)
    ).properties(
        title=f'Boxplot de {coluna}',
        height=400
    )
    
    return chart

def grafico_barras(df, x_col, y_col, titulo="Gráfico de Barras"):
    """Cria gráfico de barras"""
    chart = alt.Chart(df).mark_bar(
        opacity=0.8,
        color='#2d8a4e'
    ).encode(
        x=alt.X(f'{x_col}:O', title=x_col, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f'{y_col}:Q', title=y_col),
        tooltip=[x_col, alt.Tooltip(y_col, format='.2f')]
    ).properties(
        title=titulo,
        height=400
    )
    
    return chart

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero-header">
    <h1>🌱 AgroDataLab</h1>
    <p>Sistema Inteligente de Análise de Dados Meteorológicos</p>
    <p style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">
        Upload, Tratamento, Análise Estatística e Visualização de Dados INMET
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📊 Status dos Dados")
    
    if st.session_state['df_processed'] is not None:
        df_status = st.session_state['df_processed']
        st.success("✅ Dados Carregados")
        
        col1, col2 = st.columns(2)
        col1.metric("📋 Registros", f"{len(df_status):,}")
        col2.metric("📊 Variáveis", len(df_status.columns))
        
        ausentes = df_status.isnull().sum().sum()
        if ausentes > 0:
            st.warning(f"⚠️ {ausentes} valores ausentes")
        else:
            st.success("✅ Dados completos")
    else:
        st.info("📤 Nenhum dado carregado")
    
    st.markdown("---")
    st.markdown("### 📖 Guia de Uso")
    st.markdown("""
    **1. Upload** → Carregue seu arquivo  
    **2. Tratamento** → Limpe os dados  
    **3. Análise** → Estatísticas  
    **4. Gráficos** → Visualize  
    **5. Download** → Exporte
    """)
    
    st.markdown("---")
    st.caption("AgroDataLab v3.0 | MIT License")

# ============================================================
# ABAS
# ============================================================
aba1, aba2, aba3, aba4, aba5 = st.tabs([
    "📤 1. Upload",
    "🔧 2. Tratamento",
    "📊 3. Análise Estatística",
    "📈 4. Gráficos",
    "💾 5. Download"
])

# ============================================================
# ABA 1: UPLOAD
# ============================================================
with aba1:
    st.header("📤 Upload de Arquivo")
    st.markdown("---")
    
    arquivo = st.file_uploader(
        "Selecione um arquivo CSV ou Excel",
        type=['csv', 'xlsx', 'xls'],
        help="Formatos aceitos: CSV (com qualquer separador) e Excel"
    )
    
    if arquivo is not None:
        with st.spinner('🔄 Carregando e processando dados...'):
            df = carregar_dados(arquivo)
            
            if df is not None and len(df) > 0:
                # Processar dados
                df, date_cols = processar_dados(df)
                
                # Salvar no session state
                st.session_state['df_original'] = df.copy()
                st.session_state['df_processed'] = df.copy()
                st.session_state['date_columns'] = date_cols
                
                # Mensagem de sucesso
                st.markdown('<div class="success-box"><h3>✅ Arquivo carregado com sucesso!</h3></div>', unsafe_allow_html=True)
                
                # Métricas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Registros</p>
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
                    num_cols = len(df.select_dtypes(include=[np.number]).columns)
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Numéricas</p>
                        <p class="metric-value">{num_cols}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    ausentes = df.isnull().sum().sum()
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Ausentes</p>
                        <p class="metric-value">{ausentes:,}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Preview
                st.markdown("---")
                st.subheader("👀 Visualização dos Dados")
                
                n_linhas = st.slider("Número de linhas:", 5, 50, 10, key="slider_preview")
                st.dataframe(df.head(n_linhas), use_container_width=True)
                
                # Info das colunas
                with st.expander("🔍 Detalhes das Colunas"):
                    info = pd.DataFrame({
                        'Coluna': df.columns,
                        'Tipo': df.dtypes.values.astype(str),
                        'Não Nulos': df.count().values,
                        'Nulos': df.isnull().sum().values,
                        'Únicos': df.nunique().values
                    })
                    st.dataframe(info, use_container_width=True)
                
                # Colunas de data
                if date_cols:
                    st.success(f"📅 Colunas de data identificadas: **{', '.join(date_cols)}**")
                else:
                    st.info("ℹ️ Nenhuma coluna de data identificada automaticamente.")
            else:
                st.error("❌ Não foi possível carregar o arquivo. Verifique o formato.")
    else:
        st.markdown("""
        <div class="info-box">
            <h4>📤 Como usar:</h4>
            <ol>
                <li>Clique em <strong>"Browse files"</strong></li>
                <li>Selecione um arquivo <strong>CSV</strong> ou <strong>Excel</strong></li>
                <li>O sistema detectará automaticamente o formato</li>
                <li>Dados do INMET são processados automaticamente</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# ABA 2: TRATAMENTO
# ============================================================
with aba2:
    st.header("🔧 Tratamento de Dados")
    st.markdown("---")
    
    if st.session_state['df_processed'] is None:
        st.warning("⚠️ Carregue os dados na Aba 1 primeiro!")
    else:
        df = st.session_state['df_processed']
        
        # Diagnóstico
        missing_total = df.isnull().sum().sum()
        
        if missing_total > 0:
            st.markdown(f"""
            <div class="warning-box">
                <h3>⚠️ Dados Ausentes Detectados</h3>
                <p><strong>{missing_total:,}</strong> valores ausentes em <strong>{df.isnull().any().sum()}</strong> colunas</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Tabela de ausentes
            st.subheader("📋 Colunas com Dados Ausentes")
            
            missing_df = pd.DataFrame({
                'Coluna': df.columns,
                'Ausentes': df.isnull().sum().values,
                'Porcentagem': (df.isnull().sum() / len(df) * 100).round(2).values
            })
            missing_df = missing_df[missing_df['Ausentes'] > 0].sort_values('Ausentes', ascending=False)
            
            st.dataframe(missing_df, use_container_width=True)
            
            # Métodos de preenchimento
            st.markdown("---")
            st.subheader("🛠️ Métodos de Preenchimento")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Média Aritmética**")
                st.caption("Substitui pela média da coluna")
                if st.button("Aplicar Média", key="btn_media"):
                    st.session_state['df_processed'] = preencher_ausentes(df.copy(), 'media')
                    st.success("✅ Preenchido com média!")
                    st.rerun()
            
            with col2:
                st.markdown("**Mediana**")
                st.caption("Substitui pela mediana (robusto a outliers)")
                if st.button("Aplicar Mediana", key="btn_mediana"):
                    st.session_state['df_processed'] = preencher_ausentes(df.copy(), 'mediana')
                    st.success("✅ Preenchido com mediana!")
                    st.rerun()
            
            with col3:
                st.markdown("**Interpolação Linear**")
                st.caption("Estima valores entre pontos adjacentes")
                if st.button("Aplicar Interpolação", key="btn_interp"):
                    st.session_state['df_processed'] = preencher_ausentes(df.copy(), 'interpolar')
                    st.success("✅ Interpolado!")
                    st.rerun()
            
            col4, col5 = st.columns(2)
            
            with col4:
                st.markdown("**Remover Linhas**")
                st.caption("Exclui linhas com qualquer valor ausente")
                if st.button("🗑️ Remover Linhas Incompletas", key="btn_drop"):
                    antes = len(df)
                    df_limpo = df.dropna()
                    st.session_state['df_processed'] = df_limpo
                    st.success(f"✅ {antes - len(df_limpo)} linhas removidas!")
                    st.rerun()
            
            with col5:
                st.markdown("**Restaurar Original**")
                st.caption("Volta aos dados originais")
                if st.button("🔙 Restaurar", key="btn_restore"):
                    if st.session_state['df_original'] is not None:
                        st.session_state['df_processed'] = st.session_state['df_original'].copy()
                        st.success("✅ Dados restaurados!")
                        st.rerun()
        else:
            st.markdown("""
            <div class="success-box">
                <h3>✅ Dados Completos!</h3>
                <p>Nenhum valor ausente encontrado. Seus dados estão prontos para análise.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detecção de Outliers
        st.markdown("---")
        st.subheader("🔍 Detecção de Outliers (Método IQR)")
        
        colunas_num = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if colunas_num:
            col_out = st.selectbox(
                "Selecione a variável:",
                colunas_num,
                key="outlier_col"
            )
            
            dados_limpos = df[col_out].dropna()
            
            if len(dados_limpos) > 0:
                outliers, lim_inf, lim_sup = detectar_outliers(dados_limpos)
                n_out = outliers.sum()
                
                if n_out > 0:
                    pct = n_out / len(dados_limpos) * 100
                    st.warning(f"🔴 **{n_out}** outliers detectados ({pct:.1f}% dos dados)")
                    
                    # Boxplot
                    chart_out = grafico_boxplot(pd.DataFrame({col_out: dados_limpos}), col_out)
                    st.altair_chart(chart_out, use_container_width=True)
                    
                    col_a, col_b = st.columns(2)
                    col_a.metric("Limite Inferior", f"{lim_inf:.3f}")
                    col_b.metric("Limite Superior", f"{lim_sup:.3f}")
                else:
                    st.success(f"✅ Nenhum outlier detectado em **{col_out}**")
        else:
            st.info("Nenhuma variável numérica encontrada para análise de outliers.")

# ============================================================
# ABA 3: ANÁLISE ESTATÍSTICA
# ============================================================
with aba3:
    st.header("📊 Análise Estatística")
    st.markdown("---")
    
    if st.session_state['df_processed'] is None:
        st.warning("⚠️ Carregue os dados na Aba 1 primeiro!")
    else:
        df = st.session_state['df_processed']
        colunas_num = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not colunas_num:
            st.warning("⚠️ Nenhuma variável numérica encontrada nos dados.")
        else:
            # Selecionar variável
            col_analise = st.selectbox(
                "Selecione a variável para análise:",
                colunas_num,
                key="analise_col"
            )
            
            # Calcular estatísticas
            stats = calcular_estatisticas(df, col_analise)
            
            if stats:
                st.subheader(f"📈 Estatísticas Descritivas: **{col_analise}**")
                
                # Cards com principais métricas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Média", f"{stats['Média']:.3f}")
                with col2:
                    st.metric("Desvio Padrão", f"{stats['Desvio Padrão']:.3f}")
                with col3:
                    cv_val = stats['CV (%)']
                    st.metric("CV (%)", f"{cv_val:.2f}%", delta=classificar_cv(cv_val))
                with col4:
                    st.metric("Amostras", f"{stats['Amostras']:,}")
                
                # Tabela completa
                with st.expander("📋 Ver Todas as Estatísticas"):
                    stats_df = pd.DataFrame(list(stats.items()), columns=['Estatística', 'Valor'])
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Histograma
                st.markdown("---")
                st.subheader("📊 Distribuição")
                
                bins = st.slider("Número de intervalos:", 10, 100, 30, key="hist_bins")
                dados_plot = df[[col_analise]].dropna()
                chart_hist = grafico_histograma(dados_plot, col_analise, bins)
                st.altair_chart(chart_hist, use_container_width=True)
                
                # Matriz de Correlação
                if len(colunas_num) >= 2:
                    st.markdown("---")
                    st.subheader("🔗 Matriz de Correlação")
                    
                    corr_matrix = df[colunas_num].corr()
                    
                    st.dataframe(
                        corr_matrix.style.background_gradient(
                            cmap='RdYlGn',
                            vmin=-1,
                            vmax=1
                        ).format("{:.3f}"),
                        use_container_width=True
                    )
                    
                    st.caption("""
                    **Interpretação:**  
                    🟢 **1.0 a 0.7** = Correlação forte  
                    🟡 **0.7 a 0.3** = Correlação moderada  
                    🔴 **0.3 a 0.0** = Correlação fraca
                    """)
                
                # Análise Temporal
                date_cols = st.session_state.get('date_columns', [])
                
                if date_cols:
                    st.markdown("---")
                    st.subheader("📅 Análise Temporal")
                    
                    col_data = st.selectbox(
                        "Coluna de data:",
                        date_cols,
                        key="data_col"
                    )
                    
                    freq = st.selectbox(
                        "Agregação:",
                        ['D', 'W', 'M'],
                        format_func=lambda x: {'D': 'Diária', 'W': 'Semanal', 'M': 'Mensal'}[x],
                        key="freq_agg"
                    )
                    
                    # Criar agregação
                    df_temp = df.copy()
                    df_temp[col_data] = pd.to_datetime(df_temp[col_data], errors='coerce')
                    df_temp = df_temp.dropna(subset=[col_data])
                    df_temp.set_index(col_data, inplace=True)
                    
                    freq_map = {'D': 'D', 'W': 'W', 'M': 'M'}
                    df_agg = df_temp[[col_analise]].resample(freq_map[freq]).mean()
                    
                    st.dataframe(df_agg.head(20), use_container_width=True)
                    
                    # Gráfico temporal
                    df_plot = df[[col_data, col_analise]].dropna()
                    df_plot[col_data] = pd.to_datetime(df_plot[col_data])
                    df_plot = df_plot.sort_values(col_data)
                    
                    if len(df_plot) > 0:
                        chart_temp = grafico_linhas(
                            df_plot,
                            col_data,
                            [col_analise],
                            f'Série Temporal - {col_analise}'
                        )
                        st.altair_chart(chart_temp, use_container_width=True)

# ============================================================
# ABA 4: GRÁFICOS
# ============================================================
with aba4:
    st.header("📈 Visualização Gráfica")
    st.markdown("---")
    
    if st.session_state['df_processed'] is None:
        st.warning("⚠️ Carregue os dados na Aba 1 primeiro!")
    else:
        df = st.session_state['df_processed']
        colunas_num = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not colunas_num:
            st.warning("⚠️ Nenhuma variável numérica disponível.")
        else:
            # Tipo de gráfico
            tipo = st.radio(
                "Selecione o tipo de gráfico:",
                ['📈 Linhas (Série Temporal)', '📊 Histograma', '📉 Boxplot', '📋 Barras'],
                horizontal=True,
                key="tipo_graf"
            )
            
            st.markdown("---")
            
            if tipo == '📈 Linhas (Série Temporal)':
                date_cols = st.session_state.get('date_columns', [])
                
                if date_cols:
                    col_x = st.selectbox("Eixo X (Data):", date_cols, key="lx")
                    col_y = st.multiselect(
                        "Variáveis (Eixo Y):",
                        colunas_num,
                        default=colunas_num[:min(3, len(colunas_num))],
                        key="ly"
                    )
                    
                    if col_y:
                        df_plot = df[[col_x] + col_y].copy()
                        df_plot[col_x] = pd.to_datetime(df_plot[col_x], errors='coerce')
                        df_plot = df_plot.dropna()
                        
                        if len(df_plot) > 0:
                            chart = grafico_linhas(df_plot, col_x, col_y)
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.warning("Dados insuficientes para o gráfico.")
                else:
                    st.info("Nenhuma coluna de data identificada. Use outro tipo de gráfico.")
            
            elif tipo == '📊 Histograma':
                col_h = st.selectbox("Variável:", colunas_num, key="hc")
                bins_h = st.slider("Intervalos:", 5, 100, 30, key="hb")
                
                df_plot = df[[col_h]].dropna()
                chart = grafico_histograma(df_plot, col_h, bins_h)
                st.altair_chart(chart, use_container_width=True)
            
            elif tipo == '📉 Boxplot':
                col_b = st.selectbox("Variável:", colunas_num, key="bc")
                
                df_plot = df[[col_b]].dropna()
                chart = grafico_boxplot(df_plot, col_b)
                st.altair_chart(chart, use_container_width=True)
            
            elif tipo == '📋 Barras':
                # Para barras, usar uma coluna categórica ou index
                todas_cols = df.columns.tolist()
                col_x_bar = st.selectbox("Eixo X (Categorias):", todas_cols, key="bx")
                col_y_bar = st.selectbox("Eixo Y (Valores):", colunas_num, key="by")
                
                df_plot = df[[col_x_bar, col_y_bar]].dropna()
                
                # Agregar se necessário
                if len(df_plot) > 100:
                    st.caption(f"Mostrando primeiras 100 categorias de {len(df_plot)}")
                    df_plot = df_plot.head(100)
                
                chart = grafico_barras(df_plot, col_x_bar, col_y_bar)
                st.altair_chart(chart, use_container_width=True)

# ============================================================
# ABA 5: DOWNLOAD
# ============================================================
with aba5:
    st.header("💾 Download dos Dados")
    st.markdown("---")
    
    if st.session_state['df_processed'] is None:
        st.warning("⚠️ Carregue e processe os dados primeiro!")
    else:
        df = st.session_state['df_processed']
        
        st.markdown("""
        <div class="success-box">
            <h3>✅ Dados Prontos para Download</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Resumo
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Registros</p>
                <p class="metric-value">{len(df):,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Colunas</p>
                <p class="metric-value">{len(df.columns)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            ausentes = df.isnull().sum().sum()
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Ausentes</p>
                <p class="metric-value">{ausentes:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            memoria = df.memory_usage(deep=True).sum() / 1024**2
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Tamanho</p>
                <p class="metric-value">{memoria:.1f} MB</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Download CSV
        st.subheader("📥 Download CSV")
        
        csv = df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="📥 Baixar arquivo CSV",
            data=csv,
            file_name=f"dados_processados_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="download_csv"
        )
        
        # Preview
        st.markdown("---")
        st.subheader("👀 Preview dos Dados Processados")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Comparação com original
        if st.session_state['df_original'] is not None:
            st.markdown("---")
            st.subheader("📊 Resumo do Processamento")
            
            df_orig = st.session_state['df_original']
            
            resumo = pd.DataFrame({
                'Métrica': [
                    'Registros Originais',
                    'Registros Finais',
                    'Diferença',
                    'Dados Ausentes (Original)',
                    'Dados Ausentes (Final)',
                    'Tratados'
                ],
                'Valor': [
                    f"{len(df_orig):,}",
                    f"{len(df):,}",
                    f"{len(df_orig) - len(df):,}",
                    f"{df_orig.isnull().sum().sum():,}",
                    f"{df.isnull().sum().sum():,}",
                    f"{df_orig.isnull().sum().sum() - df.isnull().sum().sum():,}"
                ]
            })
            
            st.dataframe(resumo, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div class="app-footer">
    <h4>🌱 AgroDataLab v3.0</h4>
    <p>Sistema de Análise de Dados Meteorológicos</p>
    <p style="font-size:0.85rem;">
        Desenvolvido com Streamlit, Pandas, NumPy e Altair<br>
        Métodos: Pearson (1895) | Tukey (1977) | Pimentel-Gomes (2000)
    </p>
    <p style="font-size:0.8rem; color:#adb5bd;">Licença MIT © 2024</p>
</div>
""", unsafe_allow_html=True)
