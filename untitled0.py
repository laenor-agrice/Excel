# ============================================================
# AgroDataLab - Sistema Completo de Análise Meteorológica
# Versão 9.1 - Correção de Importação SciPy
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
import io
import csv
import traceback
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# IMPORTAÇÃO SEGURA DO SCIPY
# ============================================================
try:
    from scipy import stats as scipy_stats
    SCIPY_DISPONIVEL = True
except ImportError:
    SCIPY_DISPONIVEL = False
    scipy_stats = None

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
# CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    
    .hero-header {
        background: linear-gradient(135deg, #0d2818 0%, #1a5632 30%, #2d8a4e 70%, #4caf50 100%);
        padding: 2rem 2.5rem; border-radius: 20px; margin-bottom: 2rem;
        color: white; box-shadow: 0 20px 60px rgba(26,86,50,0.3);
    }
    .hero-header h1 { font-size: 2.5rem; font-weight: 800; margin: 0; }
    .hero-header p { font-size: 1.1rem; opacity: 0.95; margin: 0.5rem 0 0 0; }
    
    .metric-card {
        background: white; border-radius: 16px; padding: 1.2rem;
        border-left: 4px solid #2d8a4e; box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        margin-bottom: 0.5rem;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1a5632; margin: 0; }
    .metric-label { font-size: 0.8rem; color: #6c757d; font-weight: 500; text-transform: uppercase; }
    
    .stButton > button {
        background: linear-gradient(135deg, #1a5632 0%, #2d8a4e 100%);
        color: white; border: none; padding: 0.6rem 1.2rem;
        border-radius: 10px; font-weight: 600; width: 100%;
        transition: all 0.3s ease; cursor: pointer;
    }
    .stButton > button:hover { transform: translateY(-2px); }
    
    .success-box {
        background: #e8f5e9; border: 1px solid #81c784;
        border-radius: 12px; padding: 1.5rem; border-left: 5px solid #2e7d32; margin: 1rem 0;
    }
    .warning-box {
        background: #fff3e0; border: 1px solid #ffb74d;
        border-radius: 12px; padding: 1.5rem; border-left: 5px solid #e65100; margin: 1rem 0;
    }
    .info-box {
        background: #e3f2fd; border: 1px solid #64b5f6;
        border-radius: 12px; padding: 1.5rem; border-left: 5px solid #1565c0; margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem; background: #f8f9fa; padding: 0.5rem; border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.2rem; background: white; border-radius: 8px; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a5632, #2d8a4e); color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None
if 'df_processed' not in st.session_state:
    st.session_state['df_processed'] = None
if 'df_mensal' not in st.session_state:
    st.session_state['df_mensal'] = None
if 'date_columns' not in st.session_state:
    st.session_state['date_columns'] = []

# ============================================================
# FUNÇÕES BÁSICAS
# ============================================================
def corrigir_nomes_duplicados(df):
    """Renomeia colunas duplicadas"""
    cols = pd.Series(df.columns)
    for nome in cols[cols.duplicated()].unique():
        idx = cols[cols == nome].index.tolist()
        for i, pos in enumerate(idx):
            if i > 0:
                cols[pos] = f"{nome}_{i}"
    df.columns = cols
    return df

def detectar_delimitador(arquivo):
    """Detecta delimitador do CSV"""
    try:
        arquivo.seek(0)
        amostra_bytes = arquivo.read(8192)
        arquivo.seek(0)
        for enc in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
            try:
                amostra = amostra_bytes.decode(enc)
                break
            except:
                continue
        else:
            amostra = amostra_bytes.decode('utf-8', errors='ignore')
        try:
            sniffer = csv.Sniffer()
            dialecto = sniffer.sniff(amostra)
            delim = dialecto.delimiter
            linhas = amostra.strip().split('\n')
            if len(linhas) >= 2:
                if linhas[0].count(delim) == linhas[1].count(delim) > 0:
                    return delim
        except:
            pass
        for sep in [';', ',', '\t', '|']:
            linhas = amostra.strip().split('\n')
            if len(linhas) >= 2:
                if linhas[0].count(sep) == linhas[1].count(sep) > 0:
                    return sep
        return ';' if amostra.count(';') > amostra.count(',') else ','
    except:
        return ','

def carregar_dados(arquivo):
    """Carrega CSV ou Excel"""
    try:
        if arquivo.name.endswith('.csv'):
            sep = detectar_delimitador(arquivo)
            for enc, dec in [('utf-8',','),('utf-8','.'),('latin1',','),('latin1','.'),
                            ('iso-8859-1',','),('iso-8859-1','.'),('cp1252',','),('cp1252','.')]:
                try:
                    arquivo.seek(0)
                    df = pd.read_csv(arquivo, sep=sep, encoding=enc, decimal=dec,
                                   thousands='.', on_bad_lines='skip', engine='python')
                    if df.shape[1] > 1:
                        return corrigir_nomes_duplicados(df)
                except:
                    continue
            arquivo.seek(0)
            df = pd.read_csv(arquivo, sep=sep, encoding='utf-8', on_bad_lines='skip', engine='python')
            return corrigir_nomes_duplicados(df)
        else:
            return corrigir_nomes_duplicados(pd.read_excel(arquivo))
    except Exception as e:
        st.error(f"Erro: {str(e)}")
        return None

def processar_dados(df):
    """Processa dados - apenas colunas com nome de data"""
    df = df.copy()
    date_cols = []
    palavras_data = ['data', 'hora', 'date', 'time', 'timestamp', 'datetime', 'datahora']
    
    for col in df.columns:
        if any(p in str(col).lower().replace(" ","_") for p in palavras_data):
            try:
                df[col] = pd.to_datetime(df[col], format='mixed', dayfirst=True, errors='coerce')
                if df[col].notna().sum() > 0:
                    date_cols.append(col)
            except:
                pass
    
    for col in df.columns:
        if col in date_cols:
            continue
        if df[col].dtype == "object":
            amostra = df[col].dropna().astype(str).str.replace(",", ".").head(20)
            try:
                pd.to_numeric(amostra)
                df[col] = df[col].astype(str).str.replace(",", ".")
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except:
                pass
        elif df[col].dtype in ['int64', 'float64']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df, date_cols

def preencher_ausentes(df, metodo='media'):
    df = df.copy()
    cols_num = df.select_dtypes(include=[np.number]).columns
    if metodo == 'media':
        df[cols_num] = df[cols_num].fillna(df[cols_num].mean())
    elif metodo == 'mediana':
        df[cols_num] = df[cols_num].fillna(df[cols_num].median())
    elif metodo == 'interpolar':
        df[cols_num] = df[cols_num].interpolate(method='linear', limit_direction='both')
    return df

def obter_df_limpo():
    if st.session_state['df_processed'] is None:
        return None
    df = st.session_state['df_processed'].copy()
    df = corrigir_nomes_duplicados(df)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# ============================================================
# AGREGAÇÃO TEMPORAL
# ============================================================
def criar_agregacao_temporal(df, date_col, freq='M'):
    """Cria agregação temporal (média mensal ou anual)"""
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    df_temp = df_temp.dropna(subset=[date_col])
    df_temp = df_temp.set_index(date_col)
    
    colunas_num = df_temp.select_dtypes(include=[np.number]).columns
    
    if freq == 'M':
        df_agg = df_temp[colunas_num].resample('M').mean().round(2)
        df_agg.index = df_agg.index.strftime('%b/%Y')
    elif freq == 'Y':
        df_agg = df_temp[colunas_num].resample('Y').mean().round(2)
        df_agg.index = df_agg.index.strftime('%Y')
    
    return df_agg.reset_index()

# ============================================================
# ESTATÍSTICAS COMPLETAS (COM FALLBACK SEM SCIPY)
# ============================================================
def calcular_estatisticas_completas(df, coluna):
    """
    Calcula estatísticas descritivas completas.
    Se SciPy disponível: inclui Shapiro-Wilk
    Se SciPy indisponível: usa apenas pandas/numpy
    """
    dados = df[coluna].dropna()
    
    if len(dados) < 3:
        st.warning(f"**{coluna}**: poucos dados ({len(dados)} valores)")
        return None
    
    n = len(dados)
    faltantes = df[coluna].isnull().sum()
    soma = np.sum(dados)
    media = np.mean(dados)
    mediana = np.median(dados)
    
    # Moda
    moda_vals = dados.mode()
    moda = moda_vals.iloc[0] if len(moda_vals) > 0 else np.nan
    
    minimo = np.min(dados)
    maximo = np.max(dados)
    amplitude = maximo - minimo
    variancia = np.var(dados, ddof=1)
    desvio = np.std(dados, ddof=1)
    erro_padrao = desvio / np.sqrt(n)
    cv = (desvio / media * 100) if media != 0 else 0
    
    q1 = np.percentile(dados, 25)
    q2 = np.percentile(dados, 50)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1
    p10 = np.percentile(dados, 10)
    p90 = np.percentile(dados, 90)
    
    # Assimetria e Curtose (pandas nativo)
    assimetria = dados.skew()
    curtose = dados.kurtosis()
    
    # Classificação CV
    if cv <= 10:
        cv_class = "Baixo 🟢"
    elif cv <= 20:
        cv_class = "Médio 🟡"
    elif cv <= 30:
        cv_class = "Alto 🟠"
    else:
        cv_class = "Muito Alto 🔴"
    
    resultado = {
        'N': n,
        'Faltantes': faltantes,
        'Soma': round(soma, 2),
        'Média': round(media, 4),
        'Mediana': round(mediana, 4),
        'Moda': round(moda, 4) if not pd.isna(moda) else 'N/A',
        'Mínimo': round(minimo, 4),
        'Máximo': round(maximo, 4),
        'Amplitude': round(amplitude, 4),
        'Variância': round(variancia, 4),
        'Desvio Padrão': round(desvio, 4),
        'Erro Padrão': round(erro_padrao, 4),
        'CV (%)': round(cv, 2),
        'Classificação CV': cv_class,
        'Q1 (25%)': round(q1, 4),
        'Q2 (50%)': round(q2, 4),
        'Q3 (75%)': round(q3, 4),
        'IQR': round(iqr, 4),
        'P10': round(p10, 4),
        'P90': round(p90, 4),
        'Assimetria': round(assimetria, 4),
        'Curtose': round(curtose, 4),
    }
    
    # Teste de normalidade (apenas se SciPy disponível)
    if SCIPY_DISPONIVEL and 3 <= n <= 5000:
        try:
            stat_sw, p_sw = scipy_stats.shapiro(dados)
            resultado['Shapiro-Wilk W'] = round(stat_sw, 4)
            resultado['Shapiro-Wilk p'] = round(p_sw, 4)
            resultado['Dist. Normal?'] = "Sim ✅" if p_sw > 0.05 else "Não ❌"
        except:
            resultado['Shapiro-Wilk W'] = 'Erro'
            resultado['Shapiro-Wilk p'] = 'Erro'
            resultado['Dist. Normal?'] = 'N/A'
    elif SCIPY_DISPONIVEL:
        resultado['Shapiro-Wilk W'] = 'N/A (n>5000)'
        resultado['Shapiro-Wilk p'] = 'N/A (n>5000)'
        resultado['Dist. Normal?'] = 'N/A (n>5000)'
    else:
        resultado['Shapiro-Wilk W'] = 'SciPy não instalado'
        resultado['Shapiro-Wilk p'] = 'SciPy não instalado'
        resultado['Dist. Normal?'] = 'Indisponível'
    
    return resultado

# ============================================================
# FUNÇÕES DE GRÁFICOS
# ============================================================
def criar_grafico_linhas(df, x_col, y_cols, titulo="Série Temporal"):
    try:
        df = corrigir_nomes_duplicados(df)
        df = df.loc[:, ~df.columns.duplicated()]
        y_cols_validas = [c for c in y_cols if c in df.columns]
        if not y_cols_validas:
            return None
        df_melted = pd.melt(df, id_vars=[x_col], value_vars=y_cols_validas,
                           var_name='Variável', value_name='Valor')
        chart = alt.Chart(df_melted).mark_line(strokeWidth=2).encode(
            x=alt.X(f'{x_col}:T', title='Data'),
            y=alt.Y('Valor:Q', title='Valor'),
            color=alt.Color('Variável:N', legend=alt.Legend(orient='bottom')),
            tooltip=[f'{x_col}:T', 'Variável:N', alt.Tooltip('Valor:Q', format='.2f')]
        ).properties(title=titulo, height=400).interactive()
        return chart
    except Exception as e:
        st.error(f"Erro: {str(e)}")
        return None

def criar_histograma_aprimorado(df, coluna, bins=30):
    """Histograma com média e mediana"""
    try:
        df = corrigir_nomes_duplicados(df)
        if coluna not in df.columns:
            return None
        df_clean = df[[coluna]].dropna()
        if len(df_clean) < 2:
            return None
        
        media_val = df_clean[coluna].mean()
        mediana_val = df_clean[coluna].median()
        
        bars = alt.Chart(df_clean).mark_bar(opacity=0.7, color='#2d8a4e').encode(
            alt.X(f'{coluna}:Q', bin=alt.Bin(maxbins=bins), title=coluna),
            alt.Y('count()', title='Frequência')
        )
        
        rule_media = alt.Chart(pd.DataFrame({'x': [media_val]})).mark_rule(
            color='red', strokeWidth=2, strokeDash=[5,5]
        ).encode(x='x:Q')
        
        rule_mediana = alt.Chart(pd.DataFrame({'x': [mediana_val]})).mark_rule(
            color='blue', strokeWidth=2, strokeDash=[3,3]
        ).encode(x='x:Q')
        
        chart = (bars + rule_media + rule_mediana).properties(
            title=f'Distribuição de {coluna} (Média: {media_val:.2f} | Mediana: {mediana_val:.2f})',
            height=400
        )
        
        return chart
    except Exception as e:
        st.error(f"Erro: {str(e)}")
        return None

def criar_grafico_barras(df, x_col, y_col, titulo="Gráfico de Barras"):
    """Gráfico de barras para dados categóricos ou agregados"""
    try:
        df = corrigir_nomes_duplicados(df)
        if x_col not in df.columns or y_col not in df.columns:
            return None
        
        df_clean = df[[x_col, y_col]].dropna()
        if len(df_clean) < 1:
            return None
        
        if pd.api.types.is_datetime64_any_dtype(df_clean[x_col]):
            df_clean[x_col] = df_clean[x_col].dt.strftime('%b/%Y')
        
        chart = alt.Chart(df_clean).mark_bar(
            opacity=0.85, color='#2d8a4e',
            cornerRadiusTopLeft=4, cornerRadiusTopRight=4
        ).encode(
            x=alt.X(f'{x_col}:O', title=x_col, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(f'{y_col}:Q', title=y_col),
            tooltip=[x_col, alt.Tooltip(y_col, format='.2f')]
        ).properties(title=titulo, height=400)
        
        return chart
    except Exception as e:
        st.error(f"Erro: {str(e)}")
        return None

def criar_boxplot(df, coluna):
    try:
        df = corrigir_nomes_duplicados(df)
        if coluna not in df.columns:
            return None
        df_clean = df[[coluna]].dropna()
        if len(df_clean) < 5:
            return None
        chart = alt.Chart(df_clean).mark_boxplot(extent='min-max', color='#2d8a4e').encode(
            y=alt.Y(f'{coluna}:Q', title=coluna)
        ).properties(title=f'Boxplot de {coluna}', height=400)
        return chart
    except Exception as e:
        st.error(f"Erro: {str(e)}")
        return None

def detectar_outliers(dados):
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    outliers = (dados < lim_inf) | (dados > lim_sup)
    return outliers, lim_inf, lim_sup

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero-header">
    <h1>🌱 AgroDataLab</h1>
    <p>Sistema Inteligente de Análise Meteorológica e Agronômica</p>
    <p style="font-size:0.9rem; opacity:0.8;">v9.1 - Estatísticas Avançadas</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📊 Status")
    
    # Status SciPy
    if SCIPY_DISPONIVEL:
        st.success("✅ SciPy disponível")
    else:
        st.warning("⚠️ SciPy não instalado (testes de normalidade indisponíveis)")
    
    df_status = obter_df_limpo()
    if df_status is not None:
        st.success(f"✅ {len(df_status):,} registros")
        num_cols = len(df_status.select_dtypes(include=[np.number]).columns)
        date_cols = len(st.session_state.get('date_columns', []))
        st.caption(f"📊 {num_cols} numéricas | 📅 {date_cols} datas")
        if st.session_state.get('df_mensal') is not None:
            st.info("📆 Agregação temporal disponível")
    else:
        st.info("📤 Sem dados")
    st.markdown("---")
    st.caption("AgroDataLab v9.1 | MIT License")

# ============================================================
# ABAS
# ============================================================
aba1, aba2, aba3, aba4, aba5 = st.tabs([
    "📤 1. Upload",
    "🔧 2. Tratamento",
    "📊 3. Estatísticas",
    "📈 4. Gráficos",
    "💾 5. Download"
])

# ============================================================
# ABA 1: UPLOAD
# ============================================================
with aba1:
    st.header("📤 Upload de Arquivo")
    st.markdown("---")
    arquivo = st.file_uploader("Selecione CSV ou Excel", type=['csv', 'xlsx', 'xls'])
    
    if arquivo is not None:
        with st.spinner('Carregando...'):
            df = carregar_dados(arquivo)
            if df is not None and len(df) > 0:
                if df.shape[1] == 1:
                    st.error("⚠️ Arquivo lido como UMA ÚNICA coluna!")
                    st.stop()
                df, date_cols = processar_dados(df)
                df = corrigir_nomes_duplicados(df)
                df = df.loc[:, ~df.columns.duplicated()]
                st.session_state['df_original'] = df.copy()
                st.session_state['df_processed'] = df.copy()
                st.session_state['date_columns'] = date_cols
                st.session_state['df_mensal'] = None
                
                st.markdown(f'<div class="success-box"><h3>✅ {df.shape[0]:,} linhas × {df.shape[1]} colunas</h3></div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Registros", f"{len(df):,}")
                col2.metric("Colunas", len(df.columns))
                col3.metric("Numéricas", len(df.select_dtypes(include=[np.number]).columns))
                col4.metric("Datas", len(date_cols))
                
                st.markdown("---")
                st.subheader("Preview")
                st.dataframe(df.head(15), use_container_width=True)

# ============================================================
# ABA 2: TRATAMENTO
# ============================================================
with aba2:
    st.header("🔧 Tratamento de Dados")
    st.markdown("---")
    df = obter_df_limpo()
    if df is None:
        st.warning("⚠️ Carregue os dados na Aba 1 primeiro!")
    else:
        # Valores ausentes
        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            st.markdown(f'<div class="warning-box"><h3>⚠️ {missing_total:,} valores ausentes</h3></div>', unsafe_allow_html=True)
            missing_df = pd.DataFrame({
                'Coluna': df.columns,
                'Ausentes': df.isnull().sum().values,
                '%': (df.isnull().sum() / len(df) * 100).round(2).values
            })
            missing_df = missing_df[missing_df['Ausentes'] > 0].sort_values('Ausentes', ascending=False)
            st.dataframe(missing_df, use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("📊 Média", key="m1"):
                    st.session_state['df_processed'] = preencher_ausentes(df, 'media')
                    st.success("✅ Preenchido!")
                    st.rerun()
            with c2:
                if st.button("📈 Mediana", key="m2"):
                    st.session_state['df_processed'] = preencher_ausentes(df, 'mediana')
                    st.success("✅ Preenchido!")
                    st.rerun()
            with c3:
                if st.button("🔄 Interpolar", key="m3"):
                    st.session_state['df_processed'] = preencher_ausentes(df, 'interpolar')
                    st.success("✅ Preenchido!")
                    st.rerun()
            c4, c5 = st.columns(2)
            with c4:
                if st.button("🗑️ Remover Linhas", key="m4"):
                    antes = len(df)
                    st.session_state['df_processed'] = df.dropna()
                    st.success(f"✅ {antes - len(st.session_state['df_processed'])} removidas!")
                    st.rerun()
            with c5:
                if st.button("🔙 Restaurar", key="m5"):
                    if st.session_state['df_original'] is not None:
                        st.session_state['df_processed'] = st.session_state['df_original'].copy()
                        st.success("✅ Restaurado!")
                        st.rerun()
        else:
            st.markdown('<div class="success-box"><h3>✅ Nenhum valor ausente!</h3></div>', unsafe_allow_html=True)
        
        # Agregação temporal
        st.markdown("---")
        st.subheader("📆 Agregação Temporal")
        
        date_cols = st.session_state.get('date_columns', [])
        colunas_num = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if date_cols and colunas_num:
            col_data_agg = st.selectbox("Coluna de data:", date_cols, key="agg_date")
            freq_agg = st.radio("Período:", ['Mensal', 'Anual'], horizontal=True, key="agg_freq")
            
            if st.button("📊 Gerar Média " + freq_agg, key="btn_agg"):
                freq_code = 'M' if freq_agg == 'Mensal' else 'Y'
                df_agg = criar_agregacao_temporal(df, col_data_agg, freq_code)
                st.session_state['df_mensal'] = df_agg
                st.success(f"✅ Média {freq_agg.lower()} gerada! ({len(df_agg)} registros)")
                st.rerun()
            
            if st.session_state.get('df_mensal') is not None:
                st.markdown(f"#### 📋 Média {freq_agg}")
                st.dataframe(st.session_state['df_mensal'], use_container_width=True)
        else:
            st.info("📅 Necessário coluna de data e variáveis numéricas.")
        
        # Outliers
        st.markdown("---")
        st.subheader("🔍 Outliers (IQR)")
        
        if colunas_num:
            col_out = st.selectbox("Variável:", colunas_num, key="out_col")
            dados = df[col_out].dropna()
            if len(dados) >= 2:
                outliers, li, ls = detectar_outliers(dados)
                n_out = outliers.sum()
                if n_out > 0:
                    st.warning(f"🔴 {n_out} outliers ({n_out/len(dados)*100:.1f}%)")
                    df_plot = pd.DataFrame({col_out: dados.values})
                    chart = criar_boxplot(df_plot, col_out)
                    if chart:
                        st.altair_chart(chart, use_container_width=True)
                    col_a, col_b = st.columns(2)
                    col_a.metric("Lim. Inf.", f"{li:.3f}")
                    col_b.metric("Lim. Sup.", f"{ls:.3f}")
                else:
                    st.success(f"✅ Nenhum outlier em {col_out}")

# ============================================================
# ABA 3: ESTATÍSTICAS
# ============================================================
with aba3:
    st.header("📊 Análise Estatística Completa")
    st.markdown("---")
    
    # Aviso SciPy
    if not SCIPY_DISPONIVEL:
        st.warning("⚠️ SciPy não instalado. Teste de normalidade Shapiro-Wilk indisponível. As demais estatísticas funcionam normalmente.")
    
    fonte_dados = st.radio(
        "Fonte de dados:",
        ['Dados Originais', 'Média Mensal/Anual (se disponível)'],
        horizontal=True,
        key="fonte_stats"
    )
    
    if fonte_dados == 'Média Mensal/Anual (se disponível)' and st.session_state.get('df_mensal') is not None:
        df = st.session_state['df_mensal'].copy()
        st.info("📆 Usando dados agregados temporalmente")
    else:
        df = obter_df_limpo()
    
    if df is None:
        st.warning("⚠️ Carregue os dados na Aba 1 primeiro!")
    else:
        colunas_num = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not colunas_num:
            st.warning("⚠️ Nenhuma variável numérica.")
        else:
            col_analise = st.selectbox("Variável para análise:", colunas_num, key="stat_col")
            
            stats = calcular_estatisticas_completas(df, col_analise)
            
            if stats:
                st.subheader(f"📈 Estatísticas Completas: **{col_analise}**")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Média", f"{stats['Média']:.3f}")
                c2.metric("Desvio Padrão", f"{stats['Desvio Padrão']:.3f}")
                c3.metric("CV (%)", f"{stats['CV (%)']:.1f}%", stats['Classificação CV'])
                c4.metric("Normalidade", stats.get('Dist. Normal?', 'N/A'))
                
                with st.expander("📋 Ver Todas as Estatísticas (22 métricas)"):
                    stats_df = pd.DataFrame(list(stats.items()), columns=['Estatística', 'Valor'])
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.subheader("📊 Distribuição com Média e Mediana")
                bins = st.slider("Bins:", 5, 100, 30, key="bins_stats")
                df_plot = pd.DataFrame({col_analise: df[col_analise].dropna().values})
                chart_h = criar_histograma_aprimorado(df_plot, col_analise, bins)
                if chart_h:
                    st.altair_chart(chart_h, use_container_width=True)
                
                st.markdown("---")
                st.subheader("📉 Boxplot")
                chart_box = criar_boxplot(df_plot, col_analise)
                if chart_box:
                    st.altair_chart(chart_box, use_container_width=True)
                
                if len(colunas_num) >= 2:
                    st.markdown("---")
                    st.subheader("🔗 Matriz de Correlação de Pearson")
                    corr = df[colunas_num].dropna(axis=1, how="all").corr()
                    if not corr.empty:
                        st.dataframe(corr.round(3), use_container_width=True)
                
                date_cols = st.session_state.get('date_columns', [])
                if date_cols and fonte_dados == 'Dados Originais':
                    st.markdown("---")
                    st.subheader("📅 Série Temporal")
                    col_data = st.selectbox("Data:", date_cols, key="dt_s")
                    df_plot = df[[col_data, col_analise]].dropna().copy()
                    if df_plot[col_data].dtype != 'datetime64[ns]':
                        df_plot[col_data] = pd.to_datetime(df_plot[col_data], errors='coerce')
                    df_plot = df_plot.dropna()
                    if len(df_plot) > 0:
                        chart_t = criar_grafico_linhas(df_plot, col_data, [col_analise])
                        if chart_t:
                            st.altair_chart(chart_t, use_container_width=True)

# ============================================================
# ABA 4: GRÁFICOS
# ============================================================
with aba4:
    st.header("📈 Visualização Gráfica")
    st.markdown("---")
    
    fonte_graf = st.radio(
        "Fonte:",
        ['Dados Originais', 'Média Mensal/Anual'],
        horizontal=True,
        key="fonte_graf"
    )
    
    if fonte_graf == 'Média Mensal/Anual' and st.session_state.get('df_mensal') is not None:
        df = st.session_state['df_mensal'].copy()
        st.info("📆 Usando dados agregados")
    else:
        df = obter_df_limpo()
    
    if df is None:
        st.warning("⚠️ Carregue os dados primeiro!")
    else:
        colunas_num = df.select_dtypes(include=[np.number]).columns.tolist()
        todas_cols = df.columns.tolist()
        
        if not colunas_num:
            st.warning("⚠️ Sem variáveis numéricas")
        else:
            tipo = st.radio("Tipo:", ['📈 Linhas', '📊 Histograma', '📉 Boxplot', '📋 Barras'], horizontal=True, key="tipo_g")
            st.markdown("---")
            
            if tipo == '📈 Linhas':
                date_cols = st.session_state.get('date_columns', [])
                x_cols = date_cols if date_cols else todas_cols
                col_x = st.selectbox("Eixo X:", x_cols, key="gx")
                col_y = st.multiselect("Eixo Y:", colunas_num, default=colunas_num[:min(3, len(colunas_num))], key="gy")
                if col_y:
                    df_plot = df[[col_x] + col_y].dropna().copy()
                    if df_plot[col_x].dtype != 'datetime64[ns]':
                        try:
                            df_plot[col_x] = pd.to_datetime(df_plot[col_x], errors='coerce')
                        except:
                            pass
                    df_plot = df_plot.dropna()
                    if len(df_plot) > 0:
                        chart = criar_grafico_linhas(df_plot, col_x, col_y)
                        if chart:
                            st.altair_chart(chart, use_container_width=True)
            
            elif tipo == '📊 Histograma':
                col_h = st.selectbox("Variável:", colunas_num, key="gh")
                bins_h = st.slider("Bins:", 5, 100, 30, key="gb")
                df_plot = pd.DataFrame({col_h: df[col_h].dropna().values})
                chart = criar_histograma_aprimorado(df_plot, col_h, bins_h)
                if chart:
                    st.altair_chart(chart, use_container_width=True)
            
            elif tipo == '📉 Boxplot':
                col_b = st.selectbox("Variável:", colunas_num, key="gbox")
                df_plot = pd.DataFrame({col_b: df[col_b].dropna().values})
                chart = criar_boxplot(df_plot, col_b)
                if chart:
                    st.altair_chart(chart, use_container_width=True)
            
            elif tipo == '📋 Barras':
                st.info("💡 Ideal para dados categóricos ou agregados")
                
                if st.session_state.get('df_mensal') is not None and fonte_graf == 'Média Mensal/Anual':
                    x_default = df.columns[0]
                else:
                    x_default = todas_cols[0]
                
                col_x_bar = st.selectbox("Eixo X:", todas_cols, 
                                        index=todas_cols.index(x_default) if x_default in todas_cols else 0,
                                        key="bx")
                col_y_bar = st.selectbox("Eixo Y:", colunas_num, key="by")
                
                df_plot = df[[col_x_bar, col_y_bar]].dropna()
                if len(df_plot) > 0:
                    chart = criar_grafico_barras(df_plot, col_x_bar, col_y_bar,
                                                f'{col_y_bar} por {col_x_bar}')
                    if chart:
                        st.altair_chart(chart, use_container_width=True)

# ============================================================
# ABA 5: DOWNLOAD
# ============================================================
with aba5:
    st.header("💾 Download")
    st.markdown("---")
    
    df = obter_df_limpo()
    df_agg = st.session_state.get('df_mensal')
    
    if df is not None:
        st.markdown('<div class="success-box"><h3>✅ Dados disponíveis</h3></div>', unsafe_allow_html=True)
        
        st.subheader("📥 Dados Processados")
        csv_data = df.to_csv(index=False, encoding='utf-8')
        st.download_button("📥 Baixar CSV (Original)", csv_data,
                          f"dados_processados_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
        
        if df_agg is not None:
            st.markdown("---")
            st.subheader("📥 Média Mensal/Anual")
            csv_agg = df_agg.to_csv(index=False, encoding='utf-8')
            st.download_button("📥 Baixar CSV (Agregado)", csv_agg,
                              f"media_agregada_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
        
        st.markdown("---")
        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.warning("⚠️ Carregue os dados primeiro!")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:1.5rem; color:#6c757d; background:#f8f9fa; border-radius:12px;">
    <h4>🌱 AgroDataLab v9.1</h4>
    <p>Análise Climática Avançada | Estatísticas Robustas</p>
    <p style="font-size:0.8rem;">Licença MIT © 2024</p>
</div>
""", unsafe_allow_html=True)
