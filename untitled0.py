# ============================================================
# AgroDataLab - Sistema Completo de Análise Meteorológica
# Versão 11.1 - Agregação Robusta (ME/YE) + Diagnóstico
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
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None
if 'df_processed' not in st.session_state:
    st.session_state['df_processed'] = None
if 'upload_feito' not in st.session_state:
    st.session_state['upload_feito'] = False
if 'agregacoes' not in st.session_state:
    st.session_state['agregacoes'] = {}
if 'date_columns' not in st.session_state:
    st.session_state['date_columns'] = []

# ============================================================
# FREQUÊNCIAS CORRETAS (Pandas 2.2+)
# ============================================================
FREQ_MAP = {
    "Semanal": "W",
    "Mensal": "ME",
    "Anual": "YE"
}

# ============================================================
# FUNÇÕES BÁSICAS
# ============================================================
def corrigir_nomes_duplicados(df):
    cols = pd.Series(df.columns)
    for nome in cols[cols.duplicated()].unique():
        idx = cols[cols == nome].index.tolist()
        for i, pos in enumerate(idx):
            if i > 0:
                cols[pos] = f"{nome}_{i}"
    df.columns = cols
    return df

def detectar_delimitador(arquivo):
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
            delim = sniffer.sniff(amostra).delimiter
            linhas = amostra.strip().split('\n')
            if len(linhas) >= 2 and linhas[0].count(delim) == linhas[1].count(delim) > 0:
                return delim
        except:
            pass
        for sep in [';', ',', '\t', '|']:
            linhas = amostra.strip().split('\n')
            if len(linhas) >= 2 and linhas[0].count(sep) == linhas[1].count(sep) > 0:
                return sep
        return ';' if amostra.count(';') > amostra.count(',') else ','
    except:
        return ','

def carregar_dados(arquivo):
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

def obter_df_processado():
    if st.session_state['df_processed'] is None:
        return None
    df = st.session_state['df_processed'].copy()
    df = corrigir_nomes_duplicados(df)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# ============================================================
# AGREGAÇÃO TEMPORAL ROBUSTA (CORRIGIDA)
# ============================================================
def criar_agregacao_temporal(df, coluna_data, freq):
    """
    Agregação temporal robusta.
    Verifica tipo datetime antes de converter.
    """
    try:
        df_temp = df.copy()
        
        # Verificar se já é datetime, senão converter
        if not pd.api.types.is_datetime64_any_dtype(df_temp[coluna_data]):
            df_temp[coluna_data] = pd.to_datetime(
                df_temp[coluna_data],
                dayfirst=True,
                errors="coerce"
            )
        
        # Remover nulos
        df_temp = df_temp.dropna(subset=[coluna_data])
        
        if df_temp.empty:
            st.error("❌ Nenhuma data válida encontrada.")
            return None
        
        # Ordenar por data
        df_temp = df_temp.sort_values(coluna_data)
        
        # Selecionar colunas numéricas
        colunas_num = df_temp.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(colunas_num) == 0:
            st.error("❌ Nenhuma coluna numérica encontrada.")
            return None
        
        # Agregar
        resultado = (
            df_temp
            .set_index(coluna_data)[colunas_num]
            .resample(freq)
            .mean()
            .round(2)
            .reset_index()
        )
        
        return resultado
    
    except Exception as e:
        st.error(f"❌ Erro na agregação: {e}")
        with st.expander("🔍 Diagnóstico"):
            st.write(f"Frequência usada: {freq}")
            st.write(f"Tipo da coluna: {df[coluna_data].dtype}")
            st.write(f"Primeiros valores: {df[coluna_data].head(3).tolist()}")
        return None

# ============================================================
# ESTATÍSTICAS COMPLETAS
# ============================================================
def calcular_estatisticas_completas(df, coluna):
    dados = df[coluna].dropna()
    if len(dados) < 3:
        return None
    
    n = len(dados)
    media = np.mean(dados)
    mediana = np.median(dados)
    moda = dados.mode().iloc[0] if len(dados.mode()) > 0 else np.nan
    desvio = np.std(dados, ddof=1)
    erro_padrao = desvio / np.sqrt(n)
    cv = (desvio / media * 100) if media != 0 else 0
    minimo = np.min(dados)
    maximo = np.max(dados)
    
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1
    p5 = np.percentile(dados, 5)
    p95 = np.percentile(dados, 95)
    assimetria = dados.skew()
    curtose = dados.kurtosis()
    
    if cv <= 10: cv_class = "Baixo 🟢"
    elif cv <= 20: cv_class = "Médio 🟡"
    elif cv <= 30: cv_class = "Alto 🟠"
    else: cv_class = "Muito Alto 🔴"
    
    resultado = {
        'N': n, 'Média': round(media, 4), 'Mediana': round(mediana, 4),
        'Moda': round(moda, 4) if not pd.isna(moda) else 'N/A',
        'Desvio Padrão': round(desvio, 4), 'Erro Padrão': round(erro_padrao, 4),
        'CV (%)': round(cv, 2), 'Classificação CV': cv_class,
        'Mínimo': round(minimo, 4), 'Máximo': round(maximo, 4),
        'Q1': round(q1, 4), 'Q3': round(q3, 4), 'IQR': round(iqr, 4),
        'P5': round(p5, 4), 'P95': round(p95, 4),
        'Assimetria': round(assimetria, 4), 'Curtose': round(curtose, 4),
    }
    
    if SCIPY_DISPONIVEL and 3 <= n <= 5000:
        try:
            stat_sw, p_sw = scipy_stats.shapiro(dados)
            resultado['Shapiro-Wilk p'] = round(p_sw, 4)
            resultado['Dist. Normal?'] = "Sim ✅" if p_sw > 0.05 else "Não ❌"
        except:
            resultado['Dist. Normal?'] = 'N/A'
    else:
        resultado['Dist. Normal?'] = 'N/A'
    
    return resultado

def detectar_dataframe_colunas_data(df):
    """Detecta colunas datetime em qualquer DataFrame"""
    if df is None:
        return []
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

def criar_grafico_series_temporal(df, x_col, y_cols, titulo="Série Temporal"):
    try:
        df = df.loc[:, ~df.columns.duplicated()]
        y_cols_validas = [c for c in y_cols if c in df.columns]
        if not y_cols_validas:
            return None
        
        df_melted = pd.melt(df, id_vars=[x_col], value_vars=y_cols_validas,
                           var_name='Variável', value_name='Valor')
        
        chart = alt.Chart(df_melted).mark_line(strokeWidth=2).encode(
            x=alt.X(f'{x_col}:T', title='Data', axis=alt.Axis(format='%d/%m/%Y')),
            y=alt.Y('Valor:Q', title='Valor'),
            color=alt.Color('Variável:N', legend=alt.Legend(orient='bottom')),
            tooltip=[alt.Tooltip(f'{x_col}:T', title='Data', format='%d/%m/%Y'),
                    'Variável:N', alt.Tooltip('Valor:Q', format='.2f')]
        ).properties(title=titulo, height=400).interactive()
        return chart
    except Exception as e:
        st.error(f"Erro no gráfico: {str(e)}")
        return None

def detectar_outliers(dados):
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    return (dados < lim_inf) | (dados > lim_sup), lim_inf, lim_sup

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero-header">
    <h1>🌱 AgroDataLab</h1>
    <p>Sistema Inteligente de Análise Meteorológica e Agronômica</p>
    <p style="font-size:0.9rem; opacity:0.8;">v11.1 - Agregação Robusta (ME/YE)</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📊 Status")
    
    if SCIPY_DISPONIVEL:
        st.success("✅ SciPy disponível")
    
    if st.session_state['upload_feito']:
        st.success(f"✅ Dados carregados")
        
        agregacoes = st.session_state['agregacoes']
        if agregacoes:
            st.markdown("**📆 Agregações:**")
            for nome in ["Semanal", "Mensal", "Anual"]:
                df_agg = agregacoes.get(nome)
                if df_agg is not None:
                    st.caption(f"• {nome}: {len(df_agg)} registros")
        else:
            st.caption("Nenhuma agregação gerada")
    else:
        st.info("📤 Aguardando upload")
    
    st.markdown("---")
    st.caption("AgroDataLab v11.1 | MIT License")

# ============================================================
# ABAS
# ============================================================
aba1, aba2, aba3, aba4, aba5 = st.tabs([
    "📤 1. Upload",
    "🔧 2. Tratamento",
    "📊 3. Estatísticas",
    "📈 4. Série Temporal",
    "💾 5. Download"
])

# ============================================================
# ABA 1: UPLOAD (EXECUTA APENAS UMA VEZ)
# ============================================================
with aba1:
    st.header("📤 Upload de Arquivo")
    st.markdown("---")
    
    if st.session_state['upload_feito']:
        st.success("✅ Arquivo já carregado!")
        st.dataframe(st.session_state['df_original'].head(10), use_container_width=True)
    else:
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
                    st.session_state['upload_feito'] = True
                    st.session_state['agregacoes'] = {}
                    
                    st.markdown(f'<div class="success-box"><h3>✅ {df.shape[0]:,} linhas × {df.shape[1]} colunas</h3></div>', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Registros", f"{len(df):,}")
                    col2.metric("Colunas", len(df.columns))
                    col3.metric("Numéricas", len(df.select_dtypes(include=[np.number]).columns))
                    col4.metric("Datas", len(date_cols))
                    
                    st.rerun()

# ============================================================
# ABA 2: TRATAMENTO
# ============================================================
with aba2:
    st.header("🔧 Tratamento de Dados")
    st.markdown("---")
    
    if not st.session_state['upload_feito']:
        st.warning("⚠️ Faça o upload na Aba 1 primeiro!")
    else:
        df = obter_df_processado()
        if df is None:
            st.warning("⚠️ Erro ao carregar dados.")
        else:
            # Valores ausentes
            missing_total = df.isnull().sum().sum()
            if missing_total > 0:
                st.markdown(f'<div class="warning-box"><h3>⚠️ {missing_total:,} valores ausentes</h3></div>', unsafe_allow_html=True)
                
                missing_df = pd.DataFrame({
                    'Coluna': df.columns, 'Ausentes': df.isnull().sum().values,
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
                        st.session_state['df_processed'] = st.session_state['df_original'].copy()
                        st.success("✅ Restaurado!")
                        st.rerun()
            else:
                st.markdown('<div class="success-box"><h3>✅ Nenhum valor ausente!</h3></div>', unsafe_allow_html=True)
            
            # AGREGAÇÃO TEMPORAL
            st.markdown("---")
            st.subheader("📆 Agregação Temporal")
            
            date_cols = st.session_state.get('date_columns', [])
            colunas_num = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if date_cols and colunas_num:
                col_data_agg = st.selectbox("Coluna de data:", date_cols, key="agg_date")
                periodo_nome = st.radio("Período:", list(FREQ_MAP.keys()), horizontal=True, key="agg_freq")
                
                if st.button("📊 Gerar Média " + periodo_nome, key="btn_agg"):
                    df_atual = obter_df_processado()
                    freq_code = FREQ_MAP[periodo_nome]
                    
                    with st.spinner("Calculando..."):
                        df_agg = criar_agregacao_temporal(df_atual, col_data_agg, freq_code)
                        
                        if df_agg is not None:
                            st.session_state['agregacoes'][periodo_nome] = df_agg
                            st.success(f"✅ Média {periodo_nome.lower()} gerada! ({len(df_agg)} registros)")
                            st.rerun()
                
                # Mostrar agregações
                agregacoes = st.session_state['agregacoes']
                if agregacoes:
                    st.markdown("---")
                    st.markdown("### 📋 Agregações Geradas")
                    
                    for nome in ["Semanal", "Mensal", "Anual"]:
                        df_agg = agregacoes.get(nome)
                        if df_agg is not None:
                            with st.expander(f"📆 {nome} ({len(df_agg)} registros)", expanded=False):
                                st.dataframe(df_agg, use_container_width=True)
                                csv_agg = df_agg.to_csv(index=False, encoding='utf-8')
                                st.download_button(
                                    label=f"📥 Baixar {nome}",
                                    data=csv_agg,
                                    file_name=f"dados_{nome.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv",
                                    key=f"dw_{nome}"
                                )
            else:
                if not date_cols:
                    st.info("📅 Nenhuma coluna de data identificada.")
                if not colunas_num:
                    st.info("📊 Nenhuma variável numérica disponível.")
            
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
                        st.write(f"Limites: [{li:.3f}, {ls:.3f}]")
                    else:
                        st.success(f"✅ Nenhum outlier em {col_out}")

# ============================================================
# ABA 3: ESTATÍSTICAS
# ============================================================
with aba3:
    st.header("📊 Análise Estatística Completa")
    st.markdown("---")
    
    if not st.session_state['upload_feito']:
        st.warning("⚠️ Faça o upload na Aba 1 primeiro!")
    else:
        opcoes = ["Dados Originais"]
        for nome in ["Semanal", "Mensal", "Anual"]:
            if nome in st.session_state['agregacoes'] and st.session_state['agregacoes'][nome] is not None:
                opcoes.append(f"Dados {nome}")
        
        fonte = st.radio("Fonte:", opcoes, horizontal=True, key="fonte_stats")
        
        if fonte == "Dados Originais":
            df = obter_df_processado()
        else:
            nome_agg = fonte.replace("Dados ", "")
            df = st.session_state['agregacoes'].get(nome_agg)
        
        if df is None:
            st.warning("⚠️ Dados não disponíveis.")
        else:
            colunas_num = df.select_dtypes(include=[np.number]).columns.tolist()
            if not colunas_num:
                st.warning("⚠️ Nenhuma variável numérica.")
            else:
                col_analise = st.selectbox("Variável:", colunas_num, key="stat_col")
                stats = calcular_estatisticas_completas(df, col_analise)
                
                if stats:
                    st.subheader(f"📈 {col_analise} ({fonte})")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Média", f"{stats['Média']:.3f}")
                    c2.metric("Desvio", f"{stats['Desvio Padrão']:.3f}")
                    c3.metric("CV (%)", f"{stats['CV (%)']:.1f}%", stats['Classificação CV'])
                    c4.metric("Normalidade", stats.get('Dist. Normal?', 'N/A'))
                    
                    with st.expander("📋 Todas as Estatísticas"):
                        stats_df = pd.DataFrame(list(stats.items()), columns=['Estatística', 'Valor'])
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    
                    if len(colunas_num) >= 2:
                        st.markdown("---")
                        st.subheader("🔗 Correlação de Pearson")
                        corr = df[colunas_num].dropna(axis=1, how="all").corr()
                        if not corr.empty:
                            st.dataframe(corr.round(3), use_container_width=True)

# ============================================================
# ABA 4: SÉRIE TEMPORAL
# ============================================================
with aba4:
    st.header("📈 Série Temporal")
    st.markdown("---")
    
    if not st.session_state['upload_feito']:
        st.warning("⚠️ Faça o upload na Aba 1 primeiro!")
    else:
        opcoes = ["Dados Originais"]
        for nome in ["Semanal", "Mensal", "Anual"]:
            if nome in st.session_state['agregacoes'] and st.session_state['agregacoes'][nome] is not None:
                opcoes.append(f"Dados {nome}")
        
        fonte = st.radio("Fonte:", opcoes, horizontal=True, key="fonte_graf")
        
        if fonte == "Dados Originais":
            df = obter_df_processado()
        else:
            nome_agg = fonte.replace("Dados ", "")
            df = st.session_state['agregacoes'].get(nome_agg)
        
        if df is None:
            st.warning("⚠️ Dados não disponíveis.")
        else:
            colunas_num = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not colunas_num:
                st.warning("⚠️ Sem variáveis numéricas")
            else:
                # Detectar colunas de data no próprio DataFrame
                date_cols = detectar_dataframe_colunas_data(df)
                if not date_cols:
                    date_cols = st.session_state.get('date_columns', [])
                if not date_cols:
                    date_cols = [df.columns[0]]
                
                col_x = st.selectbox("Eixo X (Data):", date_cols, key="gx")
                col_y = st.multiselect("Variáveis Y:", colunas_num,
                                      default=colunas_num[:min(3, len(colunas_num))], key="gy")
                
                if col_y:
                    df_plot = df[[col_x] + col_y].dropna().copy()
                    
                    if df_plot[col_x].dtype != 'datetime64[ns]':
                        df_plot[col_x] = pd.to_datetime(df_plot[col_x], dayfirst=True, errors='coerce')
                    df_plot = df_plot.dropna()
                    
                    if len(df_plot) > 0:
                        st.caption(f"📅 Período: {df_plot[col_x].min().strftime('%d/%m/%Y')} → {df_plot[col_x].max().strftime('%d/%m/%Y')}")
                        
                        chart = criar_grafico_series_temporal(df_plot, col_x, col_y)
                        if chart:
                            st.altair_chart(chart, use_container_width=True)

# ============================================================
# ABA 5: DOWNLOAD
# ============================================================
with aba5:
    st.header("💾 Download dos Resultados")
    st.markdown("---")
    
    if not st.session_state['upload_feito']:
        st.warning("⚠️ Faça o upload na Aba 1 primeiro!")
    else:
        agregacoes = st.session_state['agregacoes']
        
        if not agregacoes:
            st.warning("⚠️ Nenhum resultado disponível.")
            st.info("💡 Vá para Aba 2 → gere as agregações (Semanal/Mensal/Anual)")
        else:
            for nome in ["Semanal", "Mensal", "Anual"]:
                df_agg = agregacoes.get(nome)
                if df_agg is not None:
                    st.subheader(f"📆 Resultados {nome} ({len(df_agg)} registros)")
                    st.dataframe(df_agg.head(10), use_container_width=True)
                    
                    csv_agg = df_agg.to_csv(index=False, encoding='utf-8')
                    st.download_button(
                        label=f"📥 Baixar CSV ({nome})",
                        data=csv_agg,
                        file_name=f"resultados_{nome.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        key=f"res_{nome}"
                    )
                    st.markdown("---")
        
        with st.expander("📥 Dados Processados (Original)", expanded=False):
            df = obter_df_processado()
            if df is not None:
                st.dataframe(df.head(5), use_container_width=True)
                csv_data = df.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="📥 Baixar Dados Processados",
                    data=csv_data,
                    file_name=f"dados_processados_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    key="proc"
                )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:1.5rem; color:#6c757d; background:#f8f9fa; border-radius:12px;">
    <h4>🌱 AgroDataLab v11.1</h4>
    <p>Agregação Robusta (ME/YE) | Diagnóstico Incluído</p>
    <p style="font-size:0.8rem;">Licença MIT © 2024</p>
</div>
""", unsafe_allow_html=True)
