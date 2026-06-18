# ============================================================
# AgroDataLab - Sistema Completo de Análise Meteorológica
# Versão Final Corrigida - 100% Funcional
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
# CSS PROFISSIONAL
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .hero-header {
        background: linear-gradient(135deg, #0d2818 0%, #1a5632 30%, #2d8a4e 70%, #4caf50 100%);
        padding: 2.5rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 20px 60px rgba(26,86,50,0.3);
    }
    
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border-left: 4px solid #2d8a4e;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a5632;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
        text-transform: uppercase;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1a5632 0%, #2d8a4e 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
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
    
    .app-footer {
        text-align: center;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 12px;
        margin-top: 3rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNÇÕES
# ============================================================

def calcular_estatisticas_descritivas(df, coluna):
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
        'n': n, 'Média': round(media, 4), 'Mediana': round(mediana, 4),
        'Desvio Padrão': round(desvio_padrao, 4), 'Variância': round(variancia, 4),
        'Mínimo': round(minimo, 4), 'Máximo': round(maximo, 4),
        'Amplitude': round(amplitude, 4), 'Q1 (25%)': round(q1, 4),
        'Q3 (75%)': round(q3, 4), 'IQR': round(iqr, 4),
        'CV (%)': round(cv, 2), 'Assimetria': round(assimetria, 4),
        'Curtose': round(curtose, 4)
    }

def classificar_cv(cv):
    if cv <= 10:
        return "Baixo 🟢"
    elif cv <= 20:
        return "Médio 🟡"
    elif cv <= 30:
        return "Alto 🟠"
    else:
        return "Muito Alto 🔴"

def detectar_outliers_iqr(dados):
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    outliers = (dados < limite_inferior) | (dados > limite_superior)
    return outliers, limite_inferior, limite_superior

def criar_grafico_linhas(df, x_col, y_cols, titulo="Série Temporal"):
    df_melted = df.melt(id_vars=[x_col], value_vars=y_cols, var_name='Variável', value_name='Valor')
    
    chart = alt.Chart(df_melted).mark_line(strokeWidth=2.5, point=True).encode(
        x=alt.X(f'{x_col}:T', title='Data'),
        y=alt.Y('Valor:Q', title='Valor'),
        color='Variável:N',
        tooltip=[f'{x_col}:T', 'Variável:N', alt.Tooltip('Valor:Q', format='.2f')]
    ).properties(title=titulo, height=400).interactive()
    
    return chart

def criar_histograma(df, coluna, bins=30):
    chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
        alt.X(f'{coluna}:Q', bin=alt.Bin(maxbins=bins), title=coluna),
        alt.Y('count()', title='Frequência'),
        tooltip=['count()']
    ).properties(title=f'Distribuição de {coluna}', height=400)
    
    return chart

def criar_boxplot(df, coluna):
    chart = alt.Chart(df).mark_boxplot(extent='min-max', color='#2d8a4e').encode(
        y=alt.Y(f'{coluna}:Q', title=coluna)
    ).properties(height=400)
    
    return chart

def carregar_dados(arquivo):
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
    
    return df

def agregar_dados_temporais(df, coluna_data, frequencia='D'):
    df = df.copy()
    df[coluna_data] = pd.to_datetime(df[coluna_data], errors='coerce')
    df.set_index(coluna_data, inplace=True)
    
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    freq_map = {'D': 'D', 'W': 'W', 'M': 'M', 'Y': 'Y'}
    df_agg = df[colunas_numericas].resample(freq_map.get(frequencia, 'D')).agg(['mean', 'std', 'min', 'max', 'count'])
    
    return df_agg.round(3)

# ============================================================
# INTERFACE
# ============================================================

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
        st.success("✅ Dados Carregados")
        st.metric("Registros", f"{len(df_status):,}")
        st.metric("Variáveis", len(df_status.columns))
        
        ausentes = df_status.isnull().sum().sum()
        total = len(df_status) * len(df_status.columns)
        pct_ausentes = (ausentes / total * 100) if total > 0 else 0
        st.metric("Dados Ausentes", f"{pct_ausentes:.1f}%")
    else:
        st.info("📤 Aguardando upload...")
    
    st.markdown("---")
    st.markdown("### 📚 Guia Rápido")
    st.markdown("""
    1. **Upload** - Carregue seus dados
    2. **Tratamento** - Limpe e prepare
    3. **Análise** - Estatísticas
    4. **Visualização** - Gráficos
    5. **Download** - Exporte
    """)

# ============================================================
# ABAS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Upload & Preview",
    "🔧 Tratamento de Dados",
    "📊 Análise Estatística",
    "📈 Visualização Gráfica",
    "💾 Download & Exportação"
])

# ============================================================
# ABA 1: UPLOAD
# ============================================================
with tab1:
    st.markdown("### 📤 Carregamento de Dados")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Arraste ou selecione seu arquivo",
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        with st.spinner('🔄 Processando...'):
            df = carregar_dados(uploaded_file)
            
            if df is not None:
                df, date_cols = processar_dados_inmet(df)
                
                st.session_state['df_original'] = df.copy()
                st.session_state['df_processed'] = df.copy()
                st.session_state['date_columns'] = date_cols
                
                st.markdown('<div class="success-box"><h3>✅ Dados Carregados!</h3></div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Registros", f"{len(df):,}")
                col2.metric("Variáveis", len(df.columns))
                col3.metric("Ausentes", df.isnull().sum().sum())
                col4.metric("Memória", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                
                st.markdown("---")
                st.markdown("### 👀 Preview dos Dados")
                st.dataframe(df.head(15), use_container_width=True, height=400)
                
                if date_cols:
                    st.success(f"📅 Colunas de data: {', '.join(date_cols)}")

# ============================================================
# ABA 2: TRATAMENTO
# ============================================================
with tab2:
    if 'df_processed' not in st.session_state or st.session_state['df_processed'] is None:
        st.info("📤 Carregue seus dados na Aba 1 primeiro!")
    else:
        df = st.session_state['df_processed']
        
        st.markdown("### 🔧 Tratamento e Limpeza de Dados")
        st.markdown("---")
        
        missing_count = df.isnull().sum().sum()
        
        if missing_count > 0:
            total_cells = len(df) * len(df.columns)
            missing_pct = (missing_count / total_cells * 100) if total_cells > 0 else 0
            
            st.markdown(f"""
            <div class="warning-box">
                <h3>⚠️ {missing_count:,} valores ausentes ({missing_pct:.2f}%)</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Mostrar colunas com ausentes
            missing_df = pd.DataFrame({
                'Coluna': df.columns,
                'Ausentes': df.isnull().sum().values,
                '%': (df.isnull().sum() / len(df) * 100).round(2).values
            })
            missing_df = missing_df[missing_df['Ausentes'] > 0].sort_values('Ausentes', ascending=False)
            st.dataframe(missing_df, use_container_width=True)
            
            st.markdown("### 🛠️ Métodos de Preenchimento")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Aplicar Média", key="btn_mean_aba2"):
                    st.session_state['df_processed'] = preencher_dados_ausentes(df.copy(), 'media')
                    st.success("✅ Preenchido com média!")
                    st.rerun()
            
            with col2:
                if st.button("📈 Aplicar Mediana", key="btn_median_aba2"):
                    st.session_state['df_processed'] = preencher_dados_ausentes(df.copy(), 'mediana')
                    st.success("✅ Preenchido com mediana!")
                    st.rerun()
            
            with col3:
                if st.button("🔄 Interpolar", key="btn_interp_aba2"):
                    st.session_state['df_processed'] = preencher_dados_ausentes(df.copy(), 'interpolacao_linear')
                    st.success("✅ Interpolado!")
                    st.rerun()
            
            col4, col5 = st.columns(2)
            
            with col4:
                if st.button("🗑️ Remover Linhas", key="btn_drop_aba2"):
                    df_dropped = df.dropna()
                    st.session_state['df_processed'] = df_dropped
                    st.success(f"✅ {len(df) - len(df_dropped)} linhas removidas!")
                    st.rerun()
            
            with col5:
                if st.button("🔙 Restaurar Original", key="btn_restore_aba2"):
                    if 'df_original' in st.session_state:
                        st.session_state['df_processed'] = st.session_state['df_original'].copy()
                        st.success("✅ Restaurado!")
                        st.rerun()
        else:
            st.markdown('<div class="success-box"><h3>✅ Nenhum valor ausente!</h3></div>', unsafe_allow_html=True)
        
        # Outliers
        st.markdown("---")
        st.markdown("### 🔍 Detecção de Outliers (IQR)")
        
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        if colunas_numericas:
            col_outlier = st.selectbox(
                "Variável para análise de outliers:",
                colunas_numericas,
                key="outlier_select_aba2"
            )
            
            dados_coluna = df[col_outlier].dropna()
            if len(dados_coluna) > 0:
                outliers, lim_inf, lim_sup = detectar_outliers_iqr(dados_coluna)
                n_outliers = outliers.sum()
                
                if n_outliers > 0:
                    st.warning(f"{n_outliers} outliers detectados ({n_outliers/len(dados_coluna)*100:.1f}%)")
                    chart = criar_boxplot(df.dropna(subset=[col_outlier]), col_outlier)
                    st.altair_chart(chart, use_container_width=True)
                    st.write(f"Limites: [{lim_inf:.2f}, {lim_sup:.2f}]")
                else:
                    st.success(f"Nenhum outlier em {col_outlier}")

# ============================================================
# ABA 3: ANÁLISE ESTATÍSTICA
# ============================================================
with tab3:
    if 'df_processed' not in st.session_state or st.session_state['df_processed'] is None:
        st.info("📤 Carregue seus dados na Aba 1 primeiro!")
    else:
        df = st.session_state['df_processed']
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        st.markdown("### 📊 Análise Estatística Detalhada")
        st.markdown("---")
        
        if colunas_numericas:
            col_analise = st.selectbox(
                "Selecione a variável:",
                colunas_numericas,
                key="stats_select_aba3"
            )
            
            stats = calcular_estatisticas_descritivas(df, col_analise)
            
            if stats:
                st.markdown(f"#### 📈 {col_analise}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Média", f"{stats['Média']:.2f}")
                col2.metric("Desvio Padrão", f"{stats['Desvio Padrão']:.2f}")
                col3.metric("CV (%)", f"{stats['CV (%)']:.1f}%", classificar_cv(stats['CV (%)']))
                col4.metric("Amostras", f"{stats['n']:,}")
                
                # Correlação
                if len(colunas_numericas) >= 2:
                    st.markdown("---")
                    st.markdown("### 🔗 Matriz de Correlação")
                    corr_matrix = df[colunas_numericas].corr()
                    st.dataframe(corr_matrix.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1).format("{:.3f}"), use_container_width=True)
                
                # Análise temporal
                if 'date_columns' in st.session_state and len(st.session_state['date_columns']) > 0:
                    st.markdown("---")
                    st.markdown("### 📅 Análise Temporal")
                    
                    coluna_data = st.selectbox(
                        "Coluna de data:",
                        st.session_state['date_columns'],
                        key="date_select_aba3"
                    )
                    
                    frequencia = st.selectbox(
                        "Período:",
                        ['D', 'W', 'M', 'Y'],
                        format_func=lambda x: {'D': 'Diário', 'W': 'Semanal', 'M': 'Mensal', 'Y': 'Anual'}[x],
                        key="freq_select_aba3"
                    )
                    
                    try:
                        df_agg = agregar_dados_temporais(df, coluna_data, frequencia)
                        st.dataframe(df_agg, use_container_width=True, height=400)
                        
                        df_temp = df[[coluna_data, col_analise]].copy()
                        df_temp[coluna_data] = pd.to_datetime(df_temp[coluna_data], errors='coerce')
                        df_temp = df_temp.dropna()
                        
                        if len(df_temp) > 0:
                            chart = criar_grafico_linhas(df_temp, coluna_data, [col_analise], f'Série Temporal - {col_analise}')
                            st.altair_chart(chart, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erro: {str(e)}")
        else:
            st.warning("Nenhuma variável numérica encontrada.")

# ============================================================
# ABA 4: VISUALIZAÇÃO
# ============================================================
with tab4:
    if 'df_processed' not in st.session_state or st.session_state['df_processed'] is None:
        st.info("📤 Carregue seus dados na Aba 1 primeiro!")
    else:
        df = st.session_state['df_processed']
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        st.markdown("### 📈 Visualização Gráfica Interativa")
        st.markdown("---")
        
        if colunas_numericas:
            tipo_grafico = st.radio(
                "Tipo de visualização:",
                ['📈 Série Temporal', '📊 Histograma', '📉 Boxplot'],
                horizontal=True,
                key="chart_type_aba4"
            )
            
            if tipo_grafico == '📈 Série Temporal':
                date_cols = st.session_state.get('date_columns', [])
                if date_cols:
                    x_col = st.selectbox("Eixo X (Data):", date_cols, key="x_select_aba4")
                    y_cols = st.multiselect(
                        "Variáveis Y:",
                        colunas_numericas,
                        default=colunas_numericas[:2] if len(colunas_numericas) >= 2 else colunas_numericas,
                        key="y_multiselect_aba4"
                    )
                    
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
                col_hist = st.selectbox("Variável:", colunas_numericas, key="hist_select_aba4")
                n_bins = st.slider("Intervalos:", 5, 100, 30, key="bins_slider_aba4")
                chart = criar_histograma(df.dropna(subset=[col_hist]), col_hist, bins=n_bins)
                st.altair_chart(chart, use_container_width=True)
            
            elif tipo_grafico == '📉 Boxplot':
                col_box = st.selectbox("Variável:", colunas_numericas, key="box_select_aba4")
                chart = criar_boxplot(df.dropna(subset=[col_box]), col_box)
                st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("Nenhuma variável numérica disponível.")

# ============================================================
# ABA 5: DOWNLOAD
# ============================================================
with tab5:
    if 'df_processed' not in st.session_state or st.session_state['df_processed'] is None:
        st.info("📤 Carregue e processe seus dados primeiro!")
    else:
        df = st.session_state['df_processed']
        
        st.markdown("### 💾 Download e Exportação")
        st.markdown("---")
        
        st.markdown('<div class="success-box"><h3>✅ Dados Prontos!</h3></div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Registros", f"{len(df):,}")
        col2.metric("Colunas", len(df.columns))
        col3.metric("Ausentes", df.isnull().sum().sum())
        col4.metric("Tamanho", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        st.markdown("---")
        
        csv = df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="📥 Baixar CSV",
            data=csv,
            file_name=f"dados_processados_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_csv_aba5"
        )
        
        st.markdown("---")
        st.markdown("### 📋 Preview")
        st.dataframe(df.head(20), use_container_width=True, height=400)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div class="app-footer">
    <h4>🌱 AgroDataLab v2.0</h4>
    <p>Sistema de Análise de Dados Meteorológicos</p>
    <p style="font-size:0.85rem;">Licença MIT © 2024</p>
    <p style="font-size:0.8rem; color:#adb5bd;">
        Pearson (1895) | Tukey (1977) | Pimentel-Gomes (2000)
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
