# ============================================================
# AgroDataLab - Sistema Completo de Análise Meteorológica
# Versão 5.0 - TODAS AS CORREÇÕES APLICADAS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
import io
import traceback
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
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None
if 'df_processed' not in st.session_state:
    st.session_state['df_processed'] = None
if 'date_columns' not in st.session_state:
    st.session_state['date_columns'] = []

# ============================================================
# FUNÇÃO CRÍTICA: CORRIGIR NOMES DUPLICADOS
# ============================================================
def corrigir_nomes_duplicados(df):
    """
    Renomeia colunas duplicadas adicionando sufixo _1, _2, etc.
    Exemplo: Vento, Vento, Vento -> Vento, Vento_1, Vento_2
    """
    cols = pd.Series(df.columns)
    
    for nome in cols[cols.duplicated()].unique():
        idx = cols[cols == nome].index.tolist()
        for i, pos in enumerate(idx):
            if i > 0:
                cols[pos] = f"{nome}_{i}"
    
    df.columns = cols
    return df

# ============================================================
# FUNÇÕES DE PROCESSAMENTO
# ============================================================

def carregar_dados(arquivo):
    """Carrega CSV ou Excel com detecção automática"""
    try:
        if arquivo.name.endswith('.csv'):
            for enc in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(arquivo, encoding=enc)
                    if df.shape[1] == 1:
                        arquivo.seek(0)
                        for sep in [';', '\t', '|']:
                            try:
                                df = pd.read_csv(arquivo, encoding=enc, sep=sep)
                                if df.shape[1] > 1:
                                    # CORREÇÃO 2: Já corrigir duplicados no carregamento
                                    return corrigir_nomes_duplicados(df)
                            except:
                                continue
                    if df.shape[1] > 1:
                        return corrigir_nomes_duplicados(df)
                except:
                    continue
            arquivo.seek(0)
            df = pd.read_csv(arquivo)
            return corrigir_nomes_duplicados(df)
        else:
            df = pd.read_excel(arquivo)
            return corrigir_nomes_duplicados(df)
    except Exception as e:
        st.error(f"Erro ao carregar: {str(e)}")
        st.code(traceback.format_exc())
        return None

def processar_dados(df):
    """
    Processa dados automaticamente
    CORREÇÃO 1: Não converte colunas categóricas para NaN
    CORREÇÃO 5: Padrões de data expandidos
    """
    df = df.copy()
    date_cols = []
    
    # CORREÇÃO 5: Padrões de data expandidos
    padroes_data = [
        'data', 'hora', 'date', 'time', 'ano', 'mes', 'mês', 'dia',
        'timestamp', 'datetime', 'datahora', 'data_coleta', 'month', 'year', 'day'
    ]
    
    for col in df.columns:
        col_lower = col.lower().replace(' ', '_')
        if any(p in col_lower for p in padroes_data):
            date_cols.append(col)
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    
    # CORREÇÃO 1: Converter apenas colunas verdadeiramente numéricas
    for col in df.columns:
        if col not in date_cols:
            if df[col].dtype == "object":
                # Testar se realmente é numérica usando amostra
                amostra = (
                    df[col]
                    .dropna()
                    .astype(str)
                    .str.replace(",", ".")
                    .head(20)
                )
                
                try:
                    pd.to_numeric(amostra)
                    # É numérica, converter
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(",", ".")
                    )
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except:
                    # Não é numérica, manter como está (categórica)
                    pass
            elif df[col].dtype in ['int64', 'float64']:
                # Já é numérica, garantir
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df, date_cols

def preencher_ausentes(df, metodo='media'):
    """Preenche valores ausentes"""
    df = df.copy()
    cols_num = df.select_dtypes(include=[np.number]).columns
    
    if metodo == 'media':
        df[cols_num] = df[cols_num].fillna(df[cols_num].mean())
    elif metodo == 'mediana':
        df[cols_num] = df[cols_num].fillna(df[cols_num].median())
    elif metodo == 'interpolar':
        df[cols_num] = df[cols_num].interpolate(method='linear', limit_direction='both')
    
    return df

# ============================================================
# FUNÇÕES ESTATÍSTICAS
# ============================================================

def calcular_estatisticas(df, coluna):
    """
    Calcula estatísticas descritivas completas
    CORREÇÃO 3: Verifica se há dados suficientes
    """
    dados = df[coluna].dropna()
    
    # CORREÇÃO 3: Verificar quantidade mínima de dados
    if len(dados) < 2:
        st.warning(f"A variável **{coluna}** possui poucos dados válidos ({len(dados)} valores).")
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
    assimetria = np.sum((dados - media) ** 3) / (n * desvio ** 3) if desvio > 0 else 0
    curtose = np.sum((dados - media) ** 4) / (n * desvio ** 4) - 3 if desvio > 0 else 0
    
    return {
        'Amostras': n, 'Média': round(media, 3), 'Mediana': round(mediana, 3),
        'Desvio Padrão': round(desvio, 3), 'Variância': round(variancia, 3),
        'Mínimo': round(minimo, 3), 'Máximo': round(maximo, 3),
        'Q1': round(q1, 3), 'Q3': round(q3, 3), 'IQR': round(iqr, 3),
        'CV (%)': round(cv, 2), 'Assimetria': round(assimetria, 3),
        'Curtose': round(curtose, 3)
    }

def classificar_cv(cv):
    if cv <= 10: return "Baixo 🟢"
    elif cv <= 20: return "Médio 🟡"
    elif cv <= 30: return "Alto 🟠"
    else: return "Muito Alto 🔴"

def detectar_outliers(dados):
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    outliers = (dados < lim_inf) | (dados > lim_sup)
    return outliers, lim_inf, lim_sup

# ============================================================
# FUNÇÕES DE GRÁFICOS (COM TRATAMENTO DE ERROS)
# ============================================================

def criar_grafico_linhas(df, x_col, y_cols, titulo="Série Temporal"):
    """Gráfico de linhas com tratamento de erros detalhado"""
    try:
        # Garantir que não há duplicatas
        df = corrigir_nomes_duplicados(df)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Verificar colunas
        y_cols_validas = [c for c in y_cols if c in df.columns]
        if not y_cols_validas:
            st.warning("Nenhuma variável Y disponível para o gráfico.")
            return None
        
        # Melt
        df_melted = pd.melt(
            df,
            id_vars=[x_col],
            value_vars=y_cols_validas,
            var_name='Variavel',
            value_name='Valor'
        )
        
        # CORREÇÃO 2: Garantir nomes únicos
        df_melted = corrigir_nomes_duplicados(df_melted)
        df_melted = df_melted.loc[:, ~df_melted.columns.duplicated()]
        
        chart = alt.Chart(df_melted).mark_line(strokeWidth=2).encode(
            x=alt.X(f'{x_col}:T', title='Data'),
            y=alt.Y('Valor:Q', title='Valor'),
            color=alt.Color('Variavel:N', legend=alt.Legend(orient='bottom')),
            tooltip=[f'{x_col}:T', 'Variavel:N', alt.Tooltip('Valor:Q', format='.2f')]
        ).properties(title=titulo, height=400).interactive()
        
        return chart
    
    except Exception as e:
        # CORREÇÃO 6: Mostrar erro detalhado
        st.error(f"Erro ao criar gráfico de linhas: {str(e)}")
        st.code(traceback.format_exc())
        return None

def criar_histograma(df, coluna, bins=30):
    """Histograma com tratamento de erros"""
    try:
        df = corrigir_nomes_duplicados(df)
        df = df.loc[:, ~df.columns.duplicated()]
        
        if coluna not in df.columns:
            st.error(f"Coluna '{coluna}' não encontrada.")
            return None
        
        df_clean = df[[coluna]].dropna()
        
        if len(df_clean) < 2:
            st.warning(f"Poucos dados para histograma de {coluna}")
            return None
        
        chart = alt.Chart(df_clean).mark_bar(opacity=0.7, color='#2d8a4e').encode(
            alt.X(f'{coluna}:Q', bin=alt.Bin(maxbins=bins), title=coluna),
            alt.Y('count()', title='Frequência'),
            tooltip=['count()']
        ).properties(title=f'Distribuição de {coluna}', height=400)
        
        return chart
    
    except Exception as e:
        st.error(f"Erro ao criar histograma: {str(e)}")
        st.code(traceback.format_exc())
        return None

def criar_boxplot(df, coluna):
    """Boxplot com tratamento de erros"""
    try:
        df = corrigir_nomes_duplicados(df)
        df = df.loc[:, ~df.columns.duplicated()]
        
        if coluna not in df.columns:
            st.error(f"Coluna '{coluna}' não encontrada.")
            return None
        
        df_clean = df[[coluna]].dropna()
        
        if len(df_clean) < 5:
            st.warning(f"Poucos dados para boxplot de {coluna}")
            return None
        
        chart = alt.Chart(df_clean).mark_boxplot(extent='min-max', color='#2d8a4e').encode(
            y=alt.Y(f'{coluna}:Q', title=coluna)
        ).properties(title=f'Boxplot de {coluna}', height=400)
        
        return chart
    
    except Exception as e:
        st.error(f"Erro ao criar boxplot: {str(e)}")
        st.code(traceback.format_exc())
        return None

# ============================================================
# FUNÇÃO AUXILIAR PARA OBTER DATAFRAME LIMPO
# ============================================================
def obter_df_limpo():
    """
    CORREÇÃO 7: Obtém DataFrame processado e aplica todas as correções
    """
    if st.session_state['df_processed'] is None:
        return None
    
    df = st.session_state['df_processed'].copy()
    df = corrigir_nomes_duplicados(df)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero-header">
    <h1>🌱 AgroDataLab</h1>
    <p>Sistema Inteligente de Análise de Dados Meteorológicos</p>
    <p style="font-size:0.9rem; opacity:0.8;">v5.0 - Todas as correções aplicadas</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📊 Status")
    
    df_status = obter_df_limpo()
    
    if df_status is not None:
        st.success(f"✅ {len(df_status):,} registros")
        st.caption(f"{len(df_status.columns)} colunas")
        
        aus = df_status.isnull().sum().sum()
        if aus > 0:
            st.warning(f"⚠️ {aus} ausentes")
        else:
            st.success("✅ Dados completos")
        
        # Mostrar tipos de colunas
        num_cols = len(df_status.select_dtypes(include=[np.number]).columns)
        cat_cols = len(df_status.select_dtypes(include=['object']).columns)
        date_cols = len(st.session_state.get('date_columns', []))
        
        st.caption(f"📊 {num_cols} numéricas | 📝 {cat_cols} categóricas | 📅 {date_cols} datas")
    else:
        st.info("📤 Sem dados")
    
    st.markdown("---")
    st.markdown("### 🔧 Correções Aplicadas")
    st.caption("""
    ✅ Colunas duplicadas renomeadas  
    ✅ Colunas categóricas preservadas  
    ✅ Padrões de data expandidos  
    ✅ Tratamento de erros detalhado  
    ✅ Validação de dados mínimos
    """)
    
    st.markdown("---")
    st.caption("AgroDataLab v5.0 | MIT License")

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
    
    arquivo = st.file_uploader(
        "Selecione CSV ou Excel",
        type=['csv', 'xlsx', 'xls'],
        help="Formatos: CSV (qualquer separador) e Excel (.xlsx, .xls)"
    )
    
    if arquivo is not None:
        with st.spinner('🔄 Carregando e processando...'):
            df = carregar_dados(arquivo)
            
            if df is not None and len(df) > 0:
                # CORREÇÃO 2: Já foi aplicada no carregar_dados()
                # Processar dados (CORREÇÃO 1 e 5 aplicadas)
                df, date_cols = processar_dados(df)
                
                # CORREÇÃO 2: Garantir novamente
                df = corrigir_nomes_duplicados(df)
                df = df.loc[:, ~df.columns.duplicated()]
                
                st.session_state['df_original'] = df.copy()
                st.session_state['df_processed'] = df.copy()
                st.session_state['date_columns'] = date_cols
                
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
                    cat_cols = len(df.select_dtypes(include=['object']).columns)
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Categóricas</p>
                        <p class="metric-value">{cat_cols}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Preview
                st.markdown("---")
                st.subheader("👀 Preview dos Dados")
                st.dataframe(df.head(15), use_container_width=True)
                
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
                
                if date_cols:
                    st.success(f"📅 **{len(date_cols)}** colunas de data: {', '.join(date_cols)}")
                else:
                    st.info("ℹ️ Nenhuma coluna de data identificada. Gráficos temporais não disponíveis.")
            else:
                st.error("❌ Não foi possível carregar o arquivo.")
    else:
        st.markdown("""
        <div class="info-box">
            <h4>📤 Instruções:</h4>
            <ol>
                <li>Clique em <strong>"Browse files"</strong></li>
                <li>Selecione arquivo <strong>CSV</strong> ou <strong>Excel</strong></li>
                <li>Detecção automática de formato e separadores</li>
                <li>Colunas duplicadas são renomeadas automaticamente</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# ABA 2: TRATAMENTO
# ============================================================
with aba2:
    st.header("🔧 Tratamento de Dados")
    st.markdown("---")
    
    # CORREÇÃO 7: Usar função auxiliar
    df = obter_df_limpo()
    
    if df is None:
        st.warning("⚠️ Carregue os dados na Aba 1 primeiro!")
    else:
        missing_total = df.isnull().sum().sum()
        
        if missing_total > 0:
            st.markdown(f"""
            <div class="warning-box">
                <h3>⚠️ {missing_total:,} valores ausentes em {df.isnull().any().sum()} colunas</h3>
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
            
            # Métodos
            st.markdown("---")
            st.subheader("🛠️ Métodos de Preenchimento")
            
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown("**Média Aritmética**")
                st.caption("Indicado para dados com distribuição normal")
                if st.button("📊 Aplicar Média", key="btn_media"):
                    st.session_state['df_processed'] = preencher_ausentes(df, 'media')
                    st.success("✅ Preenchido com média!")
                    st.rerun()
            
            with c2:
                st.markdown("**Mediana**")
                st.caption("Robusto a outliers")
                if st.button("📈 Aplicar Mediana", key="btn_mediana"):
                    st.session_state['df_processed'] = preencher_ausentes(df, 'mediana')
                    st.success("✅ Preenchido com mediana!")
                    st.rerun()
            
            with c3:
                st.markdown("**Interpolação Linear**")
                st.caption("Ideal para séries temporais")
                if st.button("🔄 Interpolar", key="btn_interp"):
                    st.session_state['df_processed'] = preencher_ausentes(df, 'interpolar')
                    st.success("✅ Interpolado!")
                    st.rerun()
            
            c4, c5 = st.columns(2)
            
            with c4:
                st.markdown("**Remover Linhas**")
                st.caption("Exclui linhas com qualquer ausente")
                if st.button("🗑️ Remover Incompletas", key="btn_drop"):
                    antes = len(df)
                    df_limpo = df.dropna()
                    df_limpo = corrigir_nomes_duplicados(df_limpo)
                    st.session_state['df_processed'] = df_limpo
                    st.success(f"✅ {antes - len(df_limpo)} linhas removidas!")
                    st.rerun()
            
            with c5:
                st.markdown("**Restaurar**")
                st.caption("Volta aos dados originais")
                if st.button("🔙 Restaurar Original", key="btn_restore"):
                    if st.session_state['df_original'] is not None:
                        st.session_state['df_processed'] = st.session_state['df_original'].copy()
                        st.success("✅ Dados restaurados!")
                        st.rerun()
        else:
            st.markdown('<div class="success-box"><h3>✅ Nenhum valor ausente!</h3></div>', unsafe_allow_html=True)
        
        # Outliers
        st.markdown("---")
        st.subheader("🔍 Detecção de Outliers (Método IQR)")
        st.caption("Tukey (1977): Limites = Q1 ± 1.5 × IQR")
        
        colunas_num = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if colunas_num:
            col_out = st.selectbox("Selecione a variável:", colunas_num, key="outlier_col")
            
            dados_limpos = df[col_out].dropna()
            
            if len(dados_limpos) >= 2:
                outliers, lim_inf, lim_sup = detectar_outliers(dados_limpos)
                n_out = outliers.sum()
                
                if n_out > 0:
                    pct = n_out / len(dados_limpos) * 100
                    st.warning(f"🔴 **{n_out}** outliers detectados ({pct:.1f}% dos dados)")
                    
                    # Boxplot
                    df_plot = pd.DataFrame({col_out: dados_limpos.values})
                    chart = criar_boxplot(df_plot, col_out)
                    if chart:
                        st.altair_chart(chart, use_container_width=True)
                    
                    col_a, col_b = st.columns(2)
                    col_a.metric("Limite Inferior", f"{lim_inf:.3f}")
                    col_b.metric("Limite Superior", f"{lim_sup:.3f}")
                    
                    # Mostrar outliers
                    with st.expander("👀 Ver valores outliers"):
                        st.dataframe(df[outliers][[col_out]].head(50), use_container_width=True)
                else:
                    st.success(f"✅ Nenhum outlier detectado em **{col_out}**")
            else:
                st.warning(f"Poucos dados válidos em {col_out}")
        else:
            st.info("ℹ️ Nenhuma variável numérica disponível para análise de outliers.")

# ============================================================
# ABA 3: ESTATÍSTICAS
# ============================================================
with aba3:
    st.header("📊 Análise Estatística")
    st.markdown("---")
    
    # CORREÇÃO 7: Usar função auxiliar
    df = obter_df_limpo()
    
    if df is None:
        st.warning("⚠️ Carregue os dados na Aba 1 primeiro!")
    else:
        colunas_num = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not colunas_num:
            st.warning("⚠️ Nenhuma variável numérica encontrada.")
        else:
            col_analise = st.selectbox(
                "Selecione a variável para análise:",
                colunas_num,
                key="stat_col"
            )
            
            # CORREÇÃO 3: A função já verifica dados mínimos
            stats = calcular_estatisticas(df, col_analise)
            
            if stats:
                st.subheader(f"📈 Estatísticas: **{col_analise}**")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Média</p>
                        <p class="metric-value">{stats['Média']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Desvio Padrão</p>
                        <p class="metric-value">{stats['Desvio Padrão']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    cv_val = stats['CV (%)']
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">CV (%)</p>
                        <p class="metric-value">{cv_val:.1f}%</p>
                        <p style="font-size:0.8rem;">{classificar_cv(cv_val)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Amostras</p>
                        <p class="metric-value">{stats['Amostras']:,}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Tabela completa
                with st.expander("📋 Ver Todas as Estatísticas"):
                    stats_df = pd.DataFrame(list(stats.items()), columns=['Estatística', 'Valor'])
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Histograma
                st.markdown("---")
                st.subheader("📊 Distribuição")
                
                bins = st.slider("Número de intervalos:", 10, 100, 30, key="hist_bins_stats")
                df_plot = pd.DataFrame({col_analise: df[col_analise].dropna().values})
                chart_hist = criar_histograma(df_plot, col_analise, bins)
                if chart_hist:
                    st.altair_chart(chart_hist, use_container_width=True)
                
                # Matriz de Correlação
                if len(colunas_num) >= 2:
                    st.markdown("---")
                    st.subheader("🔗 Matriz de Correlação de Pearson")
                    
                    # CORREÇÃO 4: Remover colunas totalmente vazias
                    corr = (
                        df[colunas_num]
                        .dropna(axis=1, how="all")
                        .corr()
                    )
                    
                    if not corr.empty:
                        st.dataframe(
                            corr.style.background_gradient(
                                cmap='RdYlGn',
                                vmin=-1,
                                vmax=1
                            ).format("{:.3f}"),
                            use_container_width=True
                        )
                        
                        st.caption("""
                        **Interpretação:**  
                        🟢 **0.7 a 1.0** → Forte  
                        🟡 **0.3 a 0.7** → Moderada  
                        🔴 **0.0 a 0.3** → Fraca
                        """)
                    else:
                        st.warning("Não foi possível calcular correlações (dados insuficientes).")
                
                # Análise Temporal
                date_cols = st.session_state.get('date_columns', [])
                
                if date_cols:
                    st.markdown("---")
                    st.subheader("📅 Análise Temporal")
                    
                    col_data = st.selectbox("Coluna de data:", date_cols, key="date_stat_col")
                    
                    df_plot = df[[col_data, col_analise]].dropna().copy()
                    df_plot = corrigir_nomes_duplicados(df_plot)
                    df_plot[col_data] = pd.to_datetime(df_plot[col_data], errors='coerce')
                    df_plot = df_plot.dropna()
                    
                    if len(df_plot) > 0:
                        chart_temp = criar_grafico_linhas(
                            df_plot,
                            col_data,
                            [col_analise],
                            f'Série Temporal - {col_analise}'
                        )
                        if chart_temp:
                            st.altair_chart(chart_temp, use_container_width=True)
                    else:
                        st.warning("Dados insuficientes para série temporal.")

# ============================================================
# ABA 4: GRÁFICOS
# ============================================================
with aba4:
    st.header("📈 Visualização Gráfica Interativa")
    st.markdown("---")
    
    # CORREÇÃO 7: Usar função auxiliar
    df = obter_df_limpo()
    
    if df is None:
        st.warning("⚠️ Carregue os dados na Aba 1 primeiro!")
    else:
        colunas_num = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not colunas_num:
            st.warning("⚠️ Nenhuma variável numérica disponível para gráficos.")
        else:
            tipo = st.radio(
                "Selecione o tipo de gráfico:",
                ['📈 Linhas (Série Temporal)', '📊 Histograma', '📉 Boxplot'],
                horizontal=True,
                key="graf_tipo"
            )
            
            st.markdown("---")
            
            if tipo == '📈 Linhas (Série Temporal)':
                date_cols = st.session_state.get('date_columns', [])
                
                if date_cols:
                    col_x = st.selectbox("Eixo X (Data):", date_cols, key="g_x")
                    col_y = st.multiselect(
                        "Variáveis Y:",
                        colunas_num,
                        default=colunas_num[:min(3, len(colunas_num))],
                        key="g_y"
                    )
                    
                    if col_y:
                        df_plot = df[[col_x] + col_y].dropna().copy()
                        df_plot = corrigir_nomes_duplicados(df_plot)
                        df_plot[col_x] = pd.to_datetime(df_plot[col_x], errors='coerce')
                        df_plot = df_plot.dropna()
                        
                        if len(df_plot) > 0:
                            chart = criar_grafico_linhas(df_plot, col_x, col_y, "Série Temporal")
                            if chart:
                                st.altair_chart(chart, use_container_width=True)
                        else:
                            st.warning("Dados insuficientes após remover nulos.")
                    else:
                        st.info("Selecione pelo menos uma variável para o eixo Y.")
                else:
                    st.info("""
                    <div class="info-box">
                        <h4>📅 Coluna de data não encontrada</h4>
                        <p>Gráficos de linha precisam de uma coluna de data. Use Histograma ou Boxplot.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            elif tipo == '📊 Histograma':
                col_h = st.selectbox("Variável:", colunas_num, key="g_hist")
                bins_h = st.slider("Intervalos:", 5, 100, 30, key="g_bins")
                
                df_plot = pd.DataFrame({col_h: df[col_h].dropna().values})
                chart = criar_histograma(df_plot, col_h, bins_h)
                if chart:
                    st.altair_chart(chart, use_container_width=True)
            
            elif tipo == '📉 Boxplot':
                col_b = st.selectbox("Variável:", colunas_num, key="g_box")
                
                df_plot = pd.DataFrame({col_b: df[col_b].dropna().values})
                chart = criar_boxplot(df_plot, col_b)
                if chart:
                    st.altair_chart(chart, use_container_width=True)

# ============================================================
# ABA 5: DOWNLOAD
# ============================================================
with aba5:
    st.header("💾 Download dos Dados Processados")
    st.markdown("---")
    
    # CORREÇÃO 7: Usar função auxiliar
    df = obter_df_limpo()
    
    if df is None:
        st.warning("⚠️ Carregue e processe os dados primeiro!")
    else:
        st.markdown('<div class="success-box"><h3>✅ Dados Prontos para Download</h3></div>', unsafe_allow_html=True)
        
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
        st.subheader("📥 Exportar CSV")
        
        csv_data = df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="📥 Baixar arquivo CSV",
            data=csv_data,
            file_name=f"dados_processados_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="download_csv_final"
        )
        
        # Preview
        st.markdown("---")
        st.subheader("👀 Preview dos Dados Processados")
        
        n_prev = st.slider("Linhas:", 5, 100, 20, key="prev_download")
        st.dataframe(df.head(n_prev), use_container_width=True)
        
        # Resumo do processamento
        if st.session_state['df_original'] is not None:
            st.markdown("---")
            st.subheader("📊 Resumo do Processamento")
            
            df_orig = st.session_state['df_original']
            
            resumo = pd.DataFrame({
                'Métrica': [
                    'Registros Originais',
                    'Registros Finais',
                    'Colunas Originais',
                    'Colunas Finais',
                    'Ausentes Originais',
                    'Ausentes Finais',
                    'Ausentes Tratados'
                ],
                'Valor': [
                    f"{len(df_orig):,}",
                    f"{len(df):,}",
                    len(df_orig.columns),
                    len(df.columns),
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
<div style="text-align:center; padding:1.5rem; color:#6c757d; background:#f8f9fa; border-radius:12px;">
    <h4>🌱 AgroDataLab v5.0</h4>
    <p>Sistema de Análise de Dados Meteorológicos</p>
    <p style="font-size:0.85rem;">
        Desenvolvido com Streamlit, Pandas, NumPy e Altair<br>
        Métodos: Pearson (1895) | Tukey (1977) | Pimentel-Gomes (2000)
    </p>
    <p style="font-size:0.8rem;">Licença MIT © 2024</p>
</div>
""", unsafe_allow_html=True)
