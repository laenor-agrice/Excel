# ============================================================
# AgroDataLab - Sistema Completo de Análise Meteorológica
# Versão 7.0 - Detecção Robusta de Delimitador e Datas
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
if 'date_columns' not in st.session_state:
    st.session_state['date_columns'] = []

# ============================================================
# FUNÇÃO: CORRIGIR NOMES DUPLICADOS
# ============================================================
def corrigir_nomes_duplicados(df):
    """Renomeia colunas duplicadas: Vento, Vento, Vento -> Vento, Vento_1, Vento_2"""
    cols = pd.Series(df.columns)
    
    for nome in cols[cols.duplicated()].unique():
        idx = cols[cols == nome].index.tolist()
        for i, pos in enumerate(idx):
            if i > 0:
                cols[pos] = f"{nome}_{i}"
    
    df.columns = cols
    return df

# ============================================================
# FUNÇÃO: DETECTAR DELIMITADOR (NOVA - ROBUSTA)
# ============================================================
def detectar_delimitador(arquivo):
    """
    Detecta automaticamente o delimitador do CSV
    Usa csv.Sniffer + fallback para delimitadores comuns
    """
    try:
        # Ler amostra para análise
        arquivo.seek(0)
        amostra_bytes = arquivo.read(8192)
        arquivo.seek(0)
        
        # Tentar decodificar
        for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
            try:
                amostra = amostra_bytes.decode(encoding)
                break
            except:
                continue
        else:
            amostra = amostra_bytes.decode('utf-8', errors='ignore')
        
        # Método 1: csv.Sniffer
        try:
            sniffer = csv.Sniffer()
            dialecto = sniffer.sniff(amostra)
            delimitador = dialecto.delimiter
            
            # Validar
            linhas = amostra.strip().split('\n')
            if len(linhas) >= 2:
                c1 = linhas[0].count(delimitador)
                c2 = linhas[1].count(delimitador)
                if c1 == c2 and c1 > 0:
                    return delimitador
        except:
            pass
        
        # Método 2: Fallback - testar delimitadores comuns
        delimitadores = [';', ',', '\t', '|']
        linhas = amostra.strip().split('\n')
        
        if len(linhas) >= 2:
            for sep in delimitadores:
                c1 = linhas[0].count(sep)
                c2 = linhas[1].count(sep)
                if c1 == c2 and c1 > 0:
                    return sep
        
        # Método 3: Qual delimitador aparece mais
        melhor_sep = ','
        max_count = 0
        for sep in delimitadores:
            count = amostra.count(sep)
            if count > max_count:
                max_count = count
                melhor_sep = sep
        
        return melhor_sep if max_count > 0 else ','
    
    except:
        return ','


# ============================================================
# FUNÇÃO: CARREGAR DADOS (CORRIGIDA)
# ============================================================
def carregar_dados(arquivo):
    """
    Carrega CSV ou Excel com:
    - Detecção automática de delimitador
    - Detecção de encoding
    - Tratamento de decimal (vírgula/ponto)
    """
    try:
        if arquivo.name.endswith('.csv'):
            
            # Detectar delimitador
            separador = detectar_delimitador(arquivo)
            
            # Tentar diferentes configurações
            configuracoes = [
                # (encoding, decimal)
                ('utf-8', ','),
                ('utf-8', '.'),
                ('latin1', ','),
                ('latin1', '.'),
                ('iso-8859-1', ','),
                ('iso-8859-1', '.'),
                ('cp1252', ','),
                ('cp1252', '.'),
            ]
            
            for encoding, decimal in configuracoes:
                try:
                    arquivo.seek(0)
                    df = pd.read_csv(
                        arquivo,
                        sep=separador,
                        encoding=encoding,
                        decimal=decimal,
                        thousands='.',
                        on_bad_lines='skip',
                        engine='python'
                    )
                    
                    # Verificar se leu corretamente (mais de 1 coluna)
                    if df.shape[1] > 1:
                        # Verificar se colunas numéricas foram lidas
                        num_cols = df.select_dtypes(include=[np.number]).columns
                        obj_cols = df.select_dtypes(include=['object']).columns
                        
                        # Se tem colunas numéricas, ótimo
                        if len(num_cols) > 0:
                            return corrigir_nomes_duplicados(df)
                        
                        # Se todas são objeto, pode ser que o decimal esteja errado
                        # Tentar converter vírgula para ponto nas colunas objeto
                        if len(obj_cols) > 0 and len(num_cols) == 0:
                            # Verificar se parece número com vírgula
                            for col in obj_cols[:3]:  # testar primeiras 3 colunas
                                amostra = df[col].dropna().head(5).astype(str)
                                if amostra.str.contains(',').any():
                                    # Provavelmente é número com vírgula decimal
                                    arquivo.seek(0)
                                    df = pd.read_csv(
                                        arquivo,
                                        sep=separador,
                                        encoding=encoding,
                                        decimal=',',
                                        thousands='.',
                                        on_bad_lines='skip',
                                        engine='python'
                                    )
                                    return corrigir_nomes_duplicados(df)
                        
                        return corrigir_nomes_duplicados(df)
                
                except Exception as e:
                    continue
            
            # Última tentativa: deixar pandas decidir tudo
            arquivo.seek(0)
            df = pd.read_csv(
                arquivo,
                sep=separador,
                encoding='utf-8',
                on_bad_lines='skip',
                engine='python'
            )
            return corrigir_nomes_duplicados(df)
        
        else:
            # Excel
            df = pd.read_excel(arquivo)
            return corrigir_nomes_duplicados(df)
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo:")
        st.code(f"{type(e).__name__}: {str(e)}")
        st.code(traceback.format_exc())
        return None

# ============================================================
# FUNÇÃO PRINCIPAL: PROCESSAR DADOS COM DETECÇÃO INTELIGENTE
# ============================================================
def processar_dados(df):
    """
    Processa dados automaticamente
    DETECÇÃO INTELIGENTE DE DATAS: analisa nome E conteúdo
    """
    df = df.copy()
    date_cols = []

    # Padrões de nome de coluna (EXPANDIDO)
    padroes_data = [
        'data', 'hora', 'date', 'time',
        'ano', 'mes', 'mês', 'dia',
        'timestamp', 'datetime',
        'datahora', 'data_coleta',
        'month', 'year', 'day',
        'período', 'periodo', 'safra',
        'tempo', 'juliano'
    ]

    # ============================================================
    # ETAPA 1: DETECÇÃO DE DATAS (NOME + CONTEÚDO)
    # ============================================================
    for col in df.columns:
        col_lower = str(col).lower().replace(" ", "_")
        eh_data = False

        # Critério 1: Nome da coluna contém padrão de data
        if any(p in col_lower for p in padroes_data):
            eh_data = True
        
        # Critério 2: Análise do conteúdo da coluna
        else:
            amostra = df[col].dropna().head(20)
            
            if len(amostra) > 0:
                try:
                    # Tentar converter para datetime
                    teste = pd.to_datetime(
                        amostra,
                        errors="coerce",
                        dayfirst=True  # Formato brasileiro: dd/mm/aaaa
                    )
                    
                    taxa_sucesso = teste.notna().mean()
                    
                    # Se 70% ou mais converteram, é data
                    if taxa_sucesso >= 0.70:
                        eh_data = True
                except:
                    pass
        
        # Se identificou como data, converter
        if eh_data:
            try:
                df[col] = pd.to_datetime(
                    df[col],
                    errors="coerce",
                    dayfirst=True
                )
                
                # Só adiciona se realmente converteu algo
                if df[col].notna().sum() > 0:
                    date_cols.append(col)
            except:
                pass

    # ============================================================
    # ETAPA 2: CONVERSÃO NUMÉRICA (PRESERVANDO CATEGÓRICAS)
    # ============================================================
    for col in df.columns:
        if col not in date_cols:
            if df[col].dtype == "object":
                # Testar se é realmente numérica
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
                    # Não é numérica, manter como categórica
                    pass
            elif df[col].dtype in ['int64', 'float64']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    return df, date_cols

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
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

def calcular_estatisticas(df, coluna):
    dados = df[coluna].dropna()
    
    if len(dados) < 2:
        st.warning(f"A variável **{coluna}** possui apenas {len(dados)} valor(es) válido(s).")
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

def obter_df_limpo():
    """Obtém DataFrame processado com todas as correções"""
    if st.session_state['df_processed'] is None:
        return None
    
    df = st.session_state['df_processed'].copy()
    df = corrigir_nomes_duplicados(df)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

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
        
        df_melted = pd.melt(
            df, id_vars=[x_col], value_vars=y_cols_validas,
            var_name='Variavel', value_name='Valor'
        )
        
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
        st.error(f"Erro no gráfico: {str(e)}")
        st.code(traceback.format_exc())
        return None

def criar_histograma(df, coluna, bins=30):
    try:
        df = corrigir_nomes_duplicados(df)
        df = df.loc[:, ~df.columns.duplicated()]
        
        if coluna not in df.columns:
            return None
        
        df_clean = df[[coluna]].dropna()
        if len(df_clean) < 2:
            return None
        
        chart = alt.Chart(df_clean).mark_bar(opacity=0.7, color='#2d8a4e').encode(
            alt.X(f'{coluna}:Q', bin=alt.Bin(maxbins=bins), title=coluna),
            alt.Y('count()', title='Frequência'),
            tooltip=['count()']
        ).properties(title=f'Distribuição de {coluna}', height=400)
        
        return chart
    except Exception as e:
        st.error(f"Erro: {str(e)}")
        st.code(traceback.format_exc())
        return None

def criar_boxplot(df, coluna):
    try:
        df = corrigir_nomes_duplicados(df)
        df = df.loc[:, ~df.columns.duplicated()]
        
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
        st.code(traceback.format_exc())
        return None

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero-header">
    <h1>🌱 AgroDataLab</h1>
    <p>Sistema Inteligente de Análise de Dados Meteorológicos</p>
    <p style="font-size:0.9rem; opacity:0.8;">v7.0 - Detecção Robusta de CSV e Datas</p>
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
        
        num_cols = len(df_status.select_dtypes(include=[np.number]).columns)
        cat_cols = len(df_status.select_dtypes(include=['object']).columns)
        date_cols = len(st.session_state.get('date_columns', []))
        
        st.caption(f"📊 {num_cols} numéricas | 📝 {cat_cols} categóricas | 📅 {date_cols} datas")
        
        if date_cols > 0:
            with st.expander("📅 Datas detectadas"):
                for dc in st.session_state['date_columns']:
                    st.caption(f"• {dc}")
    else:
        st.info("📤 Sem dados")
    
    st.markdown("---")
    st.caption("AgroDataLab v7.0 | MIT License")

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
        with st.spinner('🔄 Carregando e processando...'):
            df = carregar_dados(arquivo)
            
            if df is not None and len(df) > 0:
                
                # DIAGNÓSTICO DE LEITURA
                if df.shape[1] == 1:
                    st.error("""
                    ⚠️ **Arquivo lido como UMA ÚNICA coluna!**
                    
                    O delimitador não foi identificado corretamente.
                    
                    **Solução:**
                    1. Abra o arquivo no Excel
                    2. Salve como CSV (separado por vírgulas ou ponto-e-vírgula)
                    3. Faça upload novamente
                    """)
                    st.stop()
                
                # Processar com detecção inteligente de datas
                df, date_cols = processar_dados(df)
                
                # Garantir nomes únicos
                df = corrigir_nomes_duplicados(df)
                df = df.loc[:, ~df.columns.duplicated()]
                
                st.session_state['df_original'] = df.copy()
                st.session_state['df_processed'] = df.copy()
                st.session_state['date_columns'] = date_cols
                
                st.markdown(f"""
                <div class="success-box">
                    <h3>✅ Arquivo carregado!</h3>
                    <p><strong>{df.shape[0]:,}</strong> linhas × <strong>{df.shape[1]}</strong> colunas</p>
                </div>
                """, unsafe_allow_html=True)
                
                # DIAGNÓSTICO DE DATAS
                if date_cols:
                    st.success(f"📅 **{len(date_cols)}** colunas de data detectadas:")
                    for dc in date_cols:
                        st.caption(f"• {dc} ({df[dc].dtype})")
                else:
                    st.warning("""
                    ⚠️ **Nenhuma coluna de data foi detectada.**
                    
                    Os gráficos temporais ficarão indisponíveis.
                    """)
                
                # Métricas
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("📋 Registros", f"{len(df):,}")
                col2.metric("📊 Colunas", len(df.columns))
                
                num_cols = len(df.select_dtypes(include=[np.number]).columns)
                cat_cols = len(df.select_dtypes(include=['object']).columns)
                col3.metric("🔢 Numéricas", num_cols)
                col4.metric("📅 Datas", len(date_cols))
                
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
            else:
                st.error("❌ Não foi possível carregar o arquivo. Verifique o formato.")
    else:
        st.markdown("""
        <div class="info-box">
            <h4>📤 Instruções:</h4>
            <ol>
                <li>Clique em <strong>"Browse files"</strong></li>
                <li>Selecione um arquivo <strong>CSV</strong> ou <strong>Excel</strong></li>
                <li>O sistema detecta automaticamente:
                    <ul>
                        <li>Delimitador ( ; , tab | )</li>
                        <li>Encoding (UTF-8, Latin1, etc.)</li>
                        <li>Decimal (vírgula ou ponto)</li>
                        <li>Colunas de data (nome e conteúdo)</li>
                    </ul>
                </li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

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
            
            st.markdown("---")
            st.subheader("🛠️ Preencher Ausentes")
            
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
                    df_limpo = df.dropna()
                    st.session_state['df_processed'] = df_limpo
                    st.success(f"✅ {antes - len(df_limpo)} removidas!")
                    st.rerun()
            
            with c5:
                if st.button("🔙 Restaurar", key="m5"):
                    if st.session_state['df_original'] is not None:
                        st.session_state['df_processed'] = st.session_state['df_original'].copy()
                        st.success("✅ Restaurado!")
                        st.rerun()
        else:
            st.markdown('<div class="success-box"><h3>✅ Nenhum valor ausente!</h3></div>', unsafe_allow_html=True)
        
        # Outliers
        st.markdown("---")
        st.subheader("🔍 Outliers (IQR)")
        
        colunas_num = df.select_dtypes(include=[np.number]).columns.tolist()
        
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
    st.header("📊 Análise Estatística")
    st.markdown("---")
    
    df = obter_df_limpo()
    
    if df is None:
        st.warning("⚠️ Carregue os dados na Aba 1 primeiro!")
    else:
        colunas_num = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not colunas_num:
            st.warning("⚠️ Nenhuma variável numérica")
        else:
            col_analise = st.selectbox("Variável:", colunas_num, key="stat_col")
            
            stats = calcular_estatisticas(df, col_analise)
            
            if stats:
                st.subheader(f"📈 {col_analise}")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Média", f"{stats['Média']:.3f}")
                c2.metric("Desvio", f"{stats['Desvio Padrão']:.3f}")
                cv_v = stats['CV (%)']
                c3.metric("CV (%)", f"{cv_v:.1f}%", classificar_cv(cv_v))
                c4.metric("N", f"{stats['Amostras']:,}")
                
                # Histograma
                st.markdown("---")
                st.subheader("📊 Distribuição")
                
                bins = st.slider("Bins:", 10, 100, 30, key="bins_s")
                df_plot = pd.DataFrame({col_analise: df[col_analise].dropna().values})
                chart_h = criar_histograma(df_plot, col_analise, bins)
                if chart_h:
                    st.altair_chart(chart_h, use_container_width=True)
                
                # Correlação
                if len(colunas_num) >= 2:
                    st.markdown("---")
                    st.subheader("🔗 Correlação de Pearson")
                    
                    corr = df[colunas_num].dropna(axis=1, how="all").corr()
                    
                    if not corr.empty:
                        st.dataframe(
                            corr.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1).format("{:.3f}"),
                            use_container_width=True
                        )
                
                # Série temporal (SÓ SE HOUVER DATAS)
                date_cols = st.session_state.get('date_columns', [])
                
                if date_cols:
                    st.markdown("---")
                    st.subheader("📅 Série Temporal")
                    
                    col_data = st.selectbox("Data:", date_cols, key="dt_s")
                    
                    df_plot = df[[col_data, col_analise]].dropna().copy()
                    df_plot[col_data] = pd.to_datetime(df_plot[col_data], errors='coerce')
                    df_plot = df_plot.dropna()
                    
                    if len(df_plot) > 0:
                        chart_t = criar_grafico_linhas(df_plot, col_data, [col_analise])
                        if chart_t:
                            st.altair_chart(chart_t, use_container_width=True)
                else:
                    st.info("📅 Nenhuma coluna de data disponível para série temporal.")

# ============================================================
# ABA 4: GRÁFICOS
# ============================================================
with aba4:
    st.header("📈 Visualização Gráfica")
    st.markdown("---")
    
    df = obter_df_limpo()
    
    if df is None:
        st.warning("⚠️ Carregue os dados na Aba 1 primeiro!")
    else:
        colunas_num = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not colunas_num:
            st.warning("⚠️ Sem variáveis numéricas")
        else:
            tipo = st.radio("Tipo:", ['📈 Linhas', '📊 Histograma', '📉 Boxplot'], horizontal=True, key="tipo_g")
            
            st.markdown("---")
            
            if tipo == '📈 Linhas':
                date_cols = st.session_state.get('date_columns', [])
                
                if date_cols:
                    col_x = st.selectbox("Eixo X:", date_cols, key="gx")
                    col_y = st.multiselect("Eixo Y:", colunas_num, default=colunas_num[:min(3, len(colunas_num))], key="gy")
                    
                    if col_y:
                        df_plot = df[[col_x] + col_y].dropna().copy()
                        df_plot[col_x] = pd.to_datetime(df_plot[col_x], errors='coerce')
                        df_plot = df_plot.dropna()
                        
                        if len(df_plot) > 0:
                            chart = criar_grafico_linhas(df_plot, col_x, col_y)
                            if chart:
                                st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("📅 Nenhuma coluna de data detectada. Use Histograma ou Boxplot.")
            
            elif tipo == '📊 Histograma':
                col_h = st.selectbox("Variável:", colunas_num, key="gh")
                bins_h = st.slider("Bins:", 5, 100, 30, key="gb")
                
                df_plot = pd.DataFrame({col_h: df[col_h].dropna().values})
                chart = criar_histograma(df_plot, col_h, bins_h)
                if chart:
                    st.altair_chart(chart, use_container_width=True)
            
            elif tipo == '📉 Boxplot':
                col_b = st.selectbox("Variável:", colunas_num, key="gbox")
                
                df_plot = pd.DataFrame({col_b: df[col_b].dropna().values})
                chart = criar_boxplot(df_plot, col_b)
                if chart:
                    st.altair_chart(chart, use_container_width=True)

# ============================================================
# ABA 5: DOWNLOAD
# ============================================================
with aba5:
    st.header("💾 Download")
    st.markdown("---")
    
    df = obter_df_limpo()
    
    if df is None:
        st.warning("⚠️ Carregue os dados primeiro!")
    else:
        st.markdown('<div class="success-box"><h3>✅ Pronto!</h3></div>', unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros", f"{len(df):,}")
        c2.metric("Colunas", len(df.columns))
        c3.metric("Ausentes", df.isnull().sum().sum())
        c4.metric("Tamanho", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        st.markdown("---")
        
        csv_data = df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            "📥 Baixar CSV",
            csv_data,
            f"dados_processados_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
        
        st.markdown("---")
        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:1.5rem; color:#6c757d; background:#f8f9fa; border-radius:12px;">
    <h4>🌱 AgroDataLab v7.0</h4>
    <p>Detecção Robusta de CSV | Análise Inteligente de Datas</p>
    <p style="font-size:0.8rem;">Licença MIT © 2024</p>
</div>
""", unsafe_allow_html=True)
