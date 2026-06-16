import streamlit as st
import sys
import pandas as pd
import numpy as np
import requests
import zipfile
import json
import os
from io import BytesIO
import csv
from datetime import datetime

st.write(sys.version)

# =============================================================================
# FUNÇÃO Z-SCORE SEM SCIPY
# =============================================================================

def zscore_simples(serie):
    serie = pd.to_numeric(serie, errors="coerce")
    media = serie.mean()
    desvio = serie.std()
    if desvio == 0 or pd.isna(desvio):
        return np.zeros(len(serie))
    return (serie - media) / desvio

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA - TELA CHEIA
# =============================================================================

st.set_page_config(
    page_title="AgroClimate AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CSS GLOBAL - DESIGN COLORIDO E GRANDE
# =============================================================================

st.markdown(
    """
    <style>
    /* Estilo geral - tela cheia */
    .main {
        padding: 0rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        max-width: 100%;
    }
    
    /* Cards grandes e coloridos */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Título principal */
    .main-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 3rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(245, 87, 108, 0.3);
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    .main-header p {
        margin: 0.8rem 0 0 0;
        opacity: 0.95;
        font-size: 1.3rem;
    }
    
    /* Botões grandes */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 14px;
        border: none;
        font-weight: 700;
        padding: 0.9rem 2.5rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    
    /* Área de upload - grande e destacada */
    .stFileUploader > div {
        border: 3px dashed #764ba2;
        border-radius: 20px;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, #f8f9ff, #eef1ff);
        transition: all 0.3s ease;
        min-height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .stFileUploader > div:hover {
        border-color: #f5576c;
        background: linear-gradient(135deg, #f0f2ff, #e6eaff);
        transform: scale(1.01);
        box-shadow: 0 8px 30px rgba(118, 75, 162, 0.15);
    }
    
    /* Métricas grandes */
    .stMetric {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.3);
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        background: rgba(255, 255, 255, 1);
    }
    
    .stMetric > div {
        background-color: transparent !important;
    }
    
    .stMetric label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #4a4a6a !important;
    }
    
    .stMetric div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #764ba2 !important;
    }
    
    /* Abas grandes */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        padding: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        backdrop-filter: blur(5px);
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        color: #4a4a6a;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(102, 126, 234, 0.1);
        color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: 700;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Caixas de informação */
    .info-box {
        background: linear-gradient(135deg, #e8f0fe, #d4e4f7);
        border-left: 6px solid #667eea;
        padding: 1.5rem 2rem;
        border-radius: 14px;
        margin: 1.5rem 0;
        color: #2d3748;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
    }
    
    .config-box {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid #e8ecf1;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .config-title {
        font-weight: 700;
        color: #4a4a6a;
        margin-bottom: 0.8rem;
        font-size: 1.3rem;
    }
    
    /* Seções */
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4a4a6a;
        margin-bottom: 1.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 4px solid #667eea;
        display: inline-block;
    }
    
    /* Inputs grandes - CORRIGIDO: texto em verde escuro */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e0e4ed;
        padding: 0.8rem 1rem;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        background: white;
        color: #1a5c1a !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15);
        color: #1a5c1a !important;
    }
    
    /* Selectbox - CORRIGIDO: texto em verde escuro */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e0e4ed;
        padding: 0.3rem 1rem;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        background: white;
        color: #1a5c1a !important;
    }
    
    .stSelectbox > div > div > div {
        color: #1a5c1a !important;
    }
    
    .stSelectbox > div > div:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15);
    }
    
    /* Selectbox options - CORRIGIDO: texto em verde escuro */
    .stSelectbox > div > div > div > div {
        color: #1a5c1a !important;
    }
    
    .stSelectbox > div > div > div > div:hover {
        background-color: #e8f5e9 !important;
    }
    
    /* Checkbox grande */
    .stCheckbox > label {
        font-weight: 600;
        color: #4a4a6a;
        font-size: 1.05rem;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 16px;
        border: 1px solid #e8ecf1;
        overflow: hidden;
        background: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .stDataFrame > div {
        border-radius: 16px;
    }
    
    /* Footer */
    footer {
        visibility: hidden;
    }
    
    /* Rodapé */
    .footer {
        text-align: center;
        font-size: 14px;
        color: rgba(255,255,255,0.8);
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
        border-top: 2px solid rgba(255,255,255,0.1);
        margin-top: 2rem;
    }
    
    .footer b {
        color: white;
    }
    
    /* Nome do arquivo carregado */
    .file-info {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 14px;
        margin: 1rem 0;
        font-weight: 600;
        font-size: 1.2rem;
        box-shadow: 0 4px 20px rgba(245, 87, 108, 0.3);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .file-info span {
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 400;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# SESSION STATE
# =============================================================================

if "usuario" not in st.session_state:
    st.session_state["usuario"] = {}

if "dados_salvos" not in st.session_state:
    st.session_state["dados_salvos"] = False

if "df_original" not in st.session_state:
    st.session_state["df_original"] = None

if "df_tratado" not in st.session_state:
    st.session_state["df_tratado"] = None

if "df_consolidado" not in st.session_state:
    st.session_state["df_consolidado"] = None

if "estatisticas" not in st.session_state:
    st.session_state["estatisticas"] = None

if "relatorio_ia" not in st.session_state:
    st.session_state["relatorio_ia"] = None

if "df_indicadores" not in st.session_state:
    st.session_state["df_indicadores"] = None

if "latitude" not in st.session_state:
    st.session_state["latitude"] = -16.0

if "ia_ativada" not in st.session_state:
    st.session_state["ia_ativada"] = False

if "arquivo_nome" not in st.session_state:
    st.session_state["arquivo_nome"] = None

if "df_mensal" not in st.session_state:
    st.session_state["df_mensal"] = None

# =============================================================================
# TÍTULO PRINCIPAL
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>📊 AgroClimate AI</h1>
    <p>Sistema Inteligente para Processamento, Análise e Interpretação de Dados Climáticos e Agrícolas</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# CRIAÇÃO DAS ABAS
# =============================================================================

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Início",
    "📂 Importação",
    "🧹 Tratamento",
    "📅 Consolidação",
    "📈 Estatística",
    "📊 Gráficos",
    "🤖 IA",
    "📥 Exportação",
    "🌡️ Indicadores"
])

# =============================================================================
# ABA 0 - INÍCIO
# =============================================================================

with tab0:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Cadastro do Projeto</div>', unsafe_allow_html=True)
    st.markdown("Preencha os dados abaixo para identificação do projeto, pesquisador e instituição.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        nome_projeto = st.text_input("📌 Nome do Projeto")
        pesquisador = st.text_input("👨‍🔬 Pesquisador Responsável")
        instituicao = st.text_input("🏛️ Instituição")
    
    with col2:
        email = st.text_input("📧 E-mail")
        cidade = st.text_input("📍 Cidade")
        estado = st.text_input("🗺️ Estado")
    
    observacoes = st.text_area("📝 Observações", height=100)
    
    if st.button("💾 Salvar Cadastro", use_container_width=True):
        st.session_state["usuario"] = {
            "Projeto": nome_projeto,
            "Pesquisador": pesquisador,
            "Instituição": instituicao,
            "Email": email,
            "Cidade": cidade,
            "Estado": estado,
            "Observações": observacoes
        }
        st.session_state["dados_salvos"] = True
        st.success("✅ Cadastro salvo com sucesso!")
    
    st.markdown("---")
    
    if st.session_state["dados_salvos"]:
        st.markdown('<div class="section-title">📋 Informações Salvas</div>', unsafe_allow_html=True)
        st.json(st.session_state["usuario"])
    
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <strong>📋 Fluxo recomendado:</strong><br>
        1️⃣ Importar dados → 2️⃣ Tratar dados → 3️⃣ Consolidar dados → 
        4️⃣ Gerar estatísticas → 5️⃣ Criar gráficos → 6️⃣ Gerar relatório com IA → 
        7️⃣ Exportar resultados → 8️⃣ Avaliar indicadores agrometeorológicos
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FUNÇÕES DE IMPORTAÇÃO
# =============================================================================

def detectar_delimitador(arquivo):
    try:
        arquivo.seek(0)
        amostra = arquivo.read(5000)
        if isinstance(amostra, bytes):
            amostra = amostra.decode("utf-8", errors="ignore")
        delimitadores = [",", ";", "\t", "|"]
        contagem = {}
        for d in delimitadores:
            contagem[d] = amostra.count(d)
        return max(contagem, key=contagem.get)
    except:
        return ";"

def detectar_encoding(arquivo):
    encodings = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]
    for enc in encodings:
        try:
            arquivo.seek(0)
            conteudo = arquivo.read(3000)
            if isinstance(conteudo, bytes):
                conteudo.decode(enc)
            return enc
        except:
            continue
    return "latin1"

def remover_colunas_duplicadas(df):
    return df.loc[:, ~df.columns.duplicated()]

def converter_colunas_numericas(df):
    df2 = df.copy()
    for col in df2.columns:
        try:
            df2[col] = df2[col].astype(str).str.replace(",", ".", regex=False)
            df2[col] = pd.to_numeric(df2[col], errors="ignore")
        except:
            pass
    return df2

def detectar_colunas_data(df):
    candidatas = []
    palavras = ["data", "date", "datetime", "tempo", "hora", "timestamp"]
    for col in df.columns:
        nome = str(col).lower()
        if any(p in nome for p in palavras):
            candidatas.append(col)
    return candidatas

def converter_datas(df):
    df2 = df.copy()
    datas = detectar_colunas_data(df2)
    for col in datas:
        try:
            df2[col] = pd.to_datetime(df2[col], errors="coerce", dayfirst=True)
        except:
            pass
    return df2

def identificar_tipo_planilha(df):
    nomes = " ".join(df.columns.astype(str)).lower()
    termos_climaticos = [
        "temp", "temperatura", "umidade", "vento", "rad",
        "radiação", "radiacao", "chuva", "precip", "eto", "evap"
    ]
    score = sum(termo in nomes for termo in termos_climaticos)
    if score >= 2:
        return "📊 Climática"
    return "📋 Genérica"

def ler_planilha_universal(arquivo):
    nome = arquivo.name.lower()
    if nome.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(arquivo, engine="openpyxl")
        except:
            df = pd.read_csv(arquivo, encoding='utf-8', on_bad_lines='skip')
    else:
        delimitador = detectar_delimitador(arquivo)
        encoding = detectar_encoding(arquivo)
        arquivo.seek(0)
        df = pd.read_csv(arquivo, sep=delimitador, encoding=encoding, on_bad_lines="skip")
    
    df = remover_colunas_duplicadas(df)
    df = converter_colunas_numericas(df)
    df = converter_datas(df)
    return df

# =============================================================================
# ABA 1 - IMPORTAÇÃO
# =============================================================================

with tab1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📂 Importação Inteligente</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="config-box">
        <div class="config-title">📁 Upload do Arquivo INMET</div>
        <div style="color: #666; font-size: 1.1rem;">
            <strong>200MB per file</strong> - CSV, TXT, DAT, XLS, XLSX
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Arraste e solte o arquivo aqui",
        type=["csv", "txt", "dat", "xls", "xlsx"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            df = ler_planilha_universal(uploaded_file)
            st.session_state["df_original"] = df
            
            file_size = uploaded_file.size / (1024 * 1024)
            st.markdown(f"""
            <div class="file-info">
                📄 {uploaded_file.name}
                <span>{file_size:.1f} MB</span>
            </div>
            """, unsafe_allow_html=True)
            
            tipo_planilha = identificar_tipo_planilha(df)
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Linhas", f"{len(df):,}")
            with col2:
                st.metric("📋 Colunas", len(df.columns))
            with col3:
                st.metric("⚠️ Valores Ausentes", int(df.isna().sum().sum()))
            with col4:
                st.metric("📌 Tipo", tipo_planilha)
            
            st.markdown("---")
            st.markdown('<div class="section-title">👁 Pré-visualização</div>', unsafe_allow_html=True)
            st.dataframe(df.head(100), use_container_width=True)
            
            st.markdown("---")
            st.markdown('<div class="section-title">📋 Estrutura das Colunas</div>', unsafe_allow_html=True)
            tipos = pd.DataFrame({
                "Coluna": df.columns,
                "Tipo": df.dtypes.astype(str)
            })
            st.dataframe(tipos, use_container_width=True)
            
            st.markdown("---")
            st.markdown('<div class="section-title">⚠️ Valores Ausentes</div>', unsafe_allow_html=True)
            faltantes = pd.DataFrame({
                "Coluna": df.columns,
                "Faltantes": df.isna().sum(),
                "Percentual (%)": (df.isna().sum() / len(df) * 100).round(2)
            })
            st.dataframe(faltantes, use_container_width=True)
            
            st.success("✅ Arquivo carregado com sucesso!")
        except Exception as erro:
            st.error(f"❌ Erro ao carregar arquivo: {erro}")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #999;">
            <span style="font-size: 3rem;">📁</span>
            <p style="font-size: 1.2rem; margin-top: 0.5rem;">Nenhum arquivo selecionado</p>
            <p style="font-size: 0.9rem;">Arraste ou clique para fazer upload</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FUNÇÕES DE TRATAMENTO DE DADOS
# =============================================================================

def obter_colunas_numericas(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def preencher_media(df):
    df2 = df.copy()
    for col in obter_colunas_numericas(df2):
        media = df2[col].mean()
        df2[col] = df2[col].fillna(media)
    return df2

def preencher_mediana(df):
    df2 = df.copy()
    for col in obter_colunas_numericas(df2):
        mediana = df2[col].median()
        df2[col] = df2[col].fillna(mediana)
    return df2

def preencher_moda(df):
    df2 = df.copy()
    for col in df2.columns:
        try:
            moda = df2[col].mode()[0]
            df2[col] = df2[col].fillna(moda)
        except:
            pass
    return df2

def preencher_interpolacao(df):
    df2 = df.copy()
    for col in obter_colunas_numericas(df2):
        try:
            df2[col] = df2[col].interpolate(method="linear", limit_direction="both")
        except:
            pass
    return df2

def preencher_polinomial(df):
    df2 = df.copy()
    for col in obter_colunas_numericas(df2):
        try:
            df2[col] = df2[col].interpolate(method="polynomial", order=2)
        except:
            pass
    return df2

def remover_duplicados(df):
    return df.drop_duplicates()

def padronizar_colunas(df):
    df2 = df.copy()
    novas = []
    for col in df2.columns:
        nome = str(col)
        nome = nome.strip()
        nome = nome.replace(" ", "_")
        nome = nome.replace("-", "_")
        nome = nome.replace("/", "_")
        nome = nome.replace("(", "")
        nome = nome.replace(")", "")
        novas.append(nome)
    df2.columns = novas
    return df2

def detectar_outliers_iqr(df, fator=1.5):
    resultado = {}
    for col in obter_colunas_numericas(df):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - fator * iqr
        limite_superior = q3 + fator * iqr
        mascara = (df[col] < limite_inferior) | (df[col] > limite_superior)
        resultado[col] = mascara
    return resultado

def detectar_outliers_zscore(df, limite=3):
    resultado = {}
    for col in obter_colunas_numericas(df):
        try:
            z = np.abs(zscore_simples(df[col]))
            resultado[col] = z > limite
        except:
            resultado[col] = np.zeros(len(df), dtype=bool)
    return resultado

def aplicar_acao_outlier(df, mascaras, acao="remover"):
    df2 = df.copy()
    mascara_total = np.zeros(len(df2), dtype=bool)
    for col in mascaras:
        mascara_total |= mascaras[col]
    if acao == "remover":
        df2 = df2[~mascara_total]
    elif acao == "media":
        for col in mascaras:
            media = df2[col].mean()
            df2.loc[mascaras[col], col] = media
    elif acao == "mediana":
        for col in mascaras:
            mediana = df2[col].median()
            df2.loc[mascaras[col], col] = mediana
    return df2

# =============================================================================
# ABA 2 - TRATAMENTO
# =============================================================================

with tab2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧹 Tratamento Inteligente</div>', unsafe_allow_html=True)
    
    if "df_original" not in st.session_state or st.session_state["df_original"] is None:
        st.warning("⚠️ Importe uma planilha primeiro na aba 'Importação'.")
    else:
        df_base = st.session_state["df_original"].copy()
        
        st.markdown("""
        <div class="config-box">
            <div class="config-title">📊 Métodos de Preenchimento</div>
            <p style="margin: 0; color: #555; font-size: 1.05rem;">Técnica para preenchimento de falhas:</p>
        </div>
        """, unsafe_allow_html=True)
        
        metodo_falhas = st.selectbox(
            "Selecione o método",
            ["Nenhum", "Média", "Mediana", "Moda", "Interpolação Linear", "Interpolação Polinomial"],
            label_visibility="collapsed"
        )
        
        st.markdown("""
        <div class="config-box" style="margin-top: 1rem;">
            <div class="config-title">🎯 Detecção de Outliers</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            metodo_outlier = st.selectbox(
                "Método de detecção",
                ["Nenhum", "IQR", "Z-Score"]
            )
        with col2:
            acao_outlier = st.selectbox(
                "Ação para os outliers",
                ["remover", "media", "mediana"]
            )
        
        st.markdown("""
        <div class="config-box" style="margin-top: 1rem;">
            <div class="config-title">⚙️ Opções Adicionais</div>
        </div>
        """, unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            remover_dup = st.checkbox("Remover linhas duplicadas", value=True)
        with col4:
            padronizar = st.checkbox("Padronizar nomes das colunas", value=True)
        
        st.markdown("---")
        
        if st.button("🚀 Executar Tratamento", use_container_width=True):
            df_tratado = df_base.copy()
            
            if padronizar:
                df_tratado = padronizar_colunas(df_tratado)
            
            if remover_dup:
                antes = len(df_tratado)
                df_tratado = remover_duplicados(df_tratado)
                depois = len(df_tratado)
                removidos = antes - depois
                st.success(f"✅ {removidos} linhas duplicadas removidas.")
            
            if metodo_falhas == "Média":
                df_tratado = preencher_media(df_tratado)
            elif metodo_falhas == "Mediana":
                df_tratado = preencher_mediana(df_tratado)
            elif metodo_falhas == "Moda":
                df_tratado = preencher_moda(df_tratado)
            elif metodo_falhas == "Interpolação Linear":
                df_tratado = preencher_interpolacao(df_tratado)
            elif metodo_falhas == "Interpolação Polinomial":
                df_tratado = preencher_polinomial(df_tratado)
            
            if metodo_outlier == "IQR":
                mascaras = detectar_outliers_iqr(df_tratado)
                df_tratado = aplicar_acao_outlier(df_tratado, mascaras, acao_outlier)
            elif metodo_outlier == "Z-Score":
                mascaras = detectar_outliers_zscore(df_tratado)
                df_tratado = aplicar_acao_outlier(df_tratado, mascaras, acao_outlier)
            
            st.session_state["df_tratado"] = df_tratado
            st.success("✅ Tratamento concluído com sucesso!")
        
        if "df_tratado" in st.session_state and st.session_state["df_tratado"] is not None:
            df_resultado = st.session_state["df_tratado"]
            st.markdown("---")
            st.markdown('<div class="section-title">📊 Resultado do Tratamento</div>', unsafe_allow_html=True)
            st.dataframe(df_resultado.head(100), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Linhas", len(df_resultado))
            with col2:
                st.metric("📋 Colunas", len(df_resultado.columns))
            with col3:
                st.metric("⚠️ Valores Nulos", int(df_resultado.isna().sum().sum()))
            
            st.success("✅ Base pronta para Consolidação!")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FUNÇÕES DE CONSOLIDAÇÃO
# =============================================================================

def identificar_coluna_data(df):
    palavras = ["data", "date", "datetime", "tempo", "timestamp"]
    for col in df.columns:
        nome = str(col).lower()
        if any(p in nome for p in palavras):
            return col
    return None

def consolidar_dataframe(df):
    df2 = df.copy()
    coluna_data = identificar_coluna_data(df2)
    if coluna_data is not None:
        try:
            df2 = df2.sort_values(by=coluna_data)
        except:
            pass
    df2 = df2.reset_index(drop=True)
    return df2

def gerar_resumo_consolidacao(df):
    resumo = {
        "Linhas": len(df),
        "Colunas": len(df.columns),
        "Valores Nulos": int(df.isna().sum().sum()),
        "Colunas Numéricas": len(df.select_dtypes(include=np.number).columns)
    }
    return resumo

# =============================================================================
# ABA 3 - CONSOLIDAÇÃO
# =============================================================================

with tab3:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📅 Consolidação dos Dados</div>', unsafe_allow_html=True)
    
    if "df_tratado" not in st.session_state or st.session_state["df_tratado"] is None:
        st.warning("⚠️ Primeiro execute o tratamento dos dados.")
    else:
        df_base = st.session_state["df_tratado"].copy()
        
        st.markdown("""
        <div class="info-box">
            Esta etapa organiza a base final, ordena datas e prepara os dados para análises estatísticas.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Consolidar Dados", use_container_width=True):
            df_consolidado = consolidar_dataframe(df_base)
            st.session_state["df_consolidado"] = df_consolidado
            st.success("✅ Dados consolidados com sucesso!")
        
        if "df_consolidado" in st.session_state and st.session_state["df_consolidado"] is not None:
            df_consolidado = st.session_state["df_consolidado"]
            resumo = gerar_resumo_consolidacao(df_consolidado)
            st.markdown("---")
            st.markdown('<div class="section-title">📊 Resumo da Consolidação</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Linhas", resumo["Linhas"])
            with col2:
                st.metric("📋 Colunas", resumo["Colunas"])
            with col3:
                st.metric("⚠️ Nulos", resumo["Valores Nulos"])
            with col4:
                st.metric("🔢 Numéricas", resumo["Colunas Numéricas"])
            
            st.markdown("---")
            st.markdown('<div class="section-title">👁 Pré-visualização</div>', unsafe_allow_html=True)
            st.dataframe(df_consolidado.head(200), use_container_width=True)
            st.success("✅ Base pronta para Estatística, Gráficos e IA!")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FUNÇÕES DE ESTATÍSTICA
# =============================================================================

def classificar_cv(cv):
    if pd.isna(cv):
        return "-"
    if cv < 10:
        return "Baixo"
    elif cv < 20:
        return "Médio"
    elif cv < 30:
        return "Alto"
    else:
        return "Muito Alto"

def gerar_estatisticas(df):
    numericas = df.select_dtypes(include=np.number)
    resultados = []
    for col in numericas.columns:
        serie = numericas[col].dropna()
        if len(serie) == 0:
            continue
        media = serie.mean()
        desvio = serie.std()
        cv = (desvio / media * 100) if media != 0 else np.nan
        try:
            moda = serie.mode().iloc[0]
        except:
            moda = np.nan
        resultados.append({
            "Variável": col,
            "N": len(serie),
            "Média": round(media, 4),
            "Mediana": round(serie.median(), 4),
            "Moda": round(moda, 4) if pd.notna(moda) else np.nan,
            "Mínimo": round(serie.min(), 4),
            "Máximo": round(serie.max(), 4),
            "Amplitude": round(serie.max() - serie.min(), 4),
            "Variância": round(serie.var(), 4),
            "Desvio Padrão": round(desvio, 4),
            "CV (%)": round(cv, 2),
            "Classificação CV": classificar_cv(cv),
            "Assimetria": round(serie.skew(), 4),
            "Curtose": round(serie.kurtosis(), 4),
            "P5": round(serie.quantile(0.05), 4),
            "P25": round(serie.quantile(0.25), 4),
            "P75": round(serie.quantile(0.75), 4),
            "P95": round(serie.quantile(0.95), 4)
        })
    return pd.DataFrame(resultados)

def detectar_extremos(df):
    extremos = []
    numericas = df.select_dtypes(include=np.number)
    for col in numericas.columns:
        try:
            maior = numericas[col].max()
            menor = numericas[col].min()
            extremos.append({"Variável": col, "Tipo": "Máximo", "Valor": maior})
            extremos.append({"Variável": col, "Tipo": "Mínimo", "Valor": menor})
        except:
            pass
    return pd.DataFrame(extremos)

def calcular_media_mensal(df):
    """Calcula a média mensal dos dados"""
    df2 = df.copy()
    
    # Identificar coluna de data
    col_data = identificar_coluna_data(df2)
    
    if col_data is None:
        return None, "Nenhuma coluna de data encontrada."
    
    # Garantir que a coluna de data é datetime
    df2[col_data] = pd.to_datetime(df2[col_data], errors='coerce')
    
    # Extrair mês e ano
    df2['Ano'] = df2[col_data].dt.year
    df2['Mes'] = df2[col_data].dt.month
    df2['Ano_Mes'] = df2[col_data].dt.strftime('%Y-%m')
    
    # Selecionar apenas colunas numéricas
    numericas = df2.select_dtypes(include=np.number).columns.tolist()
    
    if not numericas:
        return None, "Nenhuma coluna numérica encontrada."
    
    # Calcular média mensal
    media_mensal = df2.groupby('Ano_Mes')[numericas].mean().reset_index()
    
    return media_mensal, None

# =============================================================================
# ABA 4 - ESTATÍSTICA
# =============================================================================

with tab4:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Estatística Descritiva</div>', unsafe_allow_html=True)
    
    if "df_consolidado" not in st.session_state or st.session_state["df_consolidado"] is None:
        st.warning("⚠️ Primeiro consolide os dados na aba 'Consolidação'.")
    else:
        df_base = st.session_state["df_consolidado"]
        
        # ============================================================
        # MÉDIA MENSAL
        # ============================================================
        st.markdown("### 📊 Média Mensal dos Dados")
        
        col_data = identificar_coluna_data(df_base)
        
        if col_data is not None:
            if st.button("📊 Calcular Média Mensal", use_container_width=True):
                media_mensal, erro = calcular_media_mensal(df_base)
                if erro:
                    st.error(f"❌ {erro}")
                else:
                    st.session_state["df_mensal"] = media_mensal
                    st.success("✅ Média mensal calculada com sucesso!")
            
            if "df_mensal" in st.session_state and st.session_state["df_mensal"] is not None:
                df_mensal = st.session_state["df_mensal"]
                st.dataframe(df_mensal, use_container_width=True)
                
                # Gráfico da média mensal
                colunas_num = df_mensal.select_dtypes(include=np.number).columns.tolist()
                if colunas_num:
                    var_mensal = st.selectbox("Selecione a variável para visualizar a média mensal", colunas_num)
                    if var_mensal:
                        st.markdown(f"**📈 Média Mensal - {var_mensal}**")
                        st.line_chart(df_mensal.set_index('Ano_Mes')[var_mensal], use_container_width=True)
        else:
            st.info("ℹ️ Nenhuma coluna de data encontrada para calcular média mensal.")
        
        st.markdown("---")
        
        # ============================================================
        # ESTATÍSTICA COMPLETA
        # ============================================================
        st.markdown("### 📊 Estatística Completa")
        estatisticas = gerar_estatisticas(df_base)
        st.dataframe(estatisticas, use_container_width=True)
        
        st.markdown("---")
        
        # ============================================================
        # VALORES EXTREMOS
        # ============================================================
        st.markdown("### 📌 Valores Extremos")
        extremos = detectar_extremos(df_base)
        st.dataframe(extremos, use_container_width=True)
        
        st.markdown("---")
        
        # ============================================================
        # MÉTRICAS RÁPIDAS
        # ============================================================
        st.markdown("### 📋 Métricas Rápidas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Variáveis", len(estatisticas))
        with col2:
            st.metric("📌 Extremos", len(extremos))
        with col3:
            st.metric("📋 Registros", len(df_base))
        
        st.markdown("---")
        
        # ============================================================
        # GRÁFICOS DE DISTRIBUIÇÃO
        # ============================================================
        numericas = df_base.select_dtypes(include=np.number).columns.tolist()
        
        if numericas:
            st.markdown('<div class="section-title">📊 Distribuição das Variáveis</div>', unsafe_allow_html=True)
            
            var_selecionada = st.selectbox("Selecione uma variável para análise gráfica", numericas)
            
            if var_selecionada:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**📊 Histograma - {var_selecionada}**")
                    hist_data = df_base[var_selecionada].dropna()
                    if len(hist_data) > 0:
                        bins = np.histogram_bin_edges(hist_data, bins='auto')
                        hist_counts = np.histogram(hist_data, bins=bins)[0]
                        hist_df = pd.DataFrame({
                            'Intervalo': [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins)-1)],
                            'Frequência': hist_counts
                        })
                        st.bar_chart(hist_df.set_index('Intervalo')['Frequência'], use_container_width=True)
                
                with col2:
                    st.markdown(f"**📦 Estatísticas - {var_selecionada}**")
                    stats = df_base[var_selecionada].describe()
                    st.metric("📉 Mínimo", round(stats['min'], 2))
                    st.metric("📊 Q1 (25%)", round(stats['25%'], 2))
                    st.metric("📈 Mediana", round(stats['50%'], 2))
                    st.metric("📊 Q3 (75%)", round(stats['75%'], 2))
                    st.metric("📈 Máximo", round(stats['max'], 2))
        
        # ============================================================
        # DOWNLOAD
        # ============================================================
        st.markdown("---")
        st.session_state["estatisticas"] = estatisticas
        csv_estatisticas = estatisticas.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Baixar Estatísticas",
            csv_estatisticas,
            file_name="estatisticas_descritivas.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# ABA 5 - GRÁFICOS
# =============================================================================

with tab5:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Visualização de Dados</div>', unsafe_allow_html=True)
    
    if "df_consolidado" not in st.session_state or st.session_state["df_consolidado"] is None:
        st.warning("⚠️ Primeiro consolide os dados na aba 'Consolidação'.")
    else:
        df = st.session_state["df_consolidado"]
        numericas = df.select_dtypes(include=np.number).columns.tolist()
        
        # Verificar se há dados mensais
        tem_mensal = "df_mensal" in st.session_state and st.session_state["df_mensal"] is not None
        
        # Opção para usar dados diários ou mensais
        tipo_dado = st.radio(
            "Selecione o tipo de dado para visualização",
            ["📊 Dados Diários", "📊 Dados Mensais"],
            horizontal=True
        )
        
        if tipo_dado == "📊 Dados Mensais" and tem_mensal:
            df_graf = st.session_state["df_mensal"]
            colunas_graf = df_graf.select_dtypes(include=np.number).columns.tolist()
            if 'Ano_Mes' in df_graf.columns:
                index_col = 'Ano_Mes'
            else:
                index_col = df_graf.columns[0]
        else:
            df_graf = df
            colunas_graf = numericas
            index_col = None
        
        if not colunas_graf:
            st.warning("Sem colunas numéricas para visualizar.")
        else:
            tipo_grafico = st.selectbox(
                "Selecione o tipo de gráfico",
                ["📈 Linhas", "📊 Barras", "📉 Área", "📊 Histograma"]
            )
            
            var_graf = st.selectbox("Selecione a variável", colunas_graf)
            
            if var_graf:
                if tipo_grafico == "📈 Linhas":
                    st.markdown(f"**📈 Série - {var_graf}**")
                    if index_col and index_col in df_graf.columns:
                        st.line_chart(df_graf.set_index(index_col)[var_graf], use_container_width=True)
                    else:
                        st.line_chart(df_graf[var_graf], use_container_width=True)
                
                elif tipo_grafico == "📊 Barras":
                    st.markdown(f"**📊 Barras - {var_graf}**")
                    if index_col and index_col in df_graf.columns:
                        st.bar_chart(df_graf.set_index(index_col)[var_graf], use_container_width=True)
                    else:
                        st.bar_chart(df_graf[var_graf], use_container_width=True)
                
                elif tipo_grafico == "📉 Área":
                    st.markdown(f"**📉 Área - {var_graf}**")
                    if index_col and index_col in df_graf.columns:
                        st.area_chart(df_graf.set_index(index_col)[var_graf], use_container_width=True)
                    else:
                        st.area_chart(df_graf[var_graf], use_container_width=True)
                
                elif tipo_grafico == "📊 Histograma":
                    st.markdown(f"**📊 Histograma - {var_graf}**")
                    hist_data = df_graf[var_graf].dropna()
                    if len(hist_data) > 0:
                        bins = np.histogram_bin_edges(hist_data, bins='auto')
                        hist_counts = np.histogram(hist_data, bins=bins)[0]
                        hist_df = pd.DataFrame({
                            'Intervalo': [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins)-1)],
                            'Frequência': hist_counts
                        })
                        st.bar_chart(hist_df.set_index('Intervalo')['Frequência'], use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📋 Estatísticas Rápidas")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📉 Mínimo", round(df_graf[var_graf].min(), 2))
            with col2:
                st.metric("📈 Máximo", round(df_graf[var_graf].max(), 2))
            with col3:
                st.metric("📊 Média", round(df_graf[var_graf].mean(), 2))
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# IA - PREPARAÇÃO DE CONTEXTO
# =============================================================================

def gerar_contexto_automatico(df):
    numericas = df.select_dtypes(include=np.number)
    contexto = f"""
Total de registros: {len(df)}
Total de variáveis: {len(df.columns)}
"""
    for col in numericas.columns:
        try:
            contexto += f"""
Variável: {col}
Média: {numericas[col].mean():.2f}
Mediana: {numericas[col].median():.2f}
Mínimo: {numericas[col].min():.2f}
Máximo: {numericas[col].max():.2f}
Desvio padrão: {numericas[col].std():.2f}
Coeficiente de variação: {(numericas[col].std()/numericas[col].mean()*100):.2f} %
"""
        except:
            pass
    return contexto

PROMPTS = {
    "📊 Relatório Científico": """
Produza um relatório científico completo.
Inclua:
- Introdução dos dados
- Interpretação estatística
- Tendências observadas
- Variabilidade
- Eventos extremos
- Conclusões técnicas
Utilize linguagem científica.
""",
    "🌡️ Relatório Meteorológico": """
Produza uma análise meteorológica detalhada.
Avalie:
- Temperatura
- Precipitação
- Umidade
- Vento
- Eventos extremos
- Tendências climáticas
""",
    "🌾 Relatório Agrícola": """
Interprete os dados sob o ponto de vista agronômico.
Avalie:
- Disponibilidade hídrica
- Riscos climáticos
- Potencial produtivo
- Conforto térmico
- Impactos agrícolas
""",
    "📋 Resumo Executivo": """
Produza um resumo executivo simples e objetivo.
Explique os resultados em linguagem acessível.
"""
}

def consultar_ia(prompt_usuario, contexto):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": contexto + "\n\n" + prompt_usuario}
                    ]
                }
            ]
        }
        resposta = requests.post(f"{url}?key={api_key}", headers={"Content-Type": "application/json"}, json=payload, timeout=120)
        if resposta.status_code == 200:
            dados = resposta.json()
            return dados["candidates"][0]["content"]["parts"][0]["text"]
        return f"Erro na API Gemini: {resposta.status_code}"
    except Exception as erro:
        return f"Erro: {erro}"

# =============================================================================
# ABA 6 - IA
# =============================================================================

with tab6:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 Inteligência Artificial</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="config-box">
        <div class="config-title">🧠 IA Gemini</div>
        <div style="display: flex; align-items: center; gap: 1rem; margin: 0.5rem 0;">
            <span style="font-weight: 500; font-size: 1.05rem;">Ativar análise com IA Gemini</span>
        </div>
        <div style="color: #555; font-size: 1rem; margin-top: 0.5rem;">
            A IA analisará padrões climáticos e fenômenos como El Niño/La Niña
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    ia_ativada = st.checkbox("Ativar IA Gemini", value=st.session_state["ia_ativada"])
    st.session_state["ia_ativada"] = ia_ativada
    
    if "df_consolidado" not in st.session_state or st.session_state["df_consolidado"] is None:
        st.warning("⚠️ Consolide os dados primeiro.")
    else:
        if ia_ativada:
            df_base = st.session_state["df_consolidado"]
            
            st.markdown("### 📊 Relatórios Inteligentes")
            tipo_relatorio = st.selectbox("Tipo de Relatório", list(PROMPTS.keys()))
            pergunta = st.text_area(
                "❓ Pergunta adicional",
                placeholder="Exemplo: Existe tendência de aumento da temperatura?",
                height=100
            )
            
            if st.button("🚀 Gerar Relatório", use_container_width=True):
                with st.spinner("Analisando dados..."):
                    contexto = gerar_contexto_automatico(df_base)
                    prompt = PROMPTS[tipo_relatorio] + "\n\n" + pergunta
                    resposta = consultar_ia(prompt, contexto)
                    st.session_state["relatorio_ia"] = resposta
            
            if "relatorio_ia" in st.session_state and st.session_state["relatorio_ia"] is not None:
                st.markdown("---")
                st.markdown("### 📄 Relatório Gerado")
                st.markdown(st.session_state["relatorio_ia"])
                st.download_button(
                    "📥 Baixar Relatório TXT",
                    st.session_state["relatorio_ia"],
                    file_name="relatorio_ia.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.info("ℹ️ Ative a IA Gemini para gerar relatórios inteligentes.")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FUNÇÕES DE EXPORTAÇÃO
# =============================================================================

def gerar_csv_unico():
    buffer = BytesIO()
    if "df_consolidado" in st.session_state and st.session_state["df_consolidado"] is not None:
        csv_data = st.session_state["df_consolidado"].to_csv(index=False)
        buffer.write(csv_data.encode('utf-8'))
    buffer.seek(0)
    return buffer

def gerar_zip_completo():
    memoria = BytesIO()
    with zipfile.ZipFile(memoria, "w", zipfile.ZIP_DEFLATED) as zipf:
        if "df_tratado" in st.session_state and st.session_state["df_tratado"] is not None:
            zipf.writestr("dados_tratados.csv", st.session_state["df_tratado"].to_csv(index=False))
        if "df_consolidado" in st.session_state and st.session_state["df_consolidado"] is not None:
            zipf.writestr("dados_consolidados.csv", st.session_state["df_consolidado"].to_csv(index=False))
        if "estatisticas" in st.session_state and st.session_state["estatisticas"] is not None:
            zipf.writestr("estatisticas.csv", st.session_state["estatisticas"].to_csv(index=False))
        if "df_indicadores" in st.session_state and st.session_state["df_indicadores"] is not None:
            zipf.writestr("indicadores.csv", st.session_state["df_indicadores"].to_csv(index=False))
        if "relatorio_ia" in st.session_state and st.session_state["relatorio_ia"] is not None:
            zipf.writestr("relatorio_ia.txt", st.session_state["relatorio_ia"])
        if "df_mensal" in st.session_state and st.session_state["df_mensal"] is not None:
            zipf.writestr("media_mensal.csv", st.session_state["df_mensal"].to_csv(index=False))
    memoria.seek(0)
    return memoria

def gerar_csv_estatisticas():
    buffer = BytesIO()
    if "estatisticas" in st.session_state and st.session_state["estatisticas"] is not None:
        csv_data = st.session_state["estatisticas"].to_csv(index=False)
        buffer.write(csv_data.encode('utf-8'))
    buffer.seek(0)
    return buffer

def gerar_csv_indicadores():
    buffer = BytesIO()
    if "df_indicadores" in st.session_state and st.session_state["df_indicadores"] is not None:
        csv_data = st.session_state["df_indicadores"].to_csv(index=False)
        buffer.write(csv_data.encode('utf-8'))
    buffer.seek(0)
    return buffer

def gerar_csv_mensal():
    buffer = BytesIO()
    if "df_mensal" in st.session_state and st.session_state["df_mensal"] is not None:
        csv_data = st.session_state["df_mensal"].to_csv(index=False)
        buffer.write(csv_data.encode('utf-8'))
    buffer.seek(0)
    return buffer

# =============================================================================
# ABA 7 - EXPORTAÇÃO
# =============================================================================

with tab7:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📥 Exportação Profissional</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Exporte seus resultados em formato CSV ou pacote ZIP completo.
    </div>
    """, unsafe_allow_html=True)
    
    tem_dados = (
        "df_consolidado" in st.session_state and 
        st.session_state["df_consolidado"] is not None
    )
    
    if not tem_dados:
        st.warning("⚠️ Não há dados consolidados para exportar. Consolide os dados primeiro.")
    else:
        st.markdown("### 📄 Exportar Arquivos Individuais")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            csv_file = gerar_csv_unico()
            st.download_button(
                "📄 Dados Consolidados",
                data=csv_file,
                file_name="dados_consolidados.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if "estatisticas" in st.session_state and st.session_state["estatisticas"] is not None:
                csv_estat = gerar_csv_estatisticas()
                st.download_button(
                    "📊 Estatísticas",
                    data=csv_estat,
                    file_name="estatisticas.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col3:
            if "df_indicadores" in st.session_state and st.session_state["df_indicadores"] is not None:
                csv_ind = gerar_csv_indicadores()
                st.download_button(
                    "🌡️ Indicadores",
                    data=csv_ind,
                    file_name="indicadores.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col4:
            if "df_mensal" in st.session_state and st.session_state["df_mensal"] is not None:
                csv_mensal = gerar_csv_mensal()
                st.download_button(
                    "📅 Média Mensal",
                    data=csv_mensal,
                    file_name="media_mensal.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        st.markdown("---")
        st.markdown("### 📦 Exportar Pacote Completo")
        
        zip_file = gerar_zip_completo()
        st.download_button(
            "📦 Baixar Pacote Completo (ZIP)",
            data=zip_file,
            file_name="projeto_completo.zip",
            mime="application/zip",
            use_container_width=True
        )
        
        st.markdown("---")
        st.success("✅ Exportação disponível em CSV e ZIP!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# INDICADORES AGROMETEOROLÓGICOS
# =============================================================================

def localizar_coluna(df, palavras):
    for col in df.columns:
        nome = str(col).lower()
        if any(p in nome for p in palavras):
            return col
    return None

def calcular_gdd(df, temperatura, temperatura_base=10):
    serie = pd.to_numeric(df[temperatura], errors="coerce")
    gdd = serie - temperatura_base
    gdd[gdd < 0] = 0
    return gdd

def calcular_precipitacao_acumulada(df, coluna):
    chuva = pd.to_numeric(df[coluna], errors="coerce")
    return chuva.cumsum()

def calcular_soma_termica(gdd):
    return gdd.cumsum()

def calcular_eto_hargreaves(tmin, tmax, tmed):
    eto = 0.0023 * (tmed + 17.8) * np.sqrt(tmax - tmin)
    return eto

def indice_conforto_termico(temperatura, umidade):
    return temperatura - (0.55 - 0.0055 * umidade) * (temperatura - 14.5)

# =============================================================================
# ABA 8 - INDICADORES
# =============================================================================

with tab8:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌡️ Indicadores Agrometeorológicos</div>', unsafe_allow_html=True)
    
    if "df_consolidado" not in st.session_state or st.session_state["df_consolidado"] is None:
        st.warning("⚠️ Consolide os dados primeiro.")
    else:
        df = st.session_state["df_consolidado"]
        st.markdown("Cálculo automático de indicadores agrícolas e meteorológicos.")
        
        st.markdown("""
        <div class="config-box">
            <div class="config-title">📍 Parâmetros Regionais</div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-weight: 500; font-size: 1.1rem;">Latitude da Estação (°):</span>
                <span style="color: #764ba2; font-weight: 700; font-size: 1.3rem;">-16,0</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        temperatura = localizar_coluna(df, ["temp", "temperatura", "tmed"])
        precipitacao = localizar_coluna(df, ["chuva", "precip", "precipitacao"])
        umidade = localizar_coluna(df, ["umidade", "ur"])
        tmin = localizar_coluna(df, ["tmin"])
        tmax = localizar_coluna(df, ["tmax"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if temperatura is not None:
                temperatura_base = st.number_input("🌡️ Temperatura Base (°C)", value=10.0, step=0.5)
                gdd = calcular_gdd(df, temperatura, temperatura_base)
                soma_termica = calcular_soma_termica(gdd)
                st.metric("📊 Graus-dia Acumulados", round(soma_termica.iloc[-1], 2))
                df["GDD"] = gdd
                df["Soma_Termica"] = soma_termica
            
            if precipitacao is not None:
                chuva_acumulada = calcular_precipitacao_acumulada(df, precipitacao)
                df["Chuva_Acumulada"] = chuva_acumulada
                st.metric("🌧️ Precipitação Acumulada (mm)", round(chuva_acumulada.iloc[-1], 2))
        
        with col2:
            if tmin is not None and tmax is not None and temperatura is not None:
                eto = calcular_eto_hargreaves(df[tmin], df[tmax], df[temperatura])
                df["ETo"] = eto
                st.metric("💧 ETo Média", round(eto.mean(), 2))
            
            if temperatura is not None and umidade is not None:
                conforto = indice_conforto_termico(df[temperatura], df[umidade])
                df["Indice_Conforto"] = conforto
                st.metric("🌡️ Conforto Médio", round(conforto.mean(), 2))
        
        st.markdown("---")
        st.markdown('<div class="section-title">📋 Indicadores Gerados</div>', unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True)
        st.session_state["df_indicadores"] = df
        
        indicadores_disponiveis = [col for col in df.columns if col in ['GDD', 'Soma_Termica', 'Chuva_Acumulada', 'ETo', 'Indice_Conforto']]
        if indicadores_disponiveis:
            st.markdown("---")
            st.markdown('<div class="section-title">📊 Visualização dos Indicadores</div>', unsafe_allow_html=True)
            
            for indicador in indicadores_disponiveis:
                st.markdown(f"**{indicador}**")
                st.line_chart(df[indicador], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# RODAPÉ INSTITUCIONAL
# =============================================================================

st.markdown("""
<div class="footer">
    <b>📊 AgroClimate AI</b><br>
    Sistema experimental desenvolvido para fins acadêmicos, educacionais, científicos e de apoio à análise de dados
    meteorológicos e agrometeorológicos.<br><br>
    Este aplicativo não substitui análises técnicas oficiais, laudos periciais, pareceres especializados ou sistemas
    operacionais de instituições governamentais.<br><br>
    © 2026 AgroClimate AI — Versão Acadêmica Experimental
</div>
""", unsafe_allow_html=True)
