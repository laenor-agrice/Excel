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
# FUNÇÕES ESTATÍSTICAS MANUAIS
# =============================================================================

def calcular_anova_manual(df, coluna, grupos):
    grupos_dados = df.groupby(grupos)[coluna].apply(list).to_dict()
    grupos_validos = {k: v for k, v in grupos_dados.items() if len(v) > 1}
    
    if len(grupos_validos) < 2:
        return None, "É necessário pelo menos 2 grupos com dados para ANOVA"
    
    dados_grupos = list(grupos_validos.values())
    nomes_grupos = list(grupos_validos.keys())
    
    todos_dados = []
    for grupo in dados_grupos:
        todos_dados.extend(grupo)
    media_geral = np.mean(todos_dados)
    n_total = len(todos_dados)
    k = len(dados_grupos)
    
    ssb = 0
    for grupo in dados_grupos:
        n_g = len(grupo)
        media_g = np.mean(grupo)
        ssb += n_g * (media_g - media_geral) ** 2
    
    ssw = 0
    for grupo in dados_grupos:
        media_g = np.mean(grupo)
        for valor in grupo:
            ssw += (valor - media_g) ** 2
    
    df_between = k - 1
    df_within = n_total - k
    msb = ssb / df_between if df_between > 0 else 0
    msw = ssw / df_within if df_within > 0 else 0
    f_stat = msb / msw if msw > 0 else 0
    
    try:
        from scipy.stats import f
        p_valor = 1 - f.cdf(f_stat, df_between, df_within)
    except:
        if f_stat > 4:
            p_valor = 0.01
        elif f_stat > 3:
            p_valor = 0.05
        elif f_stat > 2.5:
            p_valor = 0.1
        else:
            p_valor = 0.5
    
    tukey_result = None
    if p_valor < 0.05:
        try:
            medias = [np.mean(g) for g in dados_grupos]
            n_grupos = [len(g) for g in dados_grupos]
            n_medio = np.mean(n_grupos)
            erro_padrao = np.sqrt(msw * (1/n_medio + 1/n_medio))
            
            comparacoes = []
            for i in range(len(nomes_grupos)):
                for j in range(i+1, len(nomes_grupos)):
                    diff = medias[i] - medias[j]
                    q_stat = abs(diff) / erro_padrao if erro_padrao > 0 else 0
                    try:
                        from scipy.stats import t
                        p_tukey = 2 * (1 - t.cdf(abs(q_stat), df_within))
                    except:
                        p_tukey = 0.05 if q_stat > 2 else 0.1
                    
                    comparacoes.append({
                        "Grupo 1": nomes_grupos[i],
                        "Grupo 2": nomes_grupos[j],
                        "Diferença": round(diff, 4),
                        "p-value": round(p_tukey, 4),
                        "Significativo": p_tukey < 0.05
                    })
            tukey_result = comparacoes
        except:
            tukey_result = None
    
    resultado = {
        "f_statistic": f_stat,
        "p_value": p_valor,
        "significativo": p_valor < 0.05,
        "grupos": nomes_grupos,
        "df_between": df_between,
        "df_within": df_within,
        "tukey": tukey_result
    }
    return resultado, None

def calcular_teste_t_manual(df, coluna, grupos):
    dados_grupos = df.groupby(grupos)[coluna].apply(list).to_dict()
    if len(dados_grupos) != 2:
        return None, "O teste t requer exatamente 2 grupos"
    
    grupos_validos = {k: v for k, v in dados_grupos.items() if len(v) > 1}
    if len(grupos_validos) != 2:
        return None, "Cada grupo precisa ter pelo menos 2 observações"
    
    nomes = list(grupos_validos.keys())
    grupo1 = grupos_validos[nomes[0]]
    grupo2 = grupos_validos[nomes[1]]
    
    n1, n2 = len(grupo1), len(grupo2)
    media1, media2 = np.mean(grupo1), np.mean(grupo2)
    var1, var2 = np.var(grupo1, ddof=1), np.var(grupo2, ddof=1)
    
    t_stat = (media1 - media2) / np.sqrt(var1/n1 + var2/n2)
    df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    
    try:
        from scipy.stats import t
        p_valor = 2 * (1 - t.cdf(abs(t_stat), df))
    except:
        if abs(t_stat) > 2.5:
            p_valor = 0.01
        elif abs(t_stat) > 2:
            p_valor = 0.05
        elif abs(t_stat) > 1.5:
            p_valor = 0.1
        else:
            p_valor = 0.5
    
    resultado = {
        "t_statistic": t_stat,
        "p_value": p_valor,
        "df": df,
        "significativo": p_valor < 0.05,
        "grupo1": nomes[0],
        "grupo2": nomes[1],
        "media1": media1,
        "media2": media2
    }
    return resultado, None

def calcular_estatisticas_descritivas(df, coluna):
    dados = df[coluna].dropna()
    if len(dados) == 0:
        return None
    
    stats_dict = {
        "N": len(dados),
        "Média": dados.mean(),
        "Mediana": dados.median(),
        "Moda": dados.mode().iloc[0] if not dados.mode().empty else np.nan,
        "Desvio Padrão": dados.std(),
        "Variância": dados.var(),
        "CV (%)": (dados.std() / dados.mean() * 100) if dados.mean() != 0 else np.nan,
        "Mínimo": dados.min(),
        "Máximo": dados.max(),
        "Amplitude": dados.max() - dados.min(),
        "Q1 (25%)": dados.quantile(0.25),
        "Q3 (75%)": dados.quantile(0.75),
        "IQR": dados.quantile(0.75) - dados.quantile(0.25),
        "Assimetria": dados.skew(),
        "Curtose": dados.kurtosis(),
        "Erro Padrão": dados.std() / np.sqrt(len(dados))
    }
    return stats_dict

def calcular_correlacao(df, coluna1, coluna2):
    dados1 = df[coluna1].dropna()
    dados2 = df[coluna2].dropna()
    df_temp = pd.DataFrame({coluna1: dados1, coluna2: dados2}).dropna()
    if len(df_temp) < 2:
        return None
    return df_temp[coluna1].corr(df_temp[coluna2])

# =============================================================================
# FUNÇÃO PARA CALCULAR MÉDIA POR MÊS
# =============================================================================

def calcular_media_por_mes(df, coluna_data, variavel):
    """Calcula a média de uma variável para cada mês"""
    df2 = df.copy()
    df2['Mes'] = df2[coluna_data].dt.month
    
    media_mensal = df2.groupby('Mes')[variavel].mean().reset_index()
    media_mensal.columns = ['Mes', 'Media']
    
    nomes_meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                   'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    media_mensal['Mes_Nome'] = media_mensal['Mes'].apply(
        lambda x: nomes_meses[int(x)-1] if 1 <= int(x) <= 12 else str(x)
    )
    
    media_mensal = media_mensal.sort_values('Mes').reset_index(drop=True)
    
    return media_mensal

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
# CSS GLOBAL - EXATAMENTE COMO NA IMAGEM
# =============================================================================

st.markdown(
    """
    <style>
    /* Reset e estilo geral - tela cheia */
    .main {
        padding: 0rem !important;
        background: linear-gradient(135deg, #0d2b45 0%, #1a5276 40%, #2e86c1 80%, #3498db 100%);
        min-height: 100vh;
        width: 100%;
        max-width: 100%;
        margin: 0;
    }
    
    .block-container {
        padding: 0.5rem 2rem !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* ============ CABEÇALHO - DESTAQUE AZUL ============ */
    .main-header {
        background: linear-gradient(135deg, #0d2b45 0%, #1a5276 50%, #2e86c1 100%);
        padding: 2.5rem 3rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        color: white;
        box-shadow: 0 4px 25px rgba(0,0,0,0.25);
        text-align: center;
        width: 100%;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -1px;
        color: white;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
    }
    
    /* ============ CARDS - VERDE PISCINA ============ */
    .custom-card {
        background: linear-gradient(145deg, rgba(180, 230, 210, 0.92), rgba(140, 210, 190, 0.88));
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,50,30,0.12);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        width: 100%;
        color: #1a4a3a;
    }
    
    /* ============ BOTÕES ============ */
    .stButton > button {
        background: linear-gradient(135deg, #2d8a6e, #3da88a);
        color: white;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        padding: 0.7rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 2px 15px rgba(45, 138, 110, 0.25);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 25px rgba(45, 138, 110, 0.35);
        background: linear-gradient(135deg, #3da88a, #2d8a6e);
    }
    
    /* ============ UPLOAD ============ */
    .stFileUploader > div {
        border: 2px dashed #3da88a;
        border-radius: 12px;
        padding: 2.5rem 2rem;
        background: rgba(200, 240, 225, 0.3);
        transition: all 0.3s ease;
        min-height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .stFileUploader > div:hover {
        border-color: #2d8a6e;
        background: rgba(180, 230, 215, 0.4);
        transform: scale(1.01);
        box-shadow: 0 4px 20px rgba(45, 138, 110, 0.1);
    }
    
    /* ============ MÉTRICAS ============ */
    .stMetric {
        background: rgba(200, 240, 225, 0.7);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.15);
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }
    
    .stMetric:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        background: rgba(200, 240, 225, 0.85);
    }
    
    .stMetric > div {
        background-color: transparent !important;
    }
    
    .stMetric label {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #1a4a3a !important;
    }
    
    .stMetric div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1a6e5a !important;
    }
    
    /* ============ ABAS ============ */
    .stTabs {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(200, 240, 225, 0.7);
        border-radius: 12px;
        padding: 8px 12px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.06);
        backdrop-filter: blur(5px);
        margin-bottom: 1.2rem;
        justify-content: center;
        width: 100%;
        flex-wrap: wrap;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        color: #1a4a3a;
        white-space: nowrap;
        min-width: 100px;
        text-align: center;
        justify-content: center;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(45, 138, 110, 0.12);
        color: #1a6e5a;
        transform: translateY(-1px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2d8a6e, #3da88a);
        color: white !important;
        font-weight: 600;
        box-shadow: 0 2px 15px rgba(45, 138, 110, 0.25);
        transform: translateY(-1px);
    }
    
    /* ============ CAIXAS DE INFORMAÇÃO ============ */
    .info-box {
        background: rgba(200, 240, 225, 0.45);
        border-left: 4px solid #3da88a;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #1a4a3a;
        font-size: 1rem;
        box-shadow: 0 2px 10px rgba(45, 138, 110, 0.05);
    }
    
    .config-box {
        background: rgba(200, 240, 225, 0.35);
        border: 1px solid rgba(100, 200, 170, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
    
    .config-title {
        font-weight: 600;
        color: #1a4a3a;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a4a3a;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3da88a;
        display: inline-block;
    }
    
    /* ============ INPUTS ============ */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 1px solid rgba(100, 200, 170, 0.3);
        padding: 0.6rem 0.8rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: rgba(255,255,255,0.85);
        color: #1a4a3a !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #3da88a;
        box-shadow: 0 0 0 3px rgba(45, 138, 110, 0.1);
        color: #1a4a3a !important;
    }
    
    .stSelectbox {
        margin-bottom: 1rem;
        min-height: 50px;
    }
    
    .stSelectbox > div {
        min-height: 45px;
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 1px solid rgba(100, 200, 170, 0.3);
        padding: 0.4rem 0.8rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: rgba(255,255,255,0.85);
        color: #1a4a3a !important;
        min-height: 42px;
    }
    
    .stSelectbox > div > div > div {
        color: #1a4a3a !important;
        font-size: 0.95rem !important;
    }
    
    .stSelectbox > div > div:focus {
        border-color: #3da88a;
        box-shadow: 0 0 0 3px rgba(45, 138, 110, 0.1);
    }
    
    .stSelectbox > div > div > div > div {
        color: #1a4a3a !important;
        font-size: 0.9rem !important;
        padding: 6px 10px !important;
    }
    
    .stSelectbox > div > div > div > div:hover {
        background-color: rgba(200, 240, 225, 0.3) !important;
    }
    
    .stSelectbox > div > div > div:first-child {
        min-height: 32px;
        display: flex;
        align-items: center;
    }
    
    .stCheckbox > label {
        font-weight: 600;
        color: #1a4a3a;
        font-size: 0.95rem;
    }
    
    /* ============ DATAFRAMES ============ */
    .stDataFrame {
        border-radius: 12px;
        border: 1px solid rgba(100, 200, 170, 0.2);
        overflow: hidden;
        background: rgba(255,255,255,0.85);
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        width: 100%;
    }
    
    .stDataFrame > div {
        border-radius: 12px;
        width: 100%;
    }
    
    .stDataFrame table {
        width: 100% !important;
    }
    
    footer {
        visibility: hidden;
    }
    
    /* ============ RODAPÉ ============ */
    .footer {
        text-align: center;
        font-size: 12px;
        color: rgba(255,255,255,0.6);
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin-top: 1.5rem;
        width: 100%;
    }
    
    .footer b {
        color: rgba(255,255,255,0.8);
    }
    
    /* ============ FILE INFO ============ */
    .file-info {
        background: linear-gradient(135deg, #2d8a6e, #3da88a);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 2px 15px rgba(45, 138, 110, 0.25);
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
    }
    
    .file-info span {
        background: rgba(255,255,255,0.2);
        padding: 0.2rem 0.8rem;
        border-radius: 16px;
        font-weight: 400;
        font-size: 0.9rem;
    }
    
    .media-mensal-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a6e5a;
        margin-bottom: 0.8rem;
    }
    
    .row-widget {
        width: 100% !important;
    }
    
    .stColumns {
        width: 100% !important;
    }
    
    /* ============ SCROLLBAR ============ */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(200, 240, 225, 0.15);
        border-radius: 8px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3da88a;
        border-radius: 8px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #2d8a6e;
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

if "latitude" not in st.session_state:
    st.session_state["latitude"] = -16.0

if "ia_ativada" not in st.session_state:
    st.session_state["ia_ativada"] = False

if "gemini_api_key" not in st.session_state:
    st.session_state["gemini_api_key"] = ""

# =============================================================================
# TÍTULO PRINCIPAL - AZUL DESTAQUE
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

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Início",
    "📂 Importação",
    "🧹 Tratamento",
    "📅 Consolidação",
    "📈 Estatística",
    "🤖 IA",
    "📥 Exportação"
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
    
    observacoes = st.text_area("📝 Observações", height=80)
    
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
        4️⃣ Análise Estatística → 5️⃣ IA → 6️⃣ Exportar
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
            df2[col] = df2[col].astype(str)
            df2[col] = df2[col].str.strip()
            df2[col] = df2[col].str.replace(',', '.', regex=False)
            df2[col] = df2[col].str.replace(r'[^0-9.\-]', '', regex=True)
            df2[col] = df2[col].replace('', np.nan)
            df2[col] = df2[col].replace('nan', np.nan)
            df2[col] = df2[col].replace('None', np.nan)
            df2[col] = df2[col].replace('null', np.nan)
            df2[col] = pd.to_numeric(df2[col], errors='coerce')
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

def identificar_coluna_data(df):
    palavras = ["data", "date", "datetime", "tempo", "timestamp"]
    for col in df.columns:
        nome = str(col).lower()
        if any(p in nome for p in palavras):
            return col
    return None

# =============================================================================
# ABA 1 - IMPORTAÇÃO
# =============================================================================

with tab1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📂 Importação Inteligente</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="config-box">
        <div class="config-title">📁 Upload do Arquivo INMET</div>
        <div style="color: #1a4a3a; font-size: 1rem;">
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
        <div style="text-align: center; padding: 1.5rem; color: #5a8a7a;">
            <span style="font-size: 2.5rem;">📁</span>
            <p style="font-size: 1.1rem; margin-top: 0.3rem;">Nenhum arquivo selecionado</p>
            <p style="font-size: 0.9rem;">Arraste ou clique para fazer upload</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FUNÇÕES DE TRATAMENTO
# =============================================================================

def obter_colunas_numericas(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

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
            <p style="margin: 0; color: #1a4a3a; font-size: 0.95rem;">Técnica para preenchimento de falhas:</p>
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
        "Colunas Numéricas": len(df.select_dtypes(include=[np.number]).columns)
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
            
            st.markdown("---")
            st.markdown("### 🔍 Diagnóstico dos Dados")
            colunas_num = df_consolidado.select_dtypes(include=[np.number]).columns.tolist()
            if colunas_num:
                st.success(f"✅ Encontradas {len(colunas_num)} colunas numéricas")
                st.write("**Colunas numéricas:**", ", ".join(colunas_num[:10]))
            else:
                st.warning("⚠️ Nenhuma coluna numérica encontrada. Verifique se os dados foram carregados corretamente.")
                st.write("**Colunas disponíveis:**", df_consolidado.columns.tolist())
            
            st.success("✅ Base pronta para Análise Estatística!")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# ABA 4 - ESTATÍSTICA
# =============================================================================

with tab4:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Análise Estatística</div>', unsafe_allow_html=True)
    
    if "df_consolidado" not in st.session_state or st.session_state["df_consolidado"] is None:
        st.warning("⚠️ Primeiro consolide os dados na aba 'Consolidação'.")
    else:
        df = st.session_state["df_consolidado"]
        numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        col_data = identificar_coluna_data(df)
        
        if not numericas:
            st.error("❌ Nenhuma coluna numérica encontrada para análise.")
            st.info("💡 Verifique se os dados foram carregados corretamente.")
            
            st.markdown("### 📋 Estrutura do DataFrame")
            st.write(f"**Total de colunas:** {len(df.columns)}")
            st.write(f"**Total de linhas:** {len(df)}")
            st.markdown("**Colunas disponíveis:**")
            st.write(df.columns.tolist())
            st.markdown("**Tipos das colunas:**")
            st.dataframe(pd.DataFrame({"Coluna": df.columns, "Tipo": df.dtypes.astype(str)}))
        else:
            st.success(f"✅ Encontradas {len(numericas)} colunas numéricas para análise.")
            
            # ============================================================
            # MÉDIA POR MÊS - DESTAQUE PRINCIPAL
            # ============================================================
            if col_data is not None:
                st.markdown('<p class="media-mensal-title">📊 Média Mensal (Todos os anos combinados)</p>', unsafe_allow_html=True)
                st.markdown("Média de cada variável para cada mês do ano (ex: média de todos os janeiros, todos os fevereiros, etc.)")
                
                df[col_data] = pd.to_datetime(df[col_data], errors='coerce')
                
                var_mensal = st.selectbox(
                    "Selecione a variável para análise mensal",
                    numericas,
                    key="var_mensal"
                )
                
                if var_mensal:
                    media_mensal = calcular_media_por_mes(df, col_data, var_mensal)
                    
                    if media_mensal is not None and not media_mensal.empty:
                        st.markdown(f"**📈 Média Mensal - {var_mensal}**")
                        st.dataframe(media_mensal, use_container_width=True)
                        
                        st.markdown(f"**📊 Gráfico da Média Mensal - {var_mensal}**")
                        st.bar_chart(media_mensal.set_index('Mes_Nome')['Media'], use_container_width=True)
                        
                        st.markdown("**📋 Estatísticas da Média Mensal**")
                        stats = media_mensal['Media'].describe()
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mínimo", round(stats['min'], 2) if not pd.isna(stats['min']) else "-")
                        with col2:
                            st.metric("Máximo", round(stats['max'], 2) if not pd.isna(stats['max']) else "-")
                        with col3:
                            st.metric("Média", round(stats['mean'], 2) if not pd.isna(stats['mean']) else "-")
                        with col4:
                            st.metric("Desvio Padrão", round(stats['std'], 2) if not pd.isna(stats['std']) else "-")
                    else:
                        st.warning("⚠️ Nenhum dado disponível para análise.")
            else:
                st.info("ℹ️ Nenhuma coluna de data encontrada para calcular médias mensais.")
            
            st.markdown("---")
            st.markdown("### 📊 Selecione o tipo de análise estatística")
            
            tipo_analise = st.selectbox(
                "Escolha a análise estatística",
                [
                    "📊 Estatísticas Descritivas",
                    "📈 ANOVA (Análise de Variância)",
                    "📉 Teste t (Comparação de 2 grupos)",
                    "📉 Correlação",
                    "📋 Estatísticas Completas"
                ]
            )
            
            # ============================================================
            # ESTATÍSTICAS DESCRITIVAS
            # ============================================================
            if tipo_analise == "📊 Estatísticas Descritivas":
                st.markdown("#### Estatísticas Descritivas")
                
                coluna_desc = st.selectbox(
                    "Selecione a variável",
                    numericas,
                    key="desc_var"
                )
                
                if coluna_desc:
                    stats_dict = calcular_estatisticas_descritivas(df, coluna_desc)
                    if stats_dict:
                        df_stats = pd.DataFrame({
                            "Métrica": list(stats_dict.keys()),
                            "Valor": [round(v, 4) if isinstance(v, (int, float)) else v for v in stats_dict.values()]
                        })
                        st.dataframe(df_stats, use_container_width=True)
                        
                        st.markdown("#### 📊 Distribuição da Variável")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Histograma**")
                            hist_data = df[coluna_desc].dropna()
                            if len(hist_data) > 0:
                                bins = np.histogram_bin_edges(hist_data, bins='auto')
                                hist_counts = np.histogram(hist_data, bins=bins)[0]
                                hist_df = pd.DataFrame({
                                    'Intervalo': [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins)-1)],
                                    'Frequência': hist_counts
                                })
                                st.bar_chart(hist_df.set_index('Intervalo')['Frequência'], use_container_width=True)
                        
                        with col2:
                            st.markdown("**Boxplot (Resumo)**")
                            stats = df[coluna_desc].describe()
                            st.metric("📉 Mínimo", round(stats['min'], 2))
                            st.metric("📊 Q1 (25%)", round(stats['25%'], 2))
                            st.metric("📈 Mediana", round(stats['50%'], 2))
                            st.metric("📊 Q3 (75%)", round(stats['75%'], 2))
                            st.metric("📈 Máximo", round(stats['max'], 2))
            
            # ============================================================
            # ANOVA
            # ============================================================
            elif tipo_analise == "📈 ANOVA (Análise de Variância)":
                st.markdown("#### ANOVA - Análise de Variância")
                st.markdown("Compara as médias entre diferentes grupos")
                
                coluna_anova = st.selectbox(
                    "Selecione a variável resposta (numérica)",
                    numericas,
                    key="anova_var"
                )
                
                colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if 'Ano' in df.columns:
                    colunas_categoricas.append('Ano')
                if 'Mes' in df.columns:
                    colunas_categoricas.append('Mes')
                
                if not colunas_categoricas:
                    st.warning("Nenhuma coluna categórica encontrada para agrupamento.")
                else:
                    grupo_anova = st.selectbox(
                        "Selecione a coluna para agrupar (fator)",
                        colunas_categoricas,
                        key="anova_grupo"
                    )
                    
                    if coluna_anova and grupo_anova:
                        if st.button("🔬 Executar ANOVA", use_container_width=True):
                            with st.spinner("Calculando ANOVA..."):
                                resultado, erro = calcular_anova_manual(df, coluna_anova, grupo_anova)
                                
                                if erro:
                                    st.error(f"❌ {erro}")
                                else:
                                    st.success("✅ ANOVA calculada com sucesso!")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("📊 F-Statistic", round(resultado['f_statistic'], 4))
                                    with col2:
                                        st.metric("📈 P-Value", round(resultado['p_value'], 4))
                                    with col3:
                                        st.metric("📊 GL", f"{resultado['df_between']}/{resultado['df_within']}")
                                    
                                    if resultado['significativo']:
                                        st.success("✅ Resultado SIGNIFICATIVO (p < 0.05) - Há diferença entre os grupos")
                                    else:
                                        st.info("ℹ️ Resultado NÃO SIGNIFICATIVO (p >= 0.05) - Não há diferença entre os grupos")
                                    
                                    st.markdown("#### Grupos Analisados")
                                    st.write(f"**{len(resultado['grupos'])} grupos:** {', '.join(map(str, resultado['grupos']))}")
                                    
                                    if resultado['tukey'] is not None:
                                        st.markdown("#### 📊 Tukey HSD - Comparações Múltiplas")
                                        df_tukey = pd.DataFrame(resultado['tukey'])
                                        st.dataframe(df_tukey, use_container_width=True)
            
            # ============================================================
            # TESTE T
            # ============================================================
            elif tipo_analise == "📉 Teste t (Comparação de 2 grupos)":
                st.markdown("#### Teste t - Comparação de 2 grupos")
                
                coluna_t = st.selectbox(
                    "Selecione a variável (numérica)",
                    numericas,
                    key="t_var"
                )
                
                colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if 'Ano' in df.columns:
                    colunas_categoricas.append('Ano')
                if 'Mes' in df.columns:
                    colunas_categoricas.append('Mes')
                
                colunas_2_grupos = []
                for col in colunas_categoricas:
                    if len(df[col].dropna().unique()) == 2:
                        colunas_2_grupos.append(col)
                
                if not colunas_2_grupos:
                    st.warning("Nenhuma coluna com exatamente 2 grupos encontrada.")
                else:
                    grupo_t = st.selectbox(
                        "Selecione a coluna para agrupar (2 grupos)",
                        colunas_2_grupos,
                        key="t_grupo"
                    )
                    
                    if coluna_t and grupo_t:
                        if st.button("🔬 Executar Teste t", use_container_width=True):
                            with st.spinner("Calculando teste t..."):
                                resultado, erro = calcular_teste_t_manual(df, coluna_t, grupo_t)
                                
                                if erro:
                                    st.error(f"❌ {erro}")
                                else:
                                    st.success("✅ Teste t calculado com sucesso!")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("📊 t-Statistic", round(resultado['t_statistic'], 4))
                                    with col2:
                                        st.metric("📈 P-Value", round(resultado['p_value'], 4))
                                    
                                    if resultado['significativo']:
                                        st.success("✅ Resultado SIGNIFICATIVO (p < 0.05) - Há diferença entre os grupos")
                                    else:
                                        st.info("ℹ️ Resultado NÃO SIGNIFICATIVO (p >= 0.05) - Não há diferença entre os grupos")
                                    
                                    st.markdown("#### Médias dos Grupos")
                                    st.metric(f"{resultado['grupo1']}", round(resultado['media1'], 4))
                                    st.metric(f"{resultado['grupo2']}", round(resultado['media2'], 4))
            
            # ============================================================
            # CORRELAÇÃO
            # ============================================================
            elif tipo_analise == "📉 Correlação":
                st.markdown("#### Análise de Correlação")
                
                if len(numericas) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        var1_corr = st.selectbox("Variável 1", numericas, key="corr1")
                    with col2:
                        var2_corr = st.selectbox("Variável 2", [v for v in numericas if v != var1_corr], key="corr2")
                    
                    if var1_corr and var2_corr:
                        if st.button("📊 Calcular Correlação", use_container_width=True):
                            corr = calcular_correlacao(df, var1_corr, var2_corr)
                            if corr is not None:
                                st.metric(
                                    f"Correlação entre {var1_corr} e {var2_corr}",
                                    round(corr, 4),
                                    delta=f"{'Positiva' if corr > 0 else 'Negativa' if corr < 0 else 'Neutra'}"
                                )
                                
                                if abs(corr) >= 0.8:
                                    st.success("💪 Correlação Forte")
                                elif abs(corr) >= 0.5:
                                    st.info("📊 Correlação Moderada")
                                elif abs(corr) >= 0.3:
                                    st.warning("📉 Correlação Fraca")
                                else:
                                    st.info("🔍 Correlação Muito Fraca ou Inexistente")
                                
                                st.markdown("#### 📊 Gráfico de Dispersão")
                                df_scatter = df[[var1_corr, var2_corr]].dropna()
                                if len(df_scatter) > 0:
                                    st.scatter_chart(df_scatter, x=var1_corr, y=var2_corr, use_container_width=True)
                                
                                st.markdown("#### 📊 Matriz de Correlação (todas as variáveis)")
                                corr_matrix = df[numericas].corr()
                                st.dataframe(corr_matrix, use_container_width=True)
                else:
                    st.warning("Precisa de pelo menos 2 variáveis numéricas para correlação.")
            
            # ============================================================
            # ESTATÍSTICAS COMPLETAS
            # ============================================================
            else:
                st.markdown("#### 📋 Estatísticas Completas")
                st.markdown("Estatísticas descritivas para todas as variáveis numéricas")
                
                todas_estatisticas = []
                for col in numericas:
                    stats = calcular_estatisticas_descritivas(df, col)
                    if stats:
                        stats['Variável'] = col
                        todas_estatisticas.append(stats)
                
                if todas_estatisticas:
                    df_completo = pd.DataFrame(todas_estatisticas)
                    cols = ['Variável'] + [c for c in df_completo.columns if c != 'Variável']
                    df_completo = df_completo[cols]
                    st.dataframe(df_completo, use_container_width=True)
                    
                    csv_completo = df_completo.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Baixar Estatísticas Completas",
                        data=csv_completo,
                        file_name="estatisticas_completas.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    st.session_state["estatisticas"] = df_completo
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# IA - PREPARAÇÃO DE CONTEXTO
# =============================================================================

GEMINI_API_KEY = "SUA_CHAVE_API_AQUI"

def gerar_contexto_automatico(df):
    numericas = df.select_dtypes(include=[np.number])
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
    "📋 Resumo Executivo": """
Produza um resumo executivo simples e objetivo.
Explique os resultados em linguagem acessível.
"""
}

def consultar_ia(prompt_usuario, contexto):
    try:
        api_key = GEMINI_API_KEY
        if not api_key or api_key == "SUA_CHAVE_API_AQUI":
            return "⚠️ Chave API do Gemini não configurada."
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        payload = {"contents": [{"parts": [{"text": contexto + "\n\n" + prompt_usuario}]}]}
        resposta = requests.post(f"{url}?key={api_key}", headers={"Content-Type": "application/json"}, json=payload, timeout=120)
        if resposta.status_code == 200:
            dados = resposta.json()
            return dados["candidates"][0]["content"]["parts"][0]["text"]
        return f"❌ Erro: {resposta.status_code}"
    except Exception as erro:
        return f"❌ Erro: {erro}"

# =============================================================================
# ABA 5 - IA
# =============================================================================

with tab5:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 Inteligência Artificial</div>', unsafe_allow_html=True)
    
    ia_ativada = st.checkbox("Ativar IA Gemini", value=st.session_state["ia_ativada"])
    st.session_state["ia_ativada"] = ia_ativada
    
    if "df_consolidado" not in st.session_state or st.session_state["df_consolidado"] is None:
        st.warning("⚠️ Consolide os dados primeiro.")
    else:
        if ia_ativada:
            df_base = st.session_state["df_consolidado"]
            
            tipo_relatorio = st.selectbox("Tipo de Relatório", list(PROMPTS.keys()))
            pergunta = st.text_area("❓ Pergunta adicional", height=80)
            
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
            st.info("ℹ️ Ative a IA Gemini para gerar relatórios.")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FUNÇÕES DE EXPORTAÇÃO
# =============================================================================

def gerar_zip_completo():
    memoria = BytesIO()
    with zipfile.ZipFile(memoria, "w", zipfile.ZIP_DEFLATED) as zipf:
        if "df_consolidado" in st.session_state and st.session_state["df_consolidado"] is not None:
            zipf.writestr("dados_consolidados.csv", st.session_state["df_consolidado"].to_csv(index=False))
        if "estatisticas" in st.session_state and st.session_state["estatisticas"] is not None:
            zipf.writestr("estatisticas.csv", st.session_state["estatisticas"].to_csv(index=False))
        if "relatorio_ia" in st.session_state and st.session_state["relatorio_ia"] is not None:
            zipf.writestr("relatorio_ia.txt", st.session_state["relatorio_ia"])
    memoria.seek(0)
    return memoria

# =============================================================================
# ABA 6 - EXPORTAÇÃO
# =============================================================================

with tab6:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📥 Exportação</div>', unsafe_allow_html=True)
    
    if "df_consolidado" not in st.session_state or st.session_state["df_consolidado"] is None:
        st.warning("⚠️ Não há dados para exportar.")
    else:
        zip_file = gerar_zip_completo()
        st.download_button(
            "📦 Baixar Pacote Completo (ZIP)",
            data=zip_file,
            file_name="projeto_completo.zip",
            mime="application/zip",
            use_container_width=True
        )
        
        csv_consolidado = st.session_state["df_consolidado"].to_csv(index=False).encode('utf-8')
        st.download_button(
            "📄 Dados Consolidados (CSV)",
            data=csv_consolidado,
            file_name="dados_consolidados.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# RODAPÉ
# =============================================================================

st.markdown("""
<div class="footer">
    <b>📊 AgroClimate AI</b><br>
    © 2026 AgroClimate AI — Versão Acadêmica Experimental
</div>
""", unsafe_allow_html=True)
