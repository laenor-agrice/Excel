import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import datetime
import time
import requests
import re
import warnings

# ============================================================================
# CONFIGURAÇÕES GERAIS E BANIMENTO DE AVISOS
# ============================================================================
warnings.filterwarnings('ignore')
alt.data_transformers.disable_max_rows() # Permite ao Altair plotar bases massivas

st.set_page_config(
    page_title="NexusData Pro V2 - Core Analytical",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS FUTURISTA & TEMA ALTAIR (CYBERPUNK NEON)
# ============================================================================
def aplicar_css_futurista():
    """Injeta estilos CSS avançados para interface Sci-Fi/Dark Theme."""
    st.markdown("""
    <style>
        :root {
            --neon-blue: #00f3ff;
            --neon-pink: #ff00ea;
            --neon-green: #39ff14;
            --dark-bg: #090a0f;
            --panel-bg: #11141e;
        }
        .stApp {
            background-color: var(--dark-bg);
            background-image: 
                linear-gradient(rgba(0, 243, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 243, 255, 0.03) 1px, transparent 1px);
            background-size: 30px 30px;
            color: #d1d5db;
            font-family: 'Consolas', 'Courier New', monospace;
        }
        h1, h2, h3, h4 {
            color: var(--neon-blue) !important;
            text-shadow: 0 0 10px rgba(0, 243, 255, 0.5);
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: var(--panel-bg);
            border-bottom: 2px solid rgba(0, 243, 255, 0.2);
            padding: 10px 10px 0 10px;
            gap: 5px;
            border-radius: 8px 8px 0 0;
        }
        .stTabs [data-baseweb="tab"] {
            color: #6b7280;
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            transition: 0.3s;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(0, 243, 255, 0.05);
            color: var(--neon-blue) !important;
            border-top: 2px solid var(--neon-blue);
            border-left: 1px solid rgba(0, 243, 255, 0.3);
            border-right: 1px solid rgba(0, 243, 255, 0.3);
            box-shadow: 0 -5px 15px rgba(0, 243, 255, 0.1);
        }
        .stButton>button {
            background: transparent;
            border: 1px solid var(--neon-blue);
            color: var(--neon-blue);
            text-transform: uppercase;
            font-weight: bold;
            letter-spacing: 1px;
            border-radius: 0px;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: var(--neon-blue);
            color: #000;
            box-shadow: 0 0 15px var(--neon-blue);
        }
        .stDataFrame {
            border: 1px solid rgba(0, 243, 255, 0.3);
            box-shadow: 0 0 20px rgba(0,0,0,0.8);
        }
        hr {
            border-color: rgba(255, 0, 234, 0.3);
        }
        [data-testid="stMetricValue"] {
            color: var(--neon-pink) !important;
            text-shadow: 0 0 10px rgba(255, 0, 234, 0.4);
        }
        [data-testid="stSidebar"] {
            background-color: var(--panel-bg);
            border-right: 1px solid rgba(0, 243, 255, 0.2);
        }
        .stSuccess, .stInfo, .stWarning, .stError {
            background-color: var(--panel-bg) !important;
            border-radius: 0 !important;
        }
        .stSuccess { border-left: 4px solid var(--neon-green) !important; color: var(--neon-green) !important; }
        .stInfo { border-left: 4px solid var(--neon-blue) !important; color: var(--neon-blue) !important; }
        .stWarning { border-left: 4px solid #facc15 !important; color: #facc15 !important; }
        .stError { border-left: 4px solid #ef4444 !important; color: #ef4444 !important; }
    </style>
    """, unsafe_allow_html=True)

aplicar_css_futurista()

def cyberpunk_altair_theme():
    """Configura o Altair para herdar a identidade visual Sci-Fi."""
    font = "Consolas"
    label_color = "#00f3ff"
    return {
        "config": {
            "background": "transparent",
            "title": {"font": font, "color": "#ff00ea", "fontSize": 16},
            "axis": {
                "labelFont": font,
                "titleFont": font,
                "labelColor": label_color,
                "titleColor": label_color,
                "gridColor": "rgba(0, 243, 255, 0.1)",
                "domainColor": label_color,
                "tickColor": label_color
            },
            "legend": {
                "labelFont": font,
                "titleFont": font,
                "labelColor": label_color,
                "titleColor": label_color
            },
            "view": {"stroke": "transparent"}
        }
    }

alt.themes.register("cyberpunk", cyberpunk_altair_theme)
alt.themes.enable("cyberpunk")

# ============================================================================
# GERENCIAMENTO DE ESTADOS GLOBAIS
# ============================================================================
if 'usuario' not in st.session_state:
    st.session_state.usuario = {}
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'logs' not in st.session_state:
    st.session_state.logs = []

def registrar_log(mensagem):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {mensagem}")

# ============================================================================
# MOTORES DE PROCESSAMENTO DE DADOS (BASEADOS NO CÓDIGO ORIGINAL DO USUÁRIO)
# ============================================================================
def detectar_formato_e_ler(arquivo_bytes, filename):
    """Lê arquivos TXT/CSV/XLSX aplicando detecção heurística de delimitadores."""
    if filename.endswith(('.xlsx', '.xls')):
        return pd.read_excel(arquivo_bytes)
        
    try:
        conteudo = arquivo_bytes.getvalue().decode('utf-8', errors='ignore')[:5000]
        linhas = conteudo.split('\n')
    except:
        return pd.read_csv(arquivo_bytes)

    delimitador = ';'
    if linhas:
        for delim in [';', ',', '\t', '|']:
            if linhas[0] and delim in linhas[0]:
                delimitador = delim
                break
                
    arquivo_bytes.seek(0)
    try:
        df = pd.read_csv(arquivo_bytes, sep=delimitador, on_bad_lines='skip')
        return df
    except Exception as e:
        st.error(f"Falha na heurística principal. Tentando fallback. Erro: {e}")
        arquivo_bytes.seek(0)
        return pd.read_csv(arquivo_bytes, engine='python')

def tratar_outliers_iqr(df, fator=3.0):
    """Aplica Método IQR robusto (presente na arquitetura base do usuário)."""
    df_trat = df.copy()
    colunas_numericas = df_trat.select_dtypes(include=[np.number]).columns
    substituicoes = 0
    for col in colunas_numericas:
        Q1 = df_trat[col].quantile(0.25)
        Q3 = df_trat[col].quantile(0.75)
        IQR = Q3 - Q1
        inf = Q1 - fator * IQR
        sup = Q3 + fator * IQR
        
        mask = (df_trat[col] < inf) | (df_trat[col] > sup)
        qtd = mask.sum()
        if qtd > 0:
            mediana = df_trat[col].median()
            df_trat.loc[mask, col] = mediana
            substituicoes += qtd
    return df_trat, substituicoes

# ============================================================================
# API DO GEMINI (RECONSTRUÍDA SEM IMPORTAÇÕES EXTERNAS EXCETO REQUESTS)
# ============================================================================
def motor_ia_gemini(prompt_usuario, chave_api, contexto_estatistico):
    """Realiza POST direto na API generativa do Google, garantindo estabilidade."""
    if not chave_api:
        return "Erro: Chave de API Ausente. Forneça a chave no painel lateral."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={chave_api}"
    
    prompt_completo = f"""Você é uma Inteligência Artificial analista de dados tabulares.
    Você deve ajudar um pesquisador acadêmico/cientista a interpretar estatísticas do experimento dele.
    
    ESTATÍSTICAS ATUAIS (CONTEXTO DA BASE TRATADA):
    {contexto_estatistico}
    
    PEDIDO DO PESQUISADOR:
    {prompt_usuario}
    
    Forneça uma resposta analítica, pontual e que ajude a interpretar esses dados de forma profissional.
    """
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt_completo}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2048}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=25)
        if response.status_code == 200:
            res_json = response.json()
            if "candidates" in res_json and res_json["candidates"]:
                return res_json["candidates"][0]["content"]["parts"][0]["text"]
            return "A IA não conseguiu formar uma resposta interpretável."
        else:
            return f"Erro Crítico na Conexão. Status: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Falha de Protocolo (Timeout/Conexão): {e}"

# ============================================================================
# INTERFACE PRINCIPAL E ROTEAMENTO DAS ABAS
# ============================================================================
def app_nexus():
    st.markdown("<h1>🌌 NEXUS_DATA PRO Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#00f3ff; margin-top:-15px;'>Módulo Profissional de Tratamento Tabular | Sem Plotly | Altair Core</p>", unsafe_allow_html=True)

    # ------------------------------------------------------------------------
    # BARRA LATERAL (SIDEBAR) - STATUS E CONFIGS
    # ------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🎛️ Terminal de Comando")
        if st.session_state.usuario:
            st.success(f"Logado: {st.session_state.usuario.get('Nome', 'Usuário')}")
            st.caption(f"Perfil: {st.session_state.usuario.get('Perfil', 'N/A')}")
            if st.button("Encerrar Sessão", key="btn_logout"):
                st.session_state.usuario = {}
                st.rerun()
        else:
            st.warning("Autenticação Pendente.")
            
        st.markdown("---")
        chave_gemini = st.text_input("🔑 API Key do Gemini (IA)", type="password", help="Insira sua chave API do Google para desbloquear a Aba 7")
        st.markdown("---")
        st.markdown("### 📜 System Logs")
        container_logs = st.container(height=200)
        for log in reversed(st.session_state.logs[-10:]):
            container_logs.caption(log)

    # ------------------------------------------------------------------------
    # ESTRUTURA DE ABAS
    # ------------------------------------------------------------------------
    abas = st.tabs([
        "1. Credenciais", 
        "2. Ingestão (Upload)", 
        "3. Purificação", 
        "4. Temporal", 
        "5. Experimental", 
        "6. Holografia (Gráficos)", 
        "7. IA Oráculo", 
        "8. Output"
    ])

    # ================== ABA 1: CADASTRO ==================
    with abas[0]:
        st.subheader("👤 Identificação Institucional")
        st.markdown("Sistema requer credenciamento acadêmico/profissional para operar os tensores analíticos.")
        
        with st.form("form_credenciais"):
            c1, c2 = st.columns(2)
            nome = c1.text_input("Nome do Operador *")
            cidade = c1.text_input("Sede de Operação (Cidade/Estado)")
            perfil = c2.selectbox("Cargo Acadêmico/Profissional *", 
                                  ["Estudante", "Pesquisador (Mestrado)", "Pesquisador (Doutorado)", "Professor", "Cientista"])
            
            if st.form_submit_button("Validar Credenciais"):
                if nome.strip() == "":
                    st.error("Protocolo negado: Nome obrigatório.")
                else:
                    st.session_state.usuario = {"Nome": nome, "Cidade": cidade, "Perfil": perfil}
                    registrar_log(f"Novo usuário logado: {nome}")
                    st.success("Credenciais aceitas. Prossiga para a Ingestão.")
                    time.sleep(1)
                    st.rerun()

    # ================== ABA 2: UPLOAD E PREVIEW ==================
    with abas[1]:
        st.subheader("📂 Ingestão de Matriz Tabular")
        
        if not st.session_state.usuario:
            st.warning("⚠️ Terminal bloqueado. Identifique-se na Aba 1.")
        else:
            arquivo = st.file_uploader("Formato Suportado: Excel, CSV, TXT", type=['csv', 'txt', 'xlsx', 'xls'])
            
            if arquivo:
                if st.button("Executar Decodificação"):
                    with st.spinner("Analisando topologia de bytes..."):
                        df_bruto = detectar_formato_e_ler(arquivo, arquivo.name)
                        st.session_state.df_raw = df_bruto.copy()
                        st.session_state.df_clean = df_bruto.copy()
                        registrar_log(f"Matriz '{arquivo.name}' injetada. Dimensões: {df_bruto.shape}")
                        st.success(f"Leitura concluída com sucesso. Linhas: {df_bruto.shape[0]}, Colunas: {df_bruto.shape[1]}")
            
            if st.session_state.df_raw is not None:
                st.markdown("### Visão Periférica (Top 10 Registros)")
                st.dataframe(st.session_state.df_raw.head(10), use_container_width=True)

    # ================== ABA 3: TRATAMENTO E LIMPEZA ==================
    with abas[2]:
        st.subheader("⚙️ Motor de Purificação de Dados")
        
        if st.session_state.df_clean is None:
            st.info("Aguardando injeção de dados na Aba 2.")
        else:
            df_ativo = st.session_state.df_clean
            
            c_tools, c_view = st.columns([1.2, 2.8])
            
            with c_tools:
                st.markdown("#### Filtros e Triggers")
                if st.button("1. Expurgo de Nulos (NaN)"):
                    antes = len(df_ativo)
                    df_ativo = df_ativo.dropna()
                    removidos = antes - len(df_ativo)
                    st.session_state.df_clean = df_ativo
                    registrar_log(f"Drop NaN: {removidos} linhas removidas.")
                    st.success(f"{removidos} anomalias deletadas.")
                    time.sleep(0.5); st.rerun()
                    
                if st.button("2. Remoção Colunas Vazias"):
                    antes = df_ativo.shape[1]
                    df_ativo = df_ativo.dropna(axis=1, how='all')
                    removidos = antes - df_ativo.shape[1]
                    st.session_state.df_clean = df_ativo
                    registrar_log(f"Drop Colunas: {removidos} vazias removidas.")
                    st.success(f"Matriz reestruturada. {removidos} vetores anulados.")
                    time.sleep(0.5); st.rerun()
                    
                if st.button("3. Limpeza Inconsistências (Duplicatas)"):
                    antes = len(df_ativo)
                    df_ativo = df_ativo.drop_duplicates()
                    removidos = antes - len(df_ativo)
                    st.session_state.df_clean = df_ativo
                    registrar_log(f"Drop Duplicatas: {removidos} linhas.")
                    st.success(f"{removidos} clones apagados.")
                    time.sleep(0.5); st.rerun()
                    
                if st.button("4. Interpolador de Outliers (Método IQR)"):
                    df_ativo, num_subs = tratar_outliers_iqr(df_ativo, fator=3.0)
                    st.session_state.df_clean = df_ativo
                    registrar_log(f"IQR Aplicado. {num_subs} ocorrências suavizadas pela mediana.")
                    st.success(f"Análise IQR completa. {num_subs} células ajustadas.")
                    time.sleep(1); st.rerun()

                st.markdown("---")
                if st.button("🔄 Reset Global (Retornar Origem)"):
                    st.session_state.df_clean = st.session_state.df_raw.copy()
                    registrar_log("Operador acionou Reset Global.")
                    st.rerun()

            with c_view:
                st.markdown("#### Janela de Monitoramento (12 Linhas Espaciais)")
                st.markdown("*A interface foi forçada a comprimir horizontalmente todas as colunas dentro deste container.*")
                # Exibição de 12 linhas com container apertado conforme solicitado
                st.dataframe(df_ativo.head(12), use_container_width=True)
                st.caption(f"Assinatura Atual da Matriz: {df_ativo.shape[0]} Linhas | {df_ativo.shape[1]} Variáveis")

    # ================== ABA 4: ESTATÍSTICA TEMPORAL ==================
    with abas[3]:
        st.subheader("📆 Compressão Temporal (Agrupamentos)")
        if st.session_state.df_clean is None:
            st.info("Requisito: Dados carregados.")
        else:
            df_temp = st.session_state.df_clean.copy()
            colunas_num = df_temp.select_dtypes(include=[np.number]).columns.tolist()
            
            if not colunas_num:
                st.error("O sistema não detectou variáveis escalares numéricas na matriz.")
            else:
                vetor_tempo = st.selectbox("Selecione a chave temporal (Coluna de Datas):", [""] + list(df_temp.columns))
                
                if vetor_tempo:
                    if st.button("Rodar Algoritmo Temporal"):
                        try:
                            # Conversão forçada
                            df_temp[vetor_tempo] = pd.to_datetime(df_temp[vetor_tempo], errors='coerce')
                            df_temp = df_temp.dropna(subset=[vetor_tempo])
                            df_temp.set_index(vetor_tempo, inplace=True)
                            
                            t1, t2, t3 = st.tabs(["[ Média Semanal ]", "[ Média Mensal ]", "[ Média Anual ]"])
                            
                            with t1:
                                st.dataframe(df_temp[colunas_num].resample('W').mean().dropna(how='all'), use_container_width=True)
                            with t2:
                                st.dataframe(df_temp[colunas_num].resample('ME').mean().dropna(how='all'), use_container_width=True)
                            with t3:
                                st.dataframe(df_temp[colunas_num].resample('YE').mean().dropna(how='all'), use_container_width=True)
                                
                            registrar_log("Compressão temporal concluída com sucesso.")
                        except Exception as e:
                            st.error(f"Falha ao interpretar a coluna selecionada como vetor de tempo. Detalhes: {e}")

    # ================== ABA 5: ESTATÍSTICA EXPERIMENTAL ==================
    with abas[4]:
        st.subheader("🔬 Laboratório de Estatística (Literatura)")
        if st.session_state.df_clean is None:
            st.info("Requisito: Dados carregados.")
        else:
            df_exp = st.session_state.df_clean.select_dtypes(include=[np.number])
            if df_exp.empty:
                st.error("As variáveis não contêm valores contínuos/numéricos válidos.")
            else:
                st.markdown("### Resumo Descritivo Extenso")
                st.dataframe(df_exp.describe().T, use_container_width=True)
                
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("#### Dispersão e Variação")
                    var_df = pd.DataFrame({
                        "Variância": df_exp.var(),
                        "Assimetria (Skewness)": df_exp.skew(),
                        "Curtose": df_exp.kurtosis()
                    })
                    st.dataframe(var_df, use_container_width=True)
                
                with cc2:
                    st.markdown("#### Matemática Absoluta")
                    soma_df = pd.DataFrame({
                        "Soma Total": df_exp.sum(),
                        "Erro Padrão Médio": df_exp.sem()
                    })
                    st.dataframe(soma_df, use_container_width=True)

    # ================== ABA 6: GRÁFICOS (ALTAIR & SEABORN) ==================
    with abas[5]:
        st.subheader("📈 Holografia Dinâmica e Mapas Térmicos")
        st.markdown("**Atenção:** Plotly desativado. Renderização feita puramente em Altair e Matplotlib/Seaborn de alta resolução.")
        
        if st.session_state.df_clean is None:
            st.info("Requisito: Dados carregados e purificados.")
        else:
            df_plot = st.session_state.df_clean.copy()
            col_list = df_plot.columns.tolist()
            num_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
            
            tipo_graf = st.selectbox("Motor de Renderização", [
                "Altair: Dispersão (Scatter)", 
                "Altair: Linhas", 
                "Altair: Barras", 
                "Seaborn: Matriz de Correlação"
            ])
            
            if "Altair" in tipo_graf:
                c1, c2, c3 = st.columns(3)
                x_val = c1.selectbox("Eixo X", col_list)
                y_val = c2.selectbox("Eixo Y", num_cols)
                cor_val = c3.selectbox("Agrupamento (Color)", ["Nenhum"] + col_list)
                
                if st.button("Gerar Holograma Visual"):
                    cor_code = alt.value("#00f3ff") if cor_val == "Nenhum" else alt.Color(cor_val)
                    
                    try:
                        if "Dispersão" in tipo_graf:
                            grafico = alt.Chart(df_plot).mark_circle(size=60, opacity=0.8).encode(
                                x=x_val, y=y_val, color=cor_code, tooltip=col_list
                            ).interactive()
                            
                        elif "Linhas" in tipo_graf:
                            grafico = alt.Chart(df_plot).mark_line(strokeWidth=3).encode(
                                x=x_val, y=y_val, color=cor_code, tooltip=col_list
                            ).interactive()
                            
                        elif "Barras" in tipo_graf:
                            grafico = alt.Chart(df_plot).mark_bar(opacity=0.9).encode(
                                x=x_val, y=y_val, color=cor_code, tooltip=col_list
                            ).interactive()
                            
                        st.altair_chart(grafico, use_container_width=True)
                        registrar_log(f"Gráfico '{tipo_graf}' gerado com sucesso via Altair.")
                        
                    except Exception as e:
                        st.error(f"Incompatibilidade de vetores para renderização Altair: {e}")
                        
            elif "Seaborn" in tipo_graf:
                if st.button("Gerar Mapa Térmico"):
                    if len(num_cols) < 2:
                        st.warning("Matriz exige pelo menos 2 variáveis numéricas para correlação.")
                    else:
                        st.markdown("Matriz de Correlação Analítica (Pearson)")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        # Tema escuro para matplotlib para casar com o sistema
                        plt.style.use('dark_background')
                        fig.patch.set_facecolor('#090a0f')
                        ax.set_facecolor('#11141e')
                        
                        corr = df_plot[num_cols].corr()
                        sns.heatmap(corr, annot=True, cmap="cool", fmt=".2f", ax=ax, 
                                    linecolor="#00f3ff", linewidths=1)
                        
                        st.pyplot(fig)
                        registrar_log("Heatmap de Correlação Seaborn renderizado.")

    # ================== ABA 7: INTELIGÊNCIA ARTIFICIAL ==================
    with abas[6]:
        st.subheader("🧠 Oráculo IA (Integração Gemini)")
        st.markdown("O assistente lê o panorama estatístico atual e gera insights experimentais.")
        
        if st.session_state.df_clean is None:
            st.warning("Forneça a matriz de dados nas abas anteriores.")
        else:
            pergunta = st.text_area("Descreva o que você deseja que a IA analise sobre seus dados estatísticos:",
                                    "Interprete as médias, variações e indique possíveis tendências no meu experimento.")
            
            if st.button("Acionar Rede Neural Gemini"):
                if not chave_gemini:
                    st.error("Acesso Negado: Insira a API Key do Google Gemini no painel lateral esquerdo.")
                else:
                    with st.spinner("Estabelecendo conexão neural com servidores Google..."):
                        # Extrai um resumo rápido (describe limitando colunas) para enviar de contexto
                        resumo = st.session_state.df_clean.select_dtypes(include=[np.number]).describe().to_dict()
                        resposta = motor_ia_gemini(pergunta, chave_gemini, str(resumo))
                        st.info(resposta)

    # ================== ABA 8: EXPORTAÇÃO E DOWNLOAD ==================
    with abas[7]:
        st.subheader("💾 Terminal de Saída (Output Excel)")
        
        if st.session_state.df_clean is None:
            st.info("Nenhum arquivo ativo no buffer de memória.")
        else:
            st.markdown("O pesquisador pode baixar o arquivo tratado (modificado) ou o arquivo matriz original de backup.")
            
            def compilar_excel(dataframe):
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    dataframe.to_excel(writer, index=False, sheet_name='Nexus_Data')
                return buffer.getvalue()

            c_down1, c_down2 = st.columns(2)
            
            with c_down1:
                st.markdown("#### 🟢 Baixar Matriz Tratada")
                st.markdown("Download da versão limpa pelos algoritmos da Aba 3.")
                try:
                    excel_limpo = compilar_excel(st.session_state.df_clean)
                    st.download_button("📥 DOWNLOAD NEXUS_TRATADO.XLSX", data=excel_limpo, 
                                       file_name='Nexus_Tratado.xlsx', 
                                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                except Exception as e:
                    st.error(f"Erro de IO: {e}")

            with c_down2:
                st.markdown("#### 🔴 Baixar Backup Original")
                st.markdown("Download exato de como os dados entraram no upload.")
                if st.session_state.df_raw is not None:
                    try:
                        excel_bruto = compilar_excel(st.session_state.df_raw)
                        st.download_button("📥 DOWNLOAD NEXUS_ORIGINAL.XLSX", data=excel_bruto, 
                                           file_name='Nexus_Original.xlsx', 
                                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    except Exception as e:
                        st.error(f"Erro de IO: {e}")

# Executa o sistema
if __name__ == "__main__":
    app_nexus()
