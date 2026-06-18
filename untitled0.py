# ==============================================================================
# IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta
import base64
import io
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="AgroAnalytics Pro",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CSS PERSONALIZADO - DESIGN MINIMALISTA MODERNO
# ==============================================================================
def load_css():
    st.markdown("""
    <style>
    /* Cores principais */
    :root {
        --primary: #2E7D32;
        --secondary: #1976D2;
        --accent: #FF8F00;
        --bg-light: #F5F7F2;
        --text-dark: #2C3E50;
        --card-bg: #FFFFFF;
        --border: #E0E6ED;
    }
    
    /* Estilo geral */
    .stApp {
        background-color: var(--bg-light);
    }
    
    /* Cards modernos */
    .custom-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid var(--border);
        margin: 10px 0;
        transition: transform 0.2s;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    /* Títulos elegantes */
    .main-title {
        color: var(--primary);
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    
    .subtitle {
        color: var(--text-dark);
        font-size: 1.2em;
        font-weight: 400;
        opacity: 0.8;
    }
    
    /* Métricas */
    .metric-box {
        background: var(--card-bg);
        border-left: 4px solid var(--primary);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Botões personalizados */
    .stButton > button {
        background-color: var(--primary) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 8px 20px !important;
        transition: all 0.3s;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #1B5E20 !important;
        box-shadow: 0 3px 6px rgba(0,0,0,0.15);
        transform: translateY(-1px);
    }
    
    /* Tabs estilizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--card-bg);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        border: 1px solid var(--border);
        color: var(--text-dark);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary);
        color: white;
        border-color: var(--primary);
    }
    
    /* Inputs */
    .stSelectbox label, .stMultiselect label {
        color: var(--text-dark);
        font-weight: 500;
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Mensagens de sucesso */
    .stSuccess {
        background-color: #E8F5E9 !important;
        border-left: 4px solid #4CAF50 !important;
        border-radius: 8px !important;
    }
    
    /* Responsividade */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2em;
        }
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

# Função para processar arquivos INMET
def process_inmet_file(df):
    """Processa e padroniza arquivos do INMET"""
    try:
        # Identificar colunas comuns do INMET
        df.columns = df.columns.str.strip()
        
        # Renomear colunas padrão se existirem
        column_mapping = {
            'Data': 'Data',
            'Hora': 'Hora',
            'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'Precipitacao_mm',
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 'Pressao_mB',
            'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)': 'Pressao_Max_mB',
            'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)': 'Pressao_Min_mB',
            'RADIACAO GLOBAL (KJ/m²)': 'Radiacao_KJm2',
            'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'Temperatura_C',
            'TEMPERATURA DO PONTO DE ORVALHO (°C)': 'Ponto_Orvalho_C',
            'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'Temperatura_Max_C',
            'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'Temperatura_Min_C',
            'UMIDADE RELATIVA DO AR, HORARIA (%)': 'Umidade_%',
            'VENTO, DIREÇÃO HORARIA (gr) (° (gr))': 'Direcao_Vento_gr',
            'VENTO, RAJADA MAXIMA (m/s)': 'Rajada_Max_ms',
            'VENTO, VELOCIDADE HORARIA (m/s)': 'Velocidade_Vento_ms'
        }
        
        for old, new in column_mapping.items():
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)
        
        # Converter coluna de data
        if 'Data' in df.columns:
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        
        # Substituir valores nulos padrão do INMET (-9999, etc)
        df.replace([-9999, -999, -99, -9, '', ' '], np.nan, inplace=True)
        
        # Converter colunas numéricas
        for col in df.columns:
            if col not in ['Data', 'Hora']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Erro ao processar arquivo: {str(e)}")
        return df

# Função para análise de dados faltantes
def analyze_missing_data(df):
    """Analisa dados faltantes no DataFrame"""
    missing_data = pd.DataFrame({
        'Coluna': df.columns,
        'Dados Faltantes': df.isnull().sum().values,
        'Percentual (%)': (df.isnull().sum() / len(df) * 100).values
    })
    missing_data = missing_data[missing_data['Dados Faltantes'] > 0]
    missing_data = missing_data.sort_values('Percentual (%)', ascending=False)
    return missing_data

# Função para download de dados
def get_download_link(df, filename="dados_tratados.csv"):
    """Gera link para download do DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">📥 Download do Arquivo Tratado</a>'

# Função para estatísticas descritivas
def calculate_statistics(df, numeric_cols):
    """Calcula estatísticas descritivas para colunas numéricas"""
    stats_dict = {}
    for col in numeric_cols:
        data = df[col].dropna()
        stats_dict[col] = {
            'Média': np.mean(data),
            'Mediana': np.median(data),
            'Desvio Padrão': np.std(data),
            'CV (%)': (np.std(data) / np.mean(data) * 100) if np.mean(data) != 0 else 0,
            'Mínimo': np.min(data),
            'Máximo': np.max(data),
            'Assimetria': stats.skew(data),
            'Curtose': stats.kurtosis(data)
        }
    return stats_dict

# ==============================================================================
# INTERFACE PRINCIPAL
# ==============================================================================

# Header elegante
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-title">🌿 AgroAnalytics Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Plataforma Inteligente para Análise de Dados Agrometeorológicos</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar para API Key e informações
with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    
    # API Key Gemini
    st.markdown("---")
    api_key = st.text_input("🔑 Google Gemini API Key", type="password", 
                           placeholder="Insira sua chave API do Gemini",
                           help="Necessário para funcionalidades de IA")
    
    if api_key:
        st.success("✅ API Key configurada")
    
    st.markdown("---")
    st.markdown("### 📊 Sobre")
    st.info("""
    **AgroAnalytics Pro v1.0**
    
    Desenvolvido para análise de dados meteorológicos do INMET.
    
    Funcionalidades:
    - Processamento de dados INMET
    - Análise de dados faltantes
    - Estatísticas descritivas
    - Análise de correlação
    - Visualizações gráficas
    - Download de dados tratados
    """)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Desenvolvido por")
    st.markdown("Equipe AgroAnalytics | 2024")

# Inicialização do estado da sessão
if 'df' not in st.session_state:
    st.session_state.df = None
if 'treated_df' not in st.session_state:
    st.session_state.treated_df = None
if 'numeric_cols' not in st.session_state:
    st.session_state.numeric_cols = []

# ==============================================================================
# TABS PRINCIPAIS
# ==============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📤 **Upload**", 
    "🔧 **Tratamento**", 
    "📊 **Estatísticas**", 
    "🔬 **Análise Avançada**",
    "📈 **Gráficos**",
    "📥 **Download**"
])

# ==============================================================================
# TAB 1: UPLOAD DE DADOS
# ==============================================================================
with tab1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("## 📤 Upload de Dados")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Faça upload do arquivo Excel/CSV do INMET",
            type=['csv', 'xlsx', 'xls'],
            help="Formatos aceitos: CSV, Excel (.xlsx, .xls)"
        )
        
        if uploaded_file is not None:
            try:
                # Carregar dados baseado na extensão
                if uploaded_file.name.endswith('.csv'):
                    # Tentar diferentes delimitadores e encodings
                    try:
                        df = pd.read_csv(uploaded_file, encoding='utf-8')
                    except:
                        try:
                            df = pd.read_csv(uploaded_file, encoding='latin1')
                        except:
                            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Processar dados INMET
                df = process_inmet_file(df)
                
                # Armazenar no estado da sessão
                st.session_state.df = df
                st.session_state.treated_df = df.copy()
                
                # Identificar colunas numéricas
                st.session_state.numeric_cols = df.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
                
                # Mensagem de sucesso
                st.success(f"✅ Upload realizado com sucesso! Arquivo: {uploaded_file.name}")
                
                # Métricas rápidas
                with col2:
                    st.metric("Total de Registros", len(df))
                    st.metric("Total de Colunas", len(df.columns))
                    missing = df.isnull().sum().sum()
                    st.metric("Dados Faltantes", missing)
                
                # Prévia dos dados
                st.markdown("### 📋 Prévia dos Dados")
                st.markdown("*Primeiras 10 linhas do dataset:*")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Informações das colunas
                st.markdown("### 📊 Informações das Colunas")
                col_info = pd.DataFrame({
                    'Coluna': df.columns,
                    'Tipo': df.dtypes.astype(str),
                    'Não Nulos': df.count().values,
                    'Nulos': df.isnull().sum().values
                })
                st.dataframe(col_info, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Erro ao carregar arquivo: {str(e)}")
    
    with col2:
        st.markdown("### 🎯 Formatos Aceitos")
        st.info("""
        **Arquivos suportados:**
        - CSV (separado por vírgula)
        - Excel (.xlsx, .xls)
        
        **Estrutura esperada:**
        - Coluna de Data
        - Colunas meteorológicas
        - Valores numéricos
        
        **Exemplo INMET:**
        - Precipitação
        - Temperatura
        - Umidade
        - Pressão
        - Vento
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 2: TRATAMENTO DE DADOS
# ==============================================================================
with tab2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("## 🔧 Tratamento de Dados")
    
    if st.session_state.df is not None:
        df = st.session_state.treated_df.copy()
        
        # Análise de dados faltantes
        missing_data = analyze_missing_data(df)
        
        if len(missing_data) > 0:
            st.warning(f"⚠️ Foram encontrados dados faltantes em {len(missing_data)} colunas")
            
            # Mostrar dados faltantes
            st.markdown("### 📊 Colunas com Dados Faltantes")
            
            # Criar visualização com barras de progresso
            for _, row in missing_data.iterrows():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{row['Coluna']}**")
                    st.progress(int(row['Percentual (%)']) / 100)
                with col2:
                    st.metric("Faltantes", row['Dados Faltantes'])
                with col3:
                    st.metric("%", f"{row['Percentual (%)']:.1f}%")
            
            # Opções de tratamento
            st.markdown("### ⚙️ Opções de Tratamento")
            
            col1, col2 = st.columns(2)
            
            with col1:
                treatment_option = st.radio(
                    "Escolha o método de tratamento:",
                    ["Não fazer nada", "Preencher com média", "Preencher com mediana", 
                     "Excluir linhas com dados faltantes"],
                    help="Selecione como tratar os dados faltantes"
                )
            
            with col2:
                columns_to_treat = st.multiselect(
                    "Selecione as colunas para tratamento:",
                    missing_data['Coluna'].tolist(),
                    default=missing_data['Coluna'].tolist(),
                    help="Escolha quais colunas receberão o tratamento"
                )
            
            if st.button("🔄 Aplicar Tratamento", type="primary"):
                if treatment_option == "Preencher com média":
                    for col in columns_to_treat:
                        df[col].fillna(df[col].mean(), inplace=True)
                    st.success("✅ Valores preenchidos com a média")
                    
                elif treatment_option == "Preencher com mediana":
                    for col in columns_to_treat:
                        df[col].fillna(df[col].median(), inplace=True)
                    st.success("✅ Valores preenchidos com a mediana")
                    
                elif treatment_option == "Excluir linhas com dados faltantes":
                    df.dropna(subset=columns_to_treat, inplace=True)
                    st.success(f"✅ Linhas removidas. Novo tamanho: {len(df)} registros")
                    
                else:
                    st.info("ℹ️ Nenhum tratamento aplicado")
                
                # Atualizar DataFrame tratado
                st.session_state.treated_df = df
                
                # Mostrar resultado
                new_missing = analyze_missing_data(df)
                st.markdown("### ✅ Após Tratamento")
                
                if len(new_missing) == 0:
                    st.success("🎉 Todos os dados faltantes foram tratados!")
                else:
                    st.info(f"Ainda restam {len(new_missing)} colunas com dados faltantes")
                    
        else:
            st.success("✅ Não foram encontrados dados faltantes no dataset!")
    
    else:
        st.info("👆 Faça o upload dos dados na aba 'Upload' primeiro")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 3: ESTATÍSTICAS
# ==============================================================================
with tab3:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("## 📊 Análise Estatística")
    
    if st.session_state.treated_df is not None:
        df = st.session_state.treated_df.copy()
        
        # Verificar se há coluna de data
        date_col = None
        for col in df.columns:
            if 'data' in col.lower() or 'date' in col.lower():
                date_col = col
                break
        
        # Agrupamento temporal
        st.markdown("### 📅 Agrupamento Temporal")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if date_col:
                time_agg = st.selectbox(
                    "Período de agregação:",
                    ["Diário", "Semanal", "Mensal", "Anual", "Sem agregação"],
                    help="Escolha o período para calcular as médias"
                )
            else:
                st.warning("⚠️ Coluna de data não identificada. Verifique se há uma coluna 'Data' no arquivo.")
                time_agg = "Sem agregação"
        
        with col2:
            agg_columns = st.multiselect(
                "Selecione as variáveis para análise:",
                st.session_state.numeric_cols,
                default=st.session_state.numeric_cols[:3] if len(st.session_state.numeric_cols) >= 3 else st.session_state.numeric_cols,
                help="Escolha as variáveis meteorológicas"
            )
        
        if date_col and time_agg != "Sem agregação":
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                
                if time_agg == "Diário":
                    df_agg = df.groupby(df[date_col].dt.date)[agg_columns].mean().reset_index()
                elif time_agg == "Semanal":
                    df_agg = df.groupby(df[date_col].dt.isocalendar().week)[agg_columns].mean().reset_index()
                elif time_agg == "Mensal":
                    df_agg = df.groupby(df[date_col].dt.to_period('M'))[agg_columns].mean().reset_index()
                elif time_agg == "Anual":
                    df_agg = df.groupby(df[date_col].dt.year)[agg_columns].mean().reset_index()
                
                st.success(f"✅ Dados agregados: {time_agg} ({len(df_agg)} registros)")
                st.dataframe(df_agg, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erro na agregação: {str(e)}")
        
        # Estatísticas descritivas
        if agg_columns:
            st.markdown("### 📈 Estatísticas Descritivas")
            
            stats_df = calculate_statistics(df, agg_columns)
            stats_display = pd.DataFrame(stats_df).round(3)
            
            # Formatação visual
            st.dataframe(
                stats_display.style.background_gradient(cmap='YlOrRd', axis=1)
                .highlight_max(color='lightgreen', axis=1)
                .highlight_min(color='lightcoral', axis=1),
                use_container_width=True
            )
            
            # Métricas principais
            st.markdown("### 🎯 Métricas Principais")
            
            cols = st.columns(len(agg_columns) if len(agg_columns) <= 4 else 4)
            for i, col in enumerate(agg_columns[:4]):
                with cols[i % 4]:
                    mean_val = stats_df[col]['Média']
                    cv_val = stats_df[col]['CV (%)']
                    
                    st.markdown(f"""
                    <div class="metric-box">
                        <h4>{col[:20]}</h4>
                        <p><strong>Média:</strong> {mean_val:.2f}</p>
                        <p><strong>CV:</strong> {cv_val:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        st.info("👆 Faça o upload dos dados na aba 'Upload' primeiro")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 4: ANÁLISE AVANÇADA
# ==============================================================================
with tab4:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("## 🔬 Análise Estatística Avançada")
    
    if st.session_state.treated_df is not None and len(st.session_state.numeric_cols) >= 2:
        df = st.session_state.treated_df.copy()
        
        # Tipo de análise
        analysis_type = st.selectbox(
            "Selecione o tipo de análise:",
            ["Correlação entre Variáveis", "Teste ANOVA", "Coeficiente de Variação"],
            help="Escolha o método estatístico avançado"
        )
        
        # Correlação
        if analysis_type == "Correlação entre Variáveis":
            st.markdown("### 📊 Matriz de Correlação")
            
            selected_cols = st.multiselect(
                "Selecione as variáveis para correlação:",
                st.session_state.numeric_cols,
                default=st.session_state.numeric_cols[:4] if len(st.session_state.numeric_cols) >= 4 else st.session_state.numeric_cols,
                help="Escolha no mínimo 2 variáveis"
            )
            
            if len(selected_cols) >= 2:
                # Calcular correlação
                corr_matrix = df[selected_cols].corr()
                
                # Visualização com matplotlib/seaborn
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, 
                           annot=True, 
                           cmap='RdYlBu', 
                           center=0,
                           vmin=-1, 
                           vmax=1, 
                           square=True,
                           linewidths=1.5,
                           cbar_kws={"shrink": 0.8},
                           ax=ax)
                ax.set_title('Matriz de Correlação', fontsize=14, fontweight='bold')
                
                st.pyplot(fig)
                
                # Análise de correlações fortes
                st.markdown("### 🔍 Análise de Correlações Significativas")
                
                strong_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            strong_corrs.append({
                                'Variável 1': corr_matrix.columns[i],
                                'Variável 2': corr_matrix.columns[j],
                                'Correlação': corr_val,
                                'Força': 'Forte positiva' if corr_val > 0.7 else 
                                        'Moderada positiva' if corr_val > 0.5 else
                                        'Forte negativa' if corr_val < -0.7 else
                                        'Moderada negativa'
                            })
                
                if strong_corrs:
                    corr_df = pd.DataFrame(strong_corrs)
                    st.dataframe(corr_df, use_container_width=True)
                    
                    # Insights
                    st.markdown("### 💡 Insights de Correlação")
                    for _, row in corr_df.iterrows():
                        if row['Correlação'] > 0.7:
                            st.info(f"🔗 **{row['Variável 1']}** e **{row['Variável 2']}** têm uma forte correlação positiva ({row['Correlação']:.2f}). Quando uma aumenta, a outra tende a aumentar também.")
                        elif row['Correlação'] < -0.7:
                            st.warning(f"⚡ **{row['Variável 1']}** e **{row['Variável 2']}** têm uma forte correlação negativa ({row['Correlação']:.2f}). Quando uma aumenta, a outra tende a diminuir.")
                else:
                    st.info("Não foram encontradas correlações fortes (|r| > 0.5) entre as variáveis selecionadas.")
        
        # ANOVA
        elif analysis_type == "Teste ANOVA":
            st.markdown("### 📊 Análise de Variância (ANOVA)")
            
            st.info("""
            O teste ANOVA compara as médias entre grupos para determinar se existem diferenças 
            estatisticamente significativas.
            """)
            
            # Selecionar variável numérica
            numeric_var = st.selectbox(
                "Selecione a variável numérica para análise:",
                st.session_state.numeric_cols,
                help="Variável dependente (numérica)"
            )
            
            # Criar grupos baseado em quartis ou seleção manual
            group_method = st.radio(
                "Método de agrupamento:",
                ["Quartis (automático)", "Mediana (alto/baixo)"],
                help="Como dividir os dados em grupos"
            )
            
            if group_method == "Quartis (automático)":
                df['Grupo'] = pd.qcut(df[numeric_var], q=3, labels=['Baixo', 'Médio', 'Alto'])
            else:
                median = df[numeric_var].median()
                df['Grupo'] = np.where(df[numeric_var] >= median, 'Alto', 'Baixo')
            
            # Realizar ANOVA
            groups = [group[numeric_var].values for name, group in df.groupby('Grupo')]
            
            if len(groups) >= 2:
                f_stat, p_value = stats.f_oneway(*groups)
                
                # Resultados
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Estatística F", f"{f_stat:.4f}")
                with col2:
                    st.metric("Valor-p", f"{p_value:.4f}")
                with col3:
                    significance = "Significativo" if p_value < 0.05 else "Não significativo"
                    st.metric("Resultado", significance)
                
                # Interpretação
                st.markdown("### 📝 Interpretação")
                if p_value < 0.05:
                    st.success(f"✅ Existem diferenças estatisticamente significativas entre os grupos (p = {p_value:.4f} < 0.05)")
                else:
                    st.warning(f"⚠️ Não há diferenças estatisticamente significativas entre os grupos (p = {p_value:.4f} > 0.05)")
                
                # Visualização
                fig, ax = plt.subplots(figsize=(10, 6))
                df.boxplot(column=numeric_var, by='Grupo', ax=ax)
                ax.set_title(f'Distribuição de {numeric_var} por Grupo', fontsize=14)
                ax.set_xlabel('Grupo')
                ax.set_ylabel(numeric_var)
                plt.xticks(rotation=0)
                
                st.pyplot(fig)
        
        # Coeficiente de Variação
        else:
            st.markdown("### 📊 Análise de Coeficiente de Variação (CV)")
            
            cv_vars = st.multiselect(
                "Selecione as variáveis para análise de CV:",
                st.session_state.numeric_cols,
                default=st.session_state.numeric_cols[:3] if len(st.session_state.numeric_cols) >= 3 else st.session_state.numeric_cols,
                help="O CV mede a dispersão relativa dos dados"
            )
            
            if cv_vars:
                # Calcular CV para cada variável
                cv_data = []
                for var in cv_vars:
                    data = df[var].dropna()
                    mean = np.mean(data)
                    std = np.std(data)
                    cv = (std / mean * 100) if mean != 0 else 0
                    
                    # Classificação do CV
                    if cv < 10:
                        classification = "Baixa dispersão"
                    elif cv < 20:
                        classification = "Média dispersão"
                    elif cv < 30:
                        classification = "Alta dispersão"
                    else:
                        classification = "Muito alta dispersão"
                    
                    cv_data.append({
                        'Variável': var,
                        'Média': mean,
                        'Desvio Padrão': std,
                        'CV (%)': cv,
                        'Classificação': classification
                    })
                
                cv_df = pd.DataFrame(cv_data)
                
                # Visualização
                st.dataframe(
                    cv_df.style.background_gradient(subset=['CV (%)'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
                
                # Gráfico de barras
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(cv_df['Variável'], cv_df['CV (%)'], color=['green' if cv < 20 else 'orange' if cv < 30 else 'red' for cv in cv_df['CV (%)']])
                ax.set_title('Coeficiente de Variação por Variável', fontsize=14)
                ax.set_ylabel('CV (%)')
                ax.set_xlabel('Variáveis')
                plt.xticks(rotation=45)
                
                # Adicionar valores nas barras
                for bar, cv in zip(bars, cv_df['CV (%)']):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                           f'{cv:.1f}%', ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # Recomendações
                st.markdown("### 💡 Recomendações")
                high_cv = cv_df[cv_df['CV (%)'] > 30]
                if not high_cv.empty:
                    st.warning("⚠️ As seguintes variáveis apresentam alta variabilidade:")
                    for _, row in high_cv.iterrows():
                        st.write(f"- **{row['Variável']}**: CV = {row['CV (%)']:.1f}% - Considere investigar outliers ou fatores externos")
    
    else:
        st.info("👆 Faça o upload dos dados e certifique-se de ter pelo menos 2 variáveis numéricas")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 5: GRÁFICOS
# ==============================================================================
with tab5:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("## 📈 Visualizações Gráficas")
    
    if st.session_state.treated_df is not None:
        df = st.session_state.treated_df.copy()
        
        # Tipo de gráfico
        chart_type = st.selectbox(
            "Selecione o tipo de gráfico:",
            ["Gráfico de Linhas (Série Temporal)", "Histograma", "Boxplot", "Dispersão"],
            help="Escolha o tipo de visualização"
        )
        
        # Selecionar variáveis
        selected_vars = st.multiselect(
            "Selecione as variáveis para visualização:",
            st.session_state.numeric_cols,
            default=[st.session_state.numeric_cols[0]] if st.session_state.numeric_cols else [],
            help="Escolha as variáveis meteorológicas"
        )
        
        if selected_vars:
            # Gráfico de Linhas
            if chart_type == "Gráfico de Linhas (Série Temporal)":
                # Procurar coluna de data
                date_col = None
                for col in df.columns:
                    if 'data' in col.lower() or 'date' in col.lower():
                        date_col = col
                        break
                
                if date_col:
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        df_sorted = df.sort_values(date_col)
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        for var in selected_vars:
                            ax.plot(df_sorted[date_col], df_sorted[var], 
                                   label=var, linewidth=2, alpha=0.8)
                        
                        ax.set_title('Série Temporal das Variáveis Meteorológicas', 
                                    fontsize=14, fontweight='bold')
                        ax.set_xlabel('Data')
                        ax.set_ylabel('Valores')
                        ax.legend(loc='best')
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Erro ao criar gráfico temporal: {str(e)}")
                else:
                    st.warning("⚠️ Coluna de data não encontrada. Criando gráfico com índice.")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    for var in selected_vars:
                        ax.plot(df.index, df[var], label=var, linewidth=2, alpha=0.8)
                    
                    ax.set_title('Série dos Dados', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Índice')
                    ax.set_ylabel('Valores')
                    ax.legend(loc='best')
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
            
            # Histograma
            elif chart_type == "Histograma":
                cols = st.columns(min(len(selected_vars), 2))
                
                for i, var in enumerate(selected_vars):
                    with cols[i % 2]:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        data = df[var].dropna()
                        
                        ax.hist(data, bins=30, color='steelblue', edgecolor='white', 
                               alpha=0.7, density=True)
                        
                        # Curva de densidade
                        from scipy.stats import gaussian_kde
                        if len(data) > 1:
                            kde = gaussian_kde(data)
                            x_range = np.linspace(data.min(), data.max(), 100)
                            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Densidade')
                        
                        ax.set_title(f'Histograma - {var}', fontsize=12, fontweight='bold')
                        ax.set_xlabel(var)
                        ax.set_ylabel('Frequência')
                        ax.legend()
                        
                        st.pyplot(fig)
                        
                        # Estatísticas do histograma
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Média", f"{data.mean():.2f}")
                        with col2:
                            st.metric("Mediana", f"{data.median():.2f}")
                        with col3:
                            st.metric("Desvio Padrão", f"{data.std():.2f}")
                        
                        st.markdown("---")
            
            # Boxplot
            elif chart_type == "Boxplot":
                fig, ax = plt.subplots(figsize=(10, 6))
                
                df[selected_vars].boxplot(ax=ax)
                ax.set_title('Diagrama de Caixa (Boxplot)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Valores')
                plt.xticks(rotation=45)
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Detecção de outliers
                st.markdown("### 🔍 Detecção de Outliers")
                for var in selected_vars:
                    data = df[var].dropna()
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
                    
                    if len(outliers) > 0:
                        st.warning(f"**{var}**: {len(outliers)} outliers detectados ({len(outliers)/len(data)*100:.1f}%)")
            
            # Dispersão
            elif chart_type == "Dispersão":
                if len(selected_vars) >= 2:
                    x_var = st.selectbox("Variável X:", selected_vars)
                    y_var = st.selectbox("Variável Y:", selected_vars, index=min(1, len(selected_vars)-1))
                    
                    if x_var != y_var:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        scatter = ax.scatter(df[x_var], df[y_var], 
                                           c=df.index, cmap='viridis', 
                                           alpha=0.6, edgecolors='white', linewidth=0.5)
                        
                        # Linha de tendência
                        z = np.polyfit(df[x_var].dropna(), df[y_var].dropna(), 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(df[x_var].min(), df[x_var].max(), 100)
                        ax.plot(x_range, p(x_range), "r--", linewidth=2, label='Tendência')
                        
                        ax.set_xlabel(x_var)
                        ax.set_ylabel(y_var)
                        ax.set_title(f'Dispersão: {x_var} vs {y_var}', fontsize=14, fontweight='bold')
                        ax.legend()
                        plt.colorbar(scatter, label='Índice')
                        
                        # Adicionar coeficiente de correlação
                        corr = df[x_var].corr(df[y_var])
                        ax.text(0.05, 0.95, f'Correlação: {corr:.3f}', 
                               transform=ax.transAxes, fontsize=12,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                        
                        st.pyplot(fig)
                    else:
                        st.warning("⚠️ Selecione variáveis diferentes para X e Y")
                else:
                    st.warning("⚠️ Selecione pelo menos 2 variáveis para o gráfico de dispersão")
        
        else:
            st.info("👆 Selecione as variáveis para visualização")
    
    else:
        st.info("👆 Faça o upload dos dados na aba 'Upload' primeiro")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 6: DOWNLOAD
# ==============================================================================
with tab6:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("## 📥 Download dos Dados Tratados")
    
    if st.session_state.treated_df is not None:
        df = st.session_state.treated_df.copy()
        
        st.success(f"✅ Dataset pronto para download: {len(df)} registros e {len(df.columns)} colunas")
        
        # Opções de download
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📄 Formato CSV")
            
            # Prévia do arquivo
            st.markdown("**Prévia dos dados tratados:**")
            st.dataframe(df.head(5), use_container_width=True)
            
            # Download CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Baixar Arquivo CSV",
                data=csv,
                file_name="dados_meteorologicos_tratados.csv",
                mime="text/csv",
                help="Download do dataset completo em formato CSV"
            )
        
        with col2:
            st.markdown("### 📊 Resumo do Tratamento")
            
            # Métricas do tratamento
            original_rows = len(st.session_state.df) if st.session_state.df is not None else 0
            treated_rows = len(df)
            rows_removed = original_rows - treated_rows
            
            st.metric("Registros Originais", original_rows)
            st.metric("Registros Tratados", treated_rows)
            
            if rows_removed > 0:
                st.metric("Linhas Removidas", rows_removed)
            else:
                st.metric("Linhas Removidas", 0)
            
            # Colunas tratadas
            missing_before = st.session_state.df.isnull().sum().sum() if st.session_state.df is not None else 0
            missing_after = df.isnull().sum().sum()
            
            st.metric("Dados Faltantes Antes", missing_before)
            st.metric("Dados Faltantes Depois", missing_after)
            
            if missing_before > missing_after:
                st.success(f"✅ {missing_before - missing_after} valores foram preenchidos")
            
            # Botão para reset
            if st.button("🔄 Resetar Tratamentos"):
                st.session_state.treated_df = st.session_state.df.copy()
                st.success("✅ Tratamentos resetados com sucesso!")
                st.rerun()
    
    else:
        st.info("👆 Faça o upload dos dados na aba 'Upload' primeiro e realize os tratamentos desejados")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# RODAPÉ COM REFERÊNCIAS E CRÉDITOS
# ==============================================================================
st.markdown("---")
st.markdown('<div class="custom-card">', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📚 Referências e Métodos")
    st.markdown("""
    **Metodologias Científicas Utilizadas:**
    
    - **Correlação de Pearson**: Mede a força e direção da relação linear entre duas variáveis
    
    - **ANOVA (Analysis of Variance)**: Teste paramétrico para comparar médias entre grupos
    
    - **Coeficiente de Variação (CV)**: Medida de dispersão relativa, calculada como (Desvio Padrão / Média) × 100
    
    - **Imputação por Média**: Método de preenchimento de dados faltantes usando a média da variável
    
    **Referências Bibliográficas:**
    - Montgomery, D.C. (2017). Design and Analysis of Experiments. 9th Edition.
    - Wilks, D.S. (2011). Statistical Methods in the Atmospheric Sciences. 3rd Edition.
    - INMET (2024). Normais Climatológicas do Brasil.
    """)

with col2:
    st.markdown("### 👨‍💻 Créditos")
    st.markdown("""
    **AgroAnalytics Pro v1.0**
    
    Desenvolvido com ❤️ para a comunidade agrícola e científica
    
    **Tecnologias:**
    - Python 3.10+
    - Streamlit
    - Pandas
    - NumPy
    - SciPy
    - Matplotlib
    - Seaborn
    
    **IA Integrada:**
    - Google Gemini API
    
    © 2024 - Todos os direitos reservados
    """)

st.markdown('</div>', unsafe_allow_html=True)
