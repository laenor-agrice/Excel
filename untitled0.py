# ============================================================
# AgroDataLab - Análise de Dados Meteorológicos
# Versão Simplificada para Streamlit Cloud
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="AgroDataLab",
    page_icon="🌱",
    layout="wide"
)

# ============================================================
# CSS SIMPLES
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a5632 0%, #2d8a4e 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    .success-box {
        background: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .stButton > button {
        background: #1a5632;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 20px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TÍTULO
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🌱 AgroDataLab</h1>
    <p>Análise Inteligente de Dados Meteorológicos - INMET</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📋 Sobre")
    st.info("""
    **AgroDataLab v1.0**
    
    Ferramenta para análise de dados 
    meteorológicos do INMET.
    
    ✅ Upload de dados
    ✅ Tratamento de dados
    ✅ Estatísticas
    ✅ Gráficos
    ✅ Download
    """)

# ============================================================
# FUNÇÕES SIMPLES
# ============================================================

def process_data(df):
    """Processa dados básicos"""
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except:
            pass
    return df

def fill_missing(df, method='mean'):
    """Preenche valores ausentes"""
    num_cols = df.select_dtypes(include=[np.number]).columns
    if method == 'mean':
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif method == 'median':
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif method == 'interpolate':
        df[num_cols] = df[num_cols].interpolate(method='linear')
    return df

def create_line_chart(df, x_col, y_cols):
    """Gráfico de linhas"""
    df_melted = pd.melt(
        df, id_vars=[x_col], value_vars=y_cols,
        var_name='Variavel', value_name='Valor'
    )
    
    chart = alt.Chart(df_melted).mark_line(point=True).encode(
        x=alt.X(f'{x_col}:T', title='Data'),
        y=alt.Y('Valor:Q', title='Valor'),
        color='Variavel:N'
    ).properties(height=400).interactive()
    
    return chart

def create_histogram(df, cols):
    """Histograma"""
    df_melted = pd.melt(
        df, value_vars=cols,
        var_name='Variavel', value_name='Valor'
    )
    
    chart = alt.Chart(df_melted).mark_bar(opacity=0.7).encode(
        alt.X('Valor:Q', bin=alt.Bin(maxbins=30)),
        alt.Y('count()'),
        color='Variavel:N'
    ).properties(height=400)
    
    return chart

# ============================================================
# INICIALIZAR SESSION STATE
# ============================================================
if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None
if 'df_processed' not in st.session_state:
    st.session_state['df_processed'] = None

# ============================================================
# ABAS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Upload",
    "🔧 Tratamento", 
    "📊 Estatísticas",
    "📈 Gráficos",
    "💾 Download"
])

# ============================================================
# ABA 1 - UPLOAD
# ============================================================
with tab1:
    st.markdown("### 📤 Upload de Dados")
    
    uploaded_file = st.file_uploader(
        "Selecione um arquivo CSV ou Excel",
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            else:
                df = pd.read_excel(uploaded_file)
            
            df = process_data(df)
            
            st.session_state['df_original'] = df.copy()
            st.session_state['df_processed'] = df.copy()
            
            st.markdown(f"""
            <div class="success-box">
                <h4>✅ Upload realizado!</h4>
                <p>Arquivo: {uploaded_file.name}</p>
                <p>{df.shape[0]} linhas × {df.shape[1]} colunas</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Registros", len(df))
            col2.metric("Variáveis", len(df.columns))
            col3.metric("Dados Ausentes", df.isnull().sum().sum())
            
            st.markdown("### Primeiras 10 linhas:")
            st.dataframe(df.head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro: {str(e)}")
    else:
        st.info("👆 Faça upload de um arquivo para começar")

# ============================================================
# ABA 2 - TRATAMENTO
# ============================================================
with tab2:
    st.markdown("### 🔧 Tratamento de Dados")
    
    if st.session_state['df_processed'] is not None:
        df = st.session_state['df_processed']
        missing = df.isnull().sum().sum()
        
        if missing > 0:
            st.markdown(f"""
            <div class="warning-box">
                <h4>⚠️ {missing} valores ausentes encontrados</h4>
            </div>
            """, unsafe_allow_html=True)
            
            missing_df = pd.DataFrame({
                'Coluna': df.columns,
                'Ausentes': df.isnull().sum().values,
                '%': (df.isnull().sum() / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Ausentes'] > 0]
            st.dataframe(missing_df, use_container_width=True)
            
            st.markdown("### Opções:")
            
            c1, c2, c3 = st.columns(3)
            
            with c1:
                if st.button("📊 Preencher com Média"):
                    st.session_state['df_processed'] = fill_missing(df.copy(), 'mean')
                    st.success("✅ Preenchido com média!")
                    st.rerun()
            
            with c2:
                if st.button("📈 Preencher com Mediana"):
                    st.session_state['df_processed'] = fill_missing(df.copy(), 'median')
                    st.success("✅ Preenchido com mediana!")
                    st.rerun()
            
            with c3:
                if st.button("🔄 Interpolar"):
                    st.session_state['df_processed'] = fill_missing(df.copy(), 'interpolate')
                    st.success("✅ Interpolado!")
                    st.rerun()
            
            c4, c5 = st.columns(2)
            
            with c4:
                if st.button("🗑️ Remover linhas com ausentes"):
                    st.session_state['df_processed'] = df.dropna()
                    st.success(f"✅ Linhas removidas!")
                    st.rerun()
            
            with c5:
                if st.button("🔙 Restaurar original"):
                    st.session_state['df_processed'] = st.session_state['df_original'].copy()
                    st.success("✅ Restaurado!")
                    st.rerun()
        else:
            st.success("✅ Nenhum valor ausente!")
    else:
        st.warning("⚠️ Faça upload na Aba 1 primeiro!")

# ============================================================
# ABA 3 - ESTATÍSTICAS
# ============================================================
with tab3:
    st.markdown("### 📊 Estatísticas")
    
    if st.session_state['df_processed'] is not None:
        df = st.session_state['df_processed']
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(num_cols) > 0:
            st.markdown("#### Estatísticas Descritivas")
            st.dataframe(df[num_cols].describe(), use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### Coeficiente de Variação")
            
            cv_list = []
            for col in num_cols:
                mean_v = df[col].mean()
                std_v = df[col].std()
                cv = (std_v / mean_v * 100) if mean_v != 0 else 0
                
                if cv <= 10:
                    cls = "Baixo"
                elif cv <= 20:
                    cls = "Médio"
                elif cv <= 30:
                    cls = "Alto"
                else:
                    cls = "Muito Alto"
                
                cv_list.append({
                    'Variável': col,
                    'Média': round(mean_v, 2),
                    'Desvio': round(std_v, 2),
                    'CV (%)': round(cv, 2),
                    'Classificação': cls
                })
            
            cv_df = pd.DataFrame(cv_list)
            st.dataframe(cv_df, use_container_width=True)
            
            if len(num_cols) >= 2:
                st.markdown("---")
                st.markdown("#### Matriz de Correlação")
                corr = df[num_cols].corr()
                st.dataframe(corr, use_container_width=True)
        else:
            st.warning("Sem variáveis numéricas")
    else:
        st.warning("⚠️ Faça upload na Aba 1 primeiro!")

# ============================================================
# ABA 4 - GRÁFICOS
# ============================================================
with tab4:
    st.markdown("### 📈 Gráficos")
    
    if st.session_state['df_processed'] is not None:
        df = st.session_state['df_processed']
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(num_cols) > 0:
            tipo = st.radio("Tipo:", ['📈 Linhas', '📊 Histograma'], horizontal=True)
            
            if tipo == '📈 Linhas':
                date_cols = [c for c in df.columns if 'data' in c.lower() or 'date' in c.lower()]
                
                if date_cols:
                    x_col = st.selectbox("Eixo X:", date_cols)
                    y_cols = st.multiselect("Eixo Y:", num_cols, default=num_cols[:2])
                    
                    if y_cols:
                        try:
                            df_temp = df.copy()
                            df_temp[x_col] = pd.to_datetime(df_temp[x_col])
                            chart = create_line_chart(df_temp, x_col, y_cols)
                            st.altair_chart(chart, use_container_width=True)
                        except:
                            st.error("Erro ao criar gráfico")
                else:
                    st.info("Coluna de data não encontrada")
            
            else:
                cols = st.multiselect("Variáveis:", num_cols, default=num_cols[:2])
                if cols:
                    chart = create_histogram(df, cols)
                    st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("Sem variáveis numéricas")
    else:
        st.warning("⚠️ Faça upload na Aba 1 primeiro!")

# ============================================================
# ABA 5 - DOWNLOAD
# ============================================================
with tab5:
    st.markdown("### 💾 Download")
    
    if st.session_state['df_processed'] is not None:
        df = st.session_state['df_processed']
        
        st.markdown("""
        <div class="success-box">
            <h4>✅ Dados prontos para download!</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Registros", len(df))
        col2.metric("Colunas", len(df.columns))
        col3.metric("Ausentes", df.isnull().sum().sum())
        
        st.markdown("---")
        
        # Download CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar arquivo CSV",
            data=csv_data,
            file_name="dados_processados.csv",
            mime="text/csv",
            key="download_csv"
        )
        
        st.markdown("---")
        st.markdown("### 📋 Preview dos dados:")
        st.dataframe(df.head(20), use_container_width=True)
        
    else:
        st.warning("⚠️ Faça upload na Aba 1 primeiro!")

# ============================================================
# RODAPÉ
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🌱 AgroDataLab v1.0 | 2024</p>
</div>
""", unsafe_allow_html=True)
