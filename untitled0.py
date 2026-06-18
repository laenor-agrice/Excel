# ============================================================
# AgroDataLab - Análise de Dados Meteorológicos
# Versão otimizada para Streamlit Cloud
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
# CSS CUSTOMIZADO
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
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1a5632 0%, #2d8a4e 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 20px;
    }
    .success-message {
        background: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
    }
    .warning-message {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TÍTULO PRINCIPAL
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
    
    Análise de dados meteorológicos do INMET.
    
    Funcionalidades:
    - Upload de dados
    - Tratamento de valores ausentes
    - Estatísticas descritivas
    - Visualizações gráficas
    - Download dos dados
    """)

# ============================================================
# FUNÇÕES AUXILIARES (SEM SCIPY)
# ============================================================

def process_inmet_data(df):
    """Processa dados no formato INMET"""
    # Converter colunas numéricas
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except:
            pass
    return df

def fill_missing_values(df, method='mean'):
    """Preenche valores ausentes"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif method == 'interpolate':
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
    
    return df

def calculate_correlation(df):
    """Calcula correlação usando pandas"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        return df[numeric_cols].corr()
    return None

def calculate_cv(df):
    """Calcula Coeficiente de Variação"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    cv_data = []
    for col in numeric_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        cv = (std_val / mean_val * 100) if mean_val != 0 else np.nan
        
        classificacao = "Baixo" if cv <= 10 else "Médio" if cv <= 20 else "Alto" if cv <= 30 else "Muito Alto"
        
        cv_data.append({
            'Variável': col,
            'Média': round(mean_val, 3),
            'Desvio Padrão': round(std_val, 3),
            'CV (%)': round(cv, 2),
            'Classificação': classificacao
        })
    
    return pd.DataFrame(cv_data)

def create_line_chart(df, x_col, y_cols):
    """Cria gráfico de linhas"""
    df_melted = pd.melt(
        df, 
        id_vars=[x_col], 
        value_vars=y_cols,
        var_name='Variável', 
        value_name='Valor'
    )
    
    chart = alt.Chart(df_melted).mark_line(point=True).encode(
        x=alt.X(f'{x_col}:T', title='Data'),
        y=alt.Y('Valor:Q', title='Valor'),
        color='Variável:N'
    ).properties(
        width=800,
        height=400
    ).interactive()
    
    return chart

def create_histogram(df, cols):
    """Cria histograma"""
    df_melted = pd.melt(
        df, 
        value_vars=cols,
        var_name='Variável', 
        value_name='Valor'
    )
    
    chart = alt.Chart(df_melted).mark_bar(opacity=0.7).encode(
        alt.X('Valor:Q', bin=alt.Bin(maxbins=30)),
        alt.Y('count()'),
        color='Variável:N'
    ).properties(
        width=800,
        height=400
    ).facet(
        'Variável:N',
        columns=2
    )
    
    return chart

# ============================================================
# CRIAÇÃO DAS ABAS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📤 Upload",
    "🔧 Tratamento", 
    "📊 Estatísticas",
    "🔬 Análises",
    "📈 Gráficos",
    "💾 Download"
])

# ============================================================
# ABA 1 - UPLOAD
# ============================================================

with tab1:
    st.markdown("### 📤 Upload de Dados")
    
    uploaded_file = st.file_uploader(
        "Selecione o arquivo CSV ou Excel",
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        try:
            # Carregar dados
            if uploaded_file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except:
                    df = pd.read_csv(uploaded_file, encoding='latin1')
            else:
                df = pd.read_excel(uploaded_file)
            
            # Processar
            df = process_inmet_data(df)
            
            # Salvar na sessão
            st.session_state['df_original'] = df.copy()
            st.session_state['df_processed'] = df.copy()
            
            # Sucesso
            st.markdown(f"""
            <div class="success-message">
                <h4>✅ Upload realizado com sucesso!</h4>
                <p>Arquivo: {uploaded_file.name}</p>
                <p>Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Métricas
            col1, col2, col3 = st.columns(3)
            col1.metric("Registros", len(df))
            col2.metric("Variáveis", len(df.columns))
            col3.metric("Dados Ausentes", df.isnull().sum().sum())
            
            # Preview
            st.markdown("### Primeiras 10 linhas:")
            st.dataframe(df.head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro ao carregar: {str(e)}")
    else:
        st.info("👆 Faça o upload de um arquivo para começar")

# ============================================================
# ABA 2 - TRATAMENTO
# ============================================================

with tab2:
    st.markdown("### 🔧 Tratamento de Dados Ausentes")
    
    if 'df_processed' in st.session_state:
        df = st.session_state['df_processed']
        missing_count = df.isnull().sum().sum()
        
        if missing_count > 0:
            st.warning(f"⚠️ {missing_count} valores ausentes detectados")
            
            # Mostrar colunas com dados ausentes
            missing_info = pd.DataFrame({
                'Coluna': df.columns,
                'Valores Ausentes': df.isnull().sum().values,
                'Porcentagem (%)': (df.isnull().sum() / len(df) * 100).round(2)
            })
            missing_info = missing_info[missing_info['Valores Ausentes'] > 0]
            st.dataframe(missing_info, use_container_width=True)
            
            st.markdown("### Opções de Tratamento")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Preencher com Média"):
                    df_filled = fill_missing_values(df.copy(), 'mean')
                    st.session_state['df_processed'] = df_filled
                    st.success("✅ Valores preenchidos com média!")
                    st.rerun()
            
            with col2:
                if st.button("📈 Preencher com Mediana"):
                    df_filled = fill_missing_values(df.copy(), 'median')
                    st.session_state['df_processed'] = df_filled
                    st.success("✅ Valores preenchidos com mediana!")
                    st.rerun()
            
            with col3:
                if st.button("🔄 Interpolar"):
                    df_filled = fill_missing_values(df.copy(), 'interpolate')
                    st.session_state['df_processed'] = df_filled
                    st.success("✅ Valores interpolados!")
                    st.rerun()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Excluir Linhas com Dados Ausentes"):
                    df_dropped = df.dropna()
                    st.session_state['df_processed'] = df_dropped
                    st.success(f"✅ {len(df) - len(df_dropped)} linhas removidas!")
                    st.rerun()
            
            with col2:
                if st.button("🔙 Restaurar Dados Originais"):
                    if 'df_original' in st.session_state:
                        st.session_state['df_processed'] = st.session_state['df_original'].copy()
                        st.success("✅ Dados restaurados!")
                        st.rerun()
        else:
            st.success("✅ Nenhum valor ausente detectado!")
    else:
        st.warning("⚠️ Faça o upload na Aba 1 primeiro!")

# ============================================================
# ABA 3 - ESTATÍSTICAS
# ============================================================

with tab3:
    st.markdown("### 📊 Estatísticas Descritivas")
    
    if 'df_processed' in st.session_state:
        df = st.session_state['df_processed']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Estatísticas básicas
            st.markdown("#### Resumo Estatístico")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            # Agregação temporal
            date_columns = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower()]
            
            if date_columns:
                st.markdown("---")
                st.markdown("#### Agregação Temporal")
                
                date_col = st.selectbox("Coluna de data:", date_columns)
                freq = st.selectbox("Frequência:", ['Diário', 'Semanal', 'Mensal'])
                
                freq_map = {'Diário': 'D', 'Semanal': 'W', 'Mensal': 'M'}
                
                try:
                    df_temp = df.copy()
                    df_temp[date_col] = pd.to_datetime(df_temp[date_col], format='mixed')
                    df_temp.set_index(date_col, inplace=True)
                    
                    stats_grouped = df_temp[numeric_cols].resample(freq_map[freq]).agg(['mean', 'std', 'min', 'max'])
                    st.dataframe(stats_grouped.round(3), use_container_width=True)
                except:
                    st.error("Erro ao processar agregação temporal")
        else:
            st.warning("Nenhuma variável numérica encontrada")
    else:
        st.warning("⚠️ Faça o upload na Aba 1 primeiro!")

# ============================================================
# ABA 4 - ANÁLISES AVANÇADAS
# ============================================================

with tab4:
    st.markdown("### 🔬 Análises Avançadas")
    
    if 'df_processed' in st.session_state:
        df = st.session_state['df_processed']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            # Correlação
            st.markdown("#### Matriz de Correlação")
            
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                st.dataframe(corr_matrix.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1), use_container_width=True)
                
                st.markdown("""
                **Interpretação:**
                - 0.0 a 0.3: Correlação Fraca
                - 0.3 a 0.7: Correlação Moderada  
                - 0.7 a 1.0: Correlação Forte
                """)
            
            # Coeficiente de Variação
            st.markdown("---")
            st.markdown("#### Coeficiente de Variação")
            
            cv_df = calculate_cv(df)
            st.dataframe(cv_df, use_container_width=True)
            
            st.markdown("""
            **Classificação (Pimentel-Gomes, 2000):**
            - CV ≤ 10%: Baixo (alta precisão)
            - 10% < CV ≤ 20%: Médio
            - 20% < CV ≤ 30%: Alto
            - CV > 30%: Muito Alto (baixa precisão)
            """)
        else:
            st.warning("Nenhuma variável numérica encontrada")
    else:
        st.warning("⚠️ Faça o upload na Aba 1 primeiro!")

# ============================================================
# ABA 5 - GRÁFICOS
# ============================================================

with tab5:
    st.markdown("### 📈 Visualização de Dados")
    
    if 'df_processed' in st.session_state:
        df = st.session_state['df_processed']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            chart_type = st.radio(
                "Tipo de gráfico:",
                ['📈 Linhas', '📊 Histograma'],
                horizontal=True
            )
            
            if chart_type == '📈 Linhas':
                date_columns = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower()]
                
                if date_columns:
                    x_col = st.selectbox("Eixo X:", date_columns)
                    y_cols = st.multiselect(
                        "Variáveis Y:",
                        numeric_cols,
                        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                    )
                    
                    if y_cols:
                        try:
                            df_temp = df.copy()
                            df_temp[x_col] = pd.to_datetime(df_temp[x_col], format='mixed')
                            chart = create_line_chart(df_temp, x_col, y_cols)
                            st.altair_chart(chart, use_container_width=True)
                        except:
                            st.error("Erro ao criar gráfico")
            
            elif chart_type == '📊 Histograma':
                selected_cols = st.multiselect(
                    "Variáveis:",
                    numeric_cols,
                    default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
                )
                
                if selected_cols:
                    chart = create_histogram(df, selected_cols)
                    st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("Nenhuma variável numérica encontrada")
    else:
        st.warning("⚠️ Faça o upload na Aba 1 primeiro!")

# ============================================================
# ABA 6 - DOWNLOAD
# ============================================================

with tab6:
    st.markdown("### 💾 Download dos Dados")
    
    if 'df_processed' in st.session_state:
        df = st.session_state['df_processed']
        
        st.success("✅ Dados prontos para download!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar CSV",
                data=csv,
                file_name="dados_processados.csv",
                mime="text/csv"
            )
        
        with col2:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Dados', index=False)
            
            st.download_button(
                label="📥 Baixar Excel",
                data=output.getvalue(),
                file_name="dados_processados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("⚠️ Faça o upload e processamento primeiro!")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌱 AgroDataLab v1.0 | Análise de Dados Meteorológicos | 2024</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INICIALIZAÇÃO
# ============================================================

if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None
if 'df_processed' not in st.session_state:
    st.session_state['df_processed'] = None
