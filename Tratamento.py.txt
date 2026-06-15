import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Tratador de Dados INMET",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Tratador de Dados INMET")
st.markdown("Converta dados brutos horários do INMET em dados consolidados mensais")

# ============================================================================
# FUNÇÕES DE TRATAMENTO
# ============================================================================

def converter_virgula_para_ponto(valor):
    """Converte string com vírgula para float com ponto"""
    if isinstance(valor, str):
        valor = valor.replace(',', '.')
        try:
            return float(valor)
        except:
            return np.nan
    return valor

def tratar_dados_brutos(df):
    """Aplica todas as transformações necessárias nos dados brutos"""
    
    df_tratado = df.copy()
    
    # ===== 1. PADRONIZAR NOMES DAS COLUNAS =====
    mapeamento_colunas = {
        'Data': 'Date',
        'Hora (UTC)': 'Time',
        'Temp. Ins. (C)': 'T_inst',
        'Temp. Max. (C)': 'Tmax',
        'Temp. Min. (C)': 'Tmin',
        'Umi. Ins. (%)': 'UR_inst',
        'Umi. Max. (%)': 'URmax',
        'Umi. Min. (%)': 'URmin',
        'Pto Orvalho Ins. (C)': 'Td_inst',
        'Pto Orvalho Max. (C)': 'Tdmax',
        'Pto Orvalho Min. (C)': 'Tdmin',
        'Pressao Ins. (hPa)': 'Press_inst',
        'Pressao Max. (hPa)': 'Pressmax',
        'Pressao Min. (hPa)': 'Pressmin',
        'Vel. Vento (m/s)': 'U2',
        'Dir. Vento (m/s)': 'Dir_vento',
        'Raj. Vento (m/s)': 'Raj_vento',
        'Radiacao (KJ/m²)': 'Rad_KJ',
        'Chuva (mm)': 'Precipitacao'
    }
    
    # Renomeia apenas colunas que existem
    for col_antiga, col_nova in mapeamento_colunas.items():
        if col_antiga in df_tratado.columns:
            df_tratado.rename(columns={col_antiga: col_nova}, inplace=True)
    
    # ===== 2. CONVERTER VÍRGULA PARA PONTO NAS COLUNAS NUMÉRICAS =====
    colunas_numericas = ['T_inst', 'Tmax', 'Tmin', 'UR_inst', 'URmax', 'URmin',
                         'Td_inst', 'Tdmax', 'Tdmin', 'Press_inst', 'Pressmax',
                         'Pressmin', 'U2', 'Dir_vento', 'Raj_vento', 'Rad_KJ', 'Precipitacao']
    
    for col in colunas_numericas:
        if col in df_tratado.columns:
            df_tratado[col] = df_tratado[col].apply(converter_virgula_para_ponto)
    
    # ===== 3. CRIAR COLUNA DE DATA+HORA COMPLETA =====
    if 'Date' in df_tratado.columns and 'Time' in df_tratado.columns:
        # Ajustar formato da hora (ex: "0000" -> "00:00")
        df_tratado['Time'] = df_tratado['Time'].astype(str).str.zfill(4)
        df_tratado['Hora_Formatada'] = df_tratado['Time'].str[:2] + ':' + df_tratado['Time'].str[2:4]
        
        # Criar datetime completo
        df_tratado['Datetime'] = pd.to_datetime(
            df_tratado['Date'].astype(str) + ' ' + df_tratado['Hora_Formatada'],
            format='%d/%m/%Y %H:%M',
            errors='coerce'
        )
        
        # Extrair mês e ano para consolidação
        df_tratado['Ano'] = df_tratado['Datetime'].dt.year
        df_tratado['Mes'] = df_tratado['Datetime'].dt.month
        df_tratado['Ano_Mes'] = df_tratado['Datetime'].dt.strftime('%Y-%m')
    
    # ===== 4. IDENTIFICAR NOME DA ESTAÇÃO =====
    # Tentar extrair do nome do arquivo ou criar padrão
    nome_estacao = "Estacao"
    
    # ===== 5. CONSOLIDAR POR MÊS =====
    if 'Ano' in df_tratado.columns and 'Mes' in df_tratado.columns:
        
        # Função para agregar dados mensais
        df_consolidado = df_tratado.groupby(['Ano', 'Mes']).agg({
            'Tmax': 'max',
            'Tmin': 'min',
            'URmax': 'max',
            'URmin': 'min',
            'U2': 'mean'
        }).reset_index()
        
        # Criar coluna de identificação do mês
        df_consolidado['Mes'] = df_consolidado['Mes'].astype(int).astype(str).str.zfill(2)
        df_consolidado['Ano_Mes'] = df_consolidado['Ano'].astype(str) + '-' + df_consolidado['Mes']
        
        # Adicionar nome da estação
        if 'Estacao' in df.columns:
            nome_estacao = df['Estacao'].iloc[0] if len(df['Estacao'].unique()) == 1 else "Multipla"
        
        df_consolidado['Estacao'] = nome_estacao
        
        # Selecionar colunas finais
        colunas_finais = ['Mes', 'Estacao', 'Tmax', 'Tmin', 'URmax', 'URmin', 'U2']
        df_final = df_consolidado[colunas_finais]
        
        # Renomear para o padrão desejado
        df_final.columns = ['Mes', 'Estacao', 'Tmax', 'Tmin', 'URmax', 'URmin', 'U2']
        
        # Ordenar por mês
        df_final['Ano_Mes_Ordenar'] = df_final['Mes']
        df_final = df_final.sort_values('Mes').drop('Ano_Mes_Ordenar', axis=1)
        
        return df_final, df_tratado
    
    return pd.DataFrame(), df_tratado


# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

st.markdown("---")
st.subheader("📂 Upload do Arquivo Bruto INMET")

uploaded_file = st.file_uploader(
    "Carregue o arquivo CSV bruto do INMET",
    type=['csv'],
    help="O arquivo deve estar no formato padrão INMET (com colunas em português)"
)

if uploaded_file is not None:
    
    # Tentar ler o arquivo com diferentes delimitadores
    try:
        # Primeiro tenta ler com ponto e vírgula (padrão INMET)
        df_raw = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        
        # Verificar se conseguiu ler as colunas corretamente
        if 'Data' not in df_raw.columns:
            # Tentar com vírgula
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
        
        st.success("✅ Arquivo carregado com sucesso!")
        
        # Mostrar prévia dos dados brutos
        with st.expander("📋 Prévia dos Dados Brutos"):
            st.dataframe(df_raw.head(10))
            st.caption(f"Total de linhas: {len(df_raw)}")
        
        # Processar os dados
        with st.spinner("🔄 Processando dados..."):
            df_consolidado, df_detalhado = tratar_dados_brutos(df_raw)
        
        # ===== EXIBIR RESULTADOS =====
        st.markdown("---")
        st.subheader("📊 Dados Consolidados (Formato Tratado)")
        
        if not df_consolidado.empty:
            st.dataframe(df_consolidado, use_container_width=True)
            
            # Mostrar estatísticas
            st.markdown("### 📈 Estatísticas Consolidadas")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Tmax Máxima", f"{df_consolidado['Tmax'].max():.1f}°C")
            with col2:
                st.metric("Tmin Mínima", f"{df_consolidado['Tmin'].min():.1f}°C")
            with col3:
                st.metric("URmax Máxima", f"{df_consolidado['URmax'].max():.0f}%")
            with col4:
                st.metric("URmin Mínima", f"{df_consolidado['URmin'].min():.0f}%")
            
            # ===== VISUALIZAÇÕES =====
            st.markdown("---")
            st.subheader("📉 Visualizações")
            
            col_graf1, col_graf2 = st.columns(2)
            
            with col_graf1:
                st.line_chart(
                    df_consolidado.set_index('Mes')[['Tmax', 'Tmin']],
                    height=400
                )
                st.caption("Evolução das Temperaturas Máximas e Mínimas")
            
            with col_graf2:
                st.line_chart(
                    df_consolidado.set_index('Mes')[['URmax', 'URmin']],
                    height=400
                )
                st.caption("Evolução da Umidade Relativa")
            
            # ===== DOWNLOAD =====
            st.markdown("---")
            st.subheader("💾 Download dos Dados Tratados")
            
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                # Download CSV
                csv_data = df_consolidado.to_csv(index=False, sep=';', decimal=',')
                st.download_button(
                    label="📥 Baixar como CSV (Excel)",
                    data=csv_data,
                    file_name="dados_inmet_consolidados.csv",
                    mime="text/csv"
                )
            
            with col_download2:
                # Download Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_consolidado.to_excel(writer, sheet_name='Dados_Consolidados', index=False)
                    if not df_detalhado.empty:
                        df_detalhado.to_excel(writer, sheet_name='Dados_Detalhados', index=False)
                
                excel_data = output.getvalue()
                st.download_button(
                    label="📊 Baixar como Excel (XLSX)",
                    data=excel_data,
                    file_name="dados_inmet_tratados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # ===== AMOSTRA DO RESULTADO =====
            st.markdown("---")
            st.subheader("📋 Formato Final dos Dados")
            st.markdown("""
            O arquivo tratado terá o seguinte formato:
            
            | Mes | Estacao | Tmax | Tmin | URmax | URmin | U2 |
            |-----|---------|------|------|-------|-------|-----|
            
            Onde:
            - **Mes**: Ano-Mês (ex: 2019-01)
            - **Estacao**: Nome da estação meteorológica
            - **Tmax**: Temperatura máxima mensal (°C)
            - **Tmin**: Temperatura mínima mensal (°C)
            - **URmax**: Umidade relativa máxima mensal (%)
            - **URmin**: Umidade relativa mínima mensal (%)
            - **U2**: Velocidade média do vento (m/s)
            """)
            
        else:
            st.error("❌ Não foi possível consolidar os dados. Verifique o formato do arquivo.")
            
    except Exception as e:
        st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
        st.info("Verifique se o arquivo está no formato correto (CSV do INMET com separador ponto e vírgula)")

else:
    # Exibir instruções quando nenhum arquivo foi carregado
    st.info("👆 Faça o upload de um arquivo CSV bruto do INMET para iniciar o processamento")
    
    with st.expander("📖 Instruções de uso"):
        st.markdown("""
        ### Como usar:
        
        1. **Baixe os dados brutos** do site do INMET (formato CSV)
        2. **Faça o upload** do arquivo usando o botão acima
        3. **Aguarde o processamento** automático dos dados
        4. **Faça o download** dos dados tratados em CSV ou Excel
        
        ### Transformações aplicadas:
        - ✅ Conversão de vírgula para ponto em valores numéricos
        - ✅ Padronização de nomes de colunas para inglês
        - ✅ Criação de coluna de data/hora corretamente formatada
        - ✅ Consolidação mensal dos dados (máximas, mínimas e médias)
        - ✅ Cálculo de estatísticas agregadas por mês
        
        ### Formato de saída:
        - **CSV**: Para importação em outras ferramentas
        - **Excel**: Com duas abas (consolidado e detalhado)
        """)

# ============================================================================
# RODAPÉ
# ============================================================================
st.markdown("---")
st.caption("Desenvolvido para tratamento de dados INMET | Converte dados horários brutos em consolidação mensal")
