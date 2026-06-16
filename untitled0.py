import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
import base64
import warnings
import requests
import json
import re
import math
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA E ESTILO
# ============================================================================

st.set_page_config(
    page_title="INMET Professional Processor - Sistema Completo de Análise Meteorológica",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Personalizado Avançado
st.markdown("""
<style>
    /* Estilo principal */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
    
    /* Cabeçalho principal */
    .main-header {
        background: linear-gradient(135deg, rgba(27, 94, 87, 0.95), rgba(15, 55, 70, 0.95));
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 25px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        background: linear-gradient(135deg, #e8f4f8, #a0c4d6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Cards de métricas */
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.05));
        border-radius: 20px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(76, 175, 80, 0.5);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    /* Cards de alerta */
    .warning-card {
        background: linear-gradient(135deg, rgba(255,193,7,0.15), rgba(255,152,0,0.08));
        border-left: 5px solid #ff9800;
        border-radius: 15px;
        padding: 1.2rem;
        margin: 1rem 0;
        backdrop-filter: blur(5px);
    }
    
    .critical-card {
        background: linear-gradient(135deg, rgba(244,67,54,0.15), rgba(229,57,53,0.08));
        border-left: 5px solid #f44336;
        border-radius: 15px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, rgba(76,175,80,0.12), rgba(56,142,60,0.08));
        border-left: 5px solid #4caf50;
        border-radius: 15px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    
    /* Tabs personalizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.05);
        padding: 8px;
        border-radius: 50px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.08);
        border-radius: 40px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2e7d32, #1b5e20);
        color: white !important;
        box-shadow: 0 2px 10px rgba(46,125,50,0.3);
    }
    
    /* DataFrame */
    .stDataFrame {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        overflow: hidden;
    }
    
    /* Botões */
    .stButton button {
        background: linear-gradient(135deg, #2e7d32, #1b5e20);
        border: none;
        border-radius: 30px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(46,125,50,0.4);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(46,125,50,0.2), rgba(27,94,32,0.1));
        border-radius: 15px;
        font-weight: 600;
        color: #81c784 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURAÇÃO DA IA GEMINI
# ============================================================================

GEMINI_API_KEY = "AQ.Ab8RN6J5uzUogvavjQyfFr3wGWEaJbdrW3oByqhWo2bm_mMxmQ"  # Insira sua chave API aqui

def consultar_gemini(prompt, contexto=""):
    """Consulta a API Gemini para análises climáticas"""
    try:
        if not GEMINI_API_KEY:
            return "⚠️ API Key do Gemini não configurada. Insira sua chave na variável GEMINI_API_KEY."
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
        
        prompt_completo = f"""Você é um meteorologista especialista em climatologia e engenheiro agrônomo. 
        Analise os seguintes dados climáticos e forneça uma interpretação profissional.

        CONTEXTO DOS DADOS:
        {contexto}

        PERGUNTA/ANÁLISE SOLICITADA:
        {prompt}

        Por favor, forneça uma análise que inclua:
        1. Identificação de padrões climáticos anormais
        2. Possível relação com fenômenos como El Niño, La Niña, ou outras oscilações climáticas
        3. Implicações para a agricultura (se aplicável)
        4. Recomendações práticas
        5. Use linguagem técnica mas acessível a profissionais da área

        RESPOSTA:
        """
        
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt_completo}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            resultado = response.json()
            if "candidates" in resultado and resultado["candidates"]:
                return resultado["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "Não foi possível obter uma análise detalhada da IA."
        elif response.status_code == 429:
            return "⚠️ Limite de requisições da API atingido. Tente novamente em alguns instantes."
        else:
            return f"⚠️ Erro na consulta da IA: Status {response.status_code}"
            
    except Exception as e:
        return f"⚠️ Erro ao consultar IA: {str(e)}"

# ============================================================================
# FUNÇÕES AVANÇADAS DE DETECÇÃO DE FORMATO
# ============================================================================

def detectar_formato_arquivo_avancado(arquivo_bytes):
    """Detecção avançada de formato de arquivo INMET"""
    
    # Ler primeiras linhas para análise
    conteudo = arquivo_bytes.getvalue().decode('utf-8', errors='ignore')[:5000]
    linhas = conteudo.split('\n')
    
    info = {
        'delimiter': None,
        'encoding': 'utf-8',
        'has_header': True,
        'date_format': None,
        'decimal_separator': ',',
        'columns_mapping': {},
        'skiprows': 0,
        'station_name': None,
        'latitude': None,
        'longitude': None,
        'altitude': None
    }
    
    # Detectar delimitador
    delimiters = [';', ',', '\t', '|']
    for delim in delimiters:
        if delim in linhas[0]:
            info['delimiter'] = delim
            break
    
    if not info['delimiter']:
        info['delimiter'] = ';'
    
    # Detectar separador decimal
    if ',' in conteudo and '.' not in conteudo:
        info['decimal_separator'] = ','
    elif '.' in conteudo and ',' not in conteudo:
        info['decimal_separator'] = '.'
    else:
        if re.search(r'\d+,\d+', conteudo):
            info['decimal_separator'] = ','
        else:
            info['decimal_separator'] = '.'
    
    # Detectar formato de data
    padroes_data = [
        (r'\d{2}/\d{2}/\d{4}', '%d/%m/%Y'),
        (r'\d{4}-\d{2}-\d{2}', '%Y-%m-%d'),
        (r'\d{2}-\d{2}-\d{4}', '%d-%m-%Y'),
        (r'\d{2}\.\d{2}\.\d{4}', '%d.%m.%Y')
    ]
    
    for padrao, formato in padroes_data:
        if re.search(padrao, conteudo):
            info['date_format'] = formato
            break
    
    if not info['date_format']:
        info['date_format'] = '%Y-%m-%d'
    
    # Extrair informações da estação do cabeçalho
    for i, linha in enumerate(linhas[:20]):
        if 'ESTACAO' in linha.upper() or 'ESTAÇÃO' in linha.upper():
            partes = linha.split(info['delimiter'])
            if len(partes) > 1:
                info['station_name'] = partes[1].strip()
        if 'LATITUDE' in linha.upper():
            partes = linha.split(info['delimiter'])
            if len(partes) > 1:
                try:
                    info['latitude'] = float(partes[1].replace(',', '.'))
                except:
                    pass
        if 'LONGITUDE' in linha.upper():
            partes = linha.split(info['delimiter'])
            if len(partes) > 1:
                try:
                    info['longitude'] = float(partes[1].replace(',', '.'))
                except:
                    pass
    
    return info

def ler_arquivo_inteligente(arquivo_bytes):
    """Leitura inteligente com auto-detecção"""
    info = detectar_formato_arquivo_avancado(arquivo_bytes)
    
    # Tentar diferentes estratégias de leitura
    estrategias = [
        {'skiprows': 0, 'header': 0},
        {'skiprows': 1, 'header': 0},
        {'skiprows': 0, 'header': 'infer'},
        {'skiprows': 8, 'header': 0},
        {'skiprows': 9, 'header': 0}
    ]
    
    df = None
    for estrategia in estrategias:
        try:
            df_temp = pd.read_csv(
                arquivo_bytes,
                sep=info['delimiter'],
                encoding=info['encoding'],
                skiprows=estrategia['skiprows'],
                header=estrategia['header'] if isinstance(estrategia['header'], int) else 'infer',
                on_bad_lines='skip'
            )
            
            # Verificar se leu corretamente
            if len(df_temp.columns) > 3 and len(df_temp) > 10:
                df = df_temp
                info['skiprows'] = estrategia['skiprows']
                break
        except:
            continue
    
    if df is None:
        raise ValueError("Não foi possível ler o arquivo. Verifique o formato.")
    
    # Corrigir separador decimal
    if info['decimal_separator'] == ',':
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.replace(',', '.')
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
    
    return df, info
# Adicione esta função após a leitura do arquivo
def diagnosticar_colunas(df):
    st.markdown("### 🔍 Diagnóstico de Colunas")
    st.write("**Colunas encontradas no arquivo:**")
    st.write(list(df.columns))
    
    # Verificar colunas de umidade
    col_umidade = [c for c in df.columns if any(x in c.lower() for x in ['umid', 'ur', 'umidade'])]
    st.write(f"**Colunas de umidade detectadas:** {col_umidade}")
    
    # Verificar colunas de vento
    col_vento = [c for c in df.columns if any(x in c.lower() for x in ['vento', 'vel', 'u2'])]
    st.write(f"**Colunas de vento detectadas:** {col_vento}")
    
    # Verificar colunas de temperatura
    col_temp = [c for c in df.columns if any(x in c.lower() for x in ['temp', 'temperatura'])]
    st.write(f"**Colunas de temperatura detectadas:** {col_temp}")
    
    # Verificar colunas de precipitação
    col_precip = [c for c in df.columns if any(x in c.lower() for x in ['precip', 'chuva'])]
    st.write(f"**Colunas de precipitação detectadas:** {col_precip}")
    
    return col_umidade, col_vento, col_temp, col_precip
# ============================================================================
# FUNÇÕES AVANÇADAS DE TRATAMENTO DE DADOS
# ============================================================================

def padronizar_colunas(df):
    """Padronização completa de nomes de colunas"""
    
    mapeamento_completo = {
        # Datas e horários
        'data': 'Date', 'DATA': 'Date', 'Data': 'Date', 'DT_DATA': 'Date',
        'hora': 'Time', 'HORA': 'Time', 'Hora': 'Time', 'HR_DATA': 'Time',
        'datahora': 'DateTime', 'DATAHORA': 'DateTime', 'DataHora': 'DateTime',
        
        # Temperaturas
        'temp_ins': 'Temp_Inst', 'temperatura_ins': 'Temp_Inst',
        'temp_inst': 'Temp_Inst', 'TEMP_INS': 'Temp_Inst',
        'temp_max': 'Tmax', 'temperatura_maxima': 'Tmax', 'TEMPERATURA MAXIMA': 'Tmax',
        'temp_maxima': 'Tmax', 'TEMP_MAX': 'Tmax', 'temp_max_inst': 'Tmax',
        'temp_min': 'Tmin', 'temperatura_minima': 'Tmin', 'TEMPERATURA MINIMA': 'Tmin',
        'temp_minima': 'Tmin', 'TEMP_MIN': 'Tmin', 'temp_min_inst': 'Tmin',
        
        # Umidade (ADICIONAR MAIS VARIAÇÕES)
        'umid_ins': 'UR_Inst', 'umidade_ins': 'UR_Inst', 'UMIDADE INST': 'UR_Inst',
        'umid_inst': 'UR_Inst', 'umidade_inst': 'UR_Inst', 'umidade_instantanea': 'UR_Inst',
        'ur_inst': 'UR_Inst', 'UR_inst': 'UR_Inst',
        'umid_max': 'URmax', 'umidade_maxima': 'URmax', 'UMIDADE MAXIMA': 'URmax',
        'ur_max': 'URmax', 'UR_max': 'URmax',
        'umid_min': 'URmin', 'umidade_minima': 'URmin', 'UMIDADE MINIMA': 'URmin',
        'ur_min': 'URmin', 'UR_min': 'URmin',
        'umidade': 'UR_Inst', 'umidade_relativa': 'UR_Inst',
        
        # Pressão
        'pressao_ins': 'Press_Inst', 'pressao_atm': 'Press_Inst',
        'pressao_max': 'Pressmax', 'pressao_min': 'Pressmin',
        
        # Vento (ADICIONAR MAIS VARIAÇÕES)
        'vel_vento': 'U2', 'velocidade_vento': 'U2', 'VEL_VENTO': 'U2',
        'vento_medio': 'U2', 'vento': 'U2', 'vel_vento_medio': 'U2',
        'vento_rajada': 'Raj_vento', 'vento_raj': 'Raj_vento', 'rajada': 'Raj_vento',
        'dir_vento': 'Dir_vento', 'direcao_vento': 'Dir_vento', 'vento_dir': 'Dir_vento',
        
        # Radiação e precipitação
        'radiacao': 'Rad_KJ', 'radiacao_solar': 'Rad_KJ', 'rad_solar': 'Rad_KJ',
        'precip': 'Precipitacao', 'chuva': 'Precipitacao', 'PRECIPITACAO': 'Precipitacao',
        'precipitacao_total': 'Precipitacao', 'precipitacao_mensal': 'Precipitacao',
        'precipitacao_instantanea': 'Precipitacao', 'precip_inst': 'Precipitacao',
        
        # Ponto de orvalho
        'pto_orvalho': 'Td_Inst', 'ponto_orvalho': 'Td_Inst',
        'pto_orvalho_max': 'Tdmax', 'pto_orvalho_min': 'Tdmin'
    }
        
    df_renomeado = df.copy()
    
    # Renomear colunas
    for col in df_renomeado.columns:
        col_lower = col.lower().strip()
        if col_lower in mapeamento_completo:
            df_renomeado.rename(columns={col: mapeamento_completo[col_lower]}, inplace=True)
        else:
            # Tentar matching fuzzy
            for chave, valor in mapeamento_completo.items():
                if chave in col_lower or col_lower in chave:
                    df_renomeado.rename(columns={col: valor}, inplace=True)
                    break
    
    return df_renomeado

def corrigir_datas(df):
    """Correção robusta de datas"""
    
    df_corrigido = df.copy()
    
    # Identificar coluna de data
    col_data = None
    for col in df_corrigido.columns:
        if col.lower() in ['date', 'data', 'dt', 'datetime']:
            col_data = col
            break
    
    if not col_data:
        # Procurar coluna que parece conter datas
        for col in df_corrigido.columns:
            amostra = df_corrigido[col].dropna().iloc[0] if len(df_corrigido[col].dropna()) > 0 else ''
            if isinstance(amostra, str) and ('/' in amostra or '-' in amostra):
                if len(amostra) > 6:
                    col_data = col
                    break
    
    if col_data:
        # Tentar diferentes formatos
        formatos = ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d']
        
        for formato in formatos:
            try:
                df_corrigido['Date'] = pd.to_datetime(df_corrigido[col_data], format=formato, errors='coerce')
                if df_corrigido['Date'].notna().sum() > len(df_corrigido) * 0.8:
                    break
            except:
                continue
        
        # Se falhou, tenta inferência automática
        if df_corrigido['Date'].isna().sum() > len(df_corrigido) * 0.5:
            df_corrigido['Date'] = pd.to_datetime(df_corrigido[col_data], errors='coerce')
    
    # Identificar coluna de hora
    col_hora = None
    for col in df_corrigido.columns:
        if col.lower() in ['time', 'hora', 'hr']:
            col_hora = col
            break
    
    if col_hora:
        # Padronizar formato de hora
        df_corrigido['Time_Str'] = df_corrigido[col_hora].astype(str).str.zfill(4)
        df_corrigido['Hour'] = df_corrigido['Time_Str'].str[:2]
        df_corrigido['Minute'] = df_corrigido['Time_Str'].str[2:4]
        df_corrigido['DateTime'] = pd.to_datetime(
            df_corrigido['Date'].dt.strftime('%Y-%m-%d') + ' ' + 
            df_corrigido['Hour'] + ':' + df_corrigido['Minute'],
            errors='coerce'
        )
    else:
        df_corrigido['DateTime'] = df_corrigido['Date']
    
    # Extrair componentes de data
    df_corrigido['Year'] = df_corrigido['DateTime'].dt.year
    df_corrigido['Month'] = df_corrigido['DateTime'].dt.month
    df_corrigido['Day'] = df_corrigido['DateTime'].dt.day
    df_corrigido['DayOfYear'] = df_corrigido['DateTime'].dt.dayofyear
    df_corrigido['Week'] = df_corrigido['DateTime'].dt.isocalendar().week
    df_corrigido['Season'] = df_corrigido['Month'].map({12: 'Summer', 1: 'Summer', 2: 'Summer',
                                                        3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
                                                        6: 'Winter', 7: 'Winter', 8: 'Winter',
                                                        9: 'Spring', 10: 'Spring', 11: 'Spring'})
    
    return df_corrigido

# ============================================================================
# FUNÇÕES AVANÇADAS DE QUALIDADE DE DADOS
# ============================================================================

def validar_consistencia_fisica(df):
    """
    Valida a consistência física dos dados meteorológicos.
    
    Verifica se os valores estão dentro de faixas plausíveis:
    - Temperatura: -50°C a 60°C
    - Umidade: 0% a 100%
    - Precipitação: >= 0 mm
    - Velocidade do vento: >= 0 m/s
    - Pressão: 800 hPa a 1100 hPa
    
    Retorna:
        df_validado: DataFrame com os dados originais
        alertas: Lista de dicionários com os registros inconsistentes
    """
    df_validado = df.copy()
    alertas = []
    
    # Verificar temperatura
    for col in ['Tmax', 'Tmin', 'Temp_Inst']:
        if col in df_validado.columns:
            mask_invalid = (df_validado[col] < -50) | (df_validado[col] > 60)
            if mask_invalid.any():
                alertas.append({
                    'variavel': col,
                    'quantidade': mask_invalid.sum(),
                    'exemplos': df_validado.loc[mask_invalid, 'DateTime'].dt.strftime('%Y-%m-%d %H:%M').tolist()[:5]
                })
    
    # Verificar umidade
    for col in ['UR_Inst', 'URmax', 'URmin']:
        if col in df_validado.columns:
            mask_invalid = (df_validado[col] < 0) | (df_validado[col] > 100)
            if mask_invalid.any():
                alertas.append({
                    'variavel': col,
                    'quantidade': mask_invalid.sum(),
                    'exemplos': df_validado.loc[mask_invalid, 'DateTime'].dt.strftime('%Y-%m-%d %H:%M').tolist()[:5]
                })
    
    # Verificar precipitação
    if 'Precipitacao' in df_validado.columns:
        mask_invalid = df_validado['Precipitacao'] < 0
        if mask_invalid.any():
            alertas.append({
                'variavel': 'Precipitacao',
                'quantidade': mask_invalid.sum(),
                'exemplos': df_validado.loc[mask_invalid, 'DateTime'].dt.strftime('%Y-%m-%d %H:%M').tolist()[:5]
            })
    
    # Verificar vento
    if 'U2' in df_validado.columns:
        mask_invalid = df_validado['U2'] < 0
        if mask_invalid.any():
            alertas.append({
                'variavel': 'U2',
                'quantidade': mask_invalid.sum(),
                'exemplos': df_validado.loc[mask_invalid, 'DateTime'].dt.strftime('%Y-%m-%d %H:%M').tolist()[:5]
            })
    
    # Verificar pressão
    for col in ['Press_Inst', 'Pressmax', 'Pressmin']:
        if col in df_validado.columns:
            mask_invalid = (df_validado[col] < 800) | (df_validado[col] > 1100)
            if mask_invalid.any():
                alertas.append({
                    'variavel': col,
                    'quantidade': mask_invalid.sum(),
                    'exemplos': df_validado.loc[mask_invalid, 'DateTime'].dt.strftime('%Y-%m-%d %H:%M').tolist()[:5]
                })
    
    return df_validado, alertas

def analise_completa_qualidade(df):
    """Análise completa da qualidade dos dados"""
    
    qualidade = {}
    
    # Identificar colunas numéricas
    colunas_numericas = ['Tmax', 'Tmin', 'Temp_Inst', 'UR_Inst', 'URmax', 'URmin', 
                         'Precipitacao', 'U2', 'Press_Inst']
    colunas_existentes = [c for c in colunas_numericas if c in df.columns]
    
    # Estatísticas de completude
    for col in colunas_existentes:
        total = len(df)
        presentes = df[col].notna().sum()
        percentual = (presentes / total) * 100
        
        qualidade[col] = {
            'total_registros': total,
            'dados_presentes': presentes,
            'dados_faltantes': total - presentes,
            'percentual_completo': round(percentual, 2),
            'classificacao': 'Boa' if percentual >= 95 else 'Regular' if percentual >= 80 else 'Crítica'
        }
    
    # Análise de outliers (usando IQR)
    outliers_info = {}
    for col in colunas_existentes:
        dados_validos = df[col].dropna()
        if len(dados_validos) > 0:
            Q1 = dados_validos.quantile(0.25)
            Q3 = dados_validos.quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 3 * IQR
            limite_superior = Q3 + 3 * IQR
            
            outliers = df[(df[col] < limite_inferior) | (df[col] > limite_superior)]
            
            outliers_info[col] = {
                'limite_inferior': round(limite_inferior, 2),
                'limite_superior': round(limite_superior, 2),
                'num_outliers': len(outliers),
                'percentual_outliers': round((len(outliers) / len(dados_validos)) * 100, 2)
            }
    
    # Análise de consistência temporal
    if 'DateTime' in df.columns:
        df_sorted = df.sort_values('DateTime')
        intervalos = df_sorted['DateTime'].diff().dropna()
        
        qualidade['temporal'] = {
            'inicio': df_sorted['DateTime'].min(),
            'fim': df_sorted['DateTime'].max(),
            'dias_totais': (df_sorted['DateTime'].max() - df_sorted['DateTime'].min()).days,
            'intervalo_medio': str(intervalos.mean()) if not intervalos.empty else 'N/A',
            'intervalo_mediano': str(intervalos.median()) if not intervalos.empty else 'N/A'
        }
    
    return qualidade, outliers_info

def preenchimento_inteligente_falhas(df, metodo='multivariado'):
    """
    Preenchimento avançado de falhas usando múltiplas técnicas.
    
    Métodos disponíveis:
    - interpolacao_linear: Interpolação linear entre pontos válidos
    - interpolacao_spline: Interpolação spline cúbica
    - media_movel: Média móvel com janela de 24 horas
    - multivariado: Combinação de métodos (recomendado)
    """
    
    df_filled = df.copy()
    
    # Identificar colunas numéricas
    colunas_numericas = ['Tmax', 'Tmin', 'Temp_Inst', 'UR_Inst', 'URmax', 'URmin', 
                         'Precipitacao', 'U2', 'Press_Inst']
    colunas_existentes = [c for c in colunas_numericas if c in df.columns]
    
    if metodo == 'interpolacao_linear':
        for col in colunas_existentes:
            df_filled[col] = df_filled[col].interpolate(method='linear', limit_direction='both', limit=24)
            
    elif metodo == 'interpolacao_spline':
        for col in colunas_existentes:
            df_filled[col] = df_filled[col].interpolate(method='spline', order=3, limit_direction='both')
            
    elif metodo == 'media_movel':
        for col in colunas_existentes:
            media_movel = df_filled[col].rolling(window=24, min_periods=6, center=True).mean()
            df_filled[col] = df_filled[col].fillna(media_movel)
            
    elif metodo == 'multivariado':
        # Combinação de métodos
        for col in colunas_existentes:
            # Primeiro, interpolação linear
            df_filled[col] = df_filled[col].interpolate(method='linear', limit_direction='both', limit=12)
            
            # Depois, média móvel para os remanescentes
            mascara = df_filled[col].isna()
            if mascara.any():
                media_movel = df_filled[col].rolling(window=48, min_periods=12, center=True).mean()
                df_filled.loc[mascara, col] = media_movel.loc[mascara]
            
            # Por último, preenchimento sazonal
            if 'Month' in df.columns:
                mascara = df_filled[col].isna()
                if mascara.any():
                    medias_mensais = df_filled.groupby('Month')[col].transform('mean')
                    df_filled.loc[mascara, col] = medias_mensais.loc[mascara]
    
    # Verificar se ainda há NaN e preencher com mediana
    for col in colunas_existentes:
        if df_filled[col].isna().any():
            df_filled[col].fillna(df_filled[col].median(), inplace=True)
    
    return df_filled

def detectar_eventos_extremos(df):
    """Detecção automática de eventos climáticos extremos"""
    
    eventos = []
    
    # Temperaturas extremas
    if 'Tmax' in df.columns:
        tmax_99 = df['Tmax'].quantile(0.99)
        dias_calor_extremo = df[df['Tmax'] > tmax_99]
        if len(dias_calor_extremo) > 0:
            eventos.append({
                'tipo': 'Calor Extremo',
                'descricao': f"{len(dias_calor_extremo)} dias com temperatura máxima acima de {tmax_99:.1f}°C",
                'datas': dias_calor_extremo['DateTime'].dt.strftime('%Y-%m-%d').tolist()[:10],
                'gravidade': 'Alta'
            })
    
    if 'Tmin' in df.columns:
        tmin_1 = df['Tmin'].quantile(0.01)
        dias_frio_extremo = df[df['Tmin'] < tmin_1]
        if len(dias_frio_extremo) > 0:
            eventos.append({
                'tipo': 'Frio Extremo',
                'descricao': f"{len(dias_frio_extremo)} dias com temperatura mínima abaixo de {tmin_1:.1f}°C",
                'datas': dias_frio_extremo['DateTime'].dt.strftime('%Y-%m-%d').tolist()[:10],
                'gravidade': 'Alta'
            })
    
    # Precipitação extrema
    if 'Precipitacao' in df.columns:
        # Dias com chuva intensa (> 50mm)
        chuva_intensa = df[df['Precipitacao'] > 50]
        if len(chuva_intensa) > 0:
            eventos.append({
                'tipo': 'Chuva Intensa',
                'descricao': f"{len(chuva_intensa)} dias com precipitação > 50mm",
                'datas': chuva_intensa['DateTime'].dt.strftime('%Y-%m-%d').tolist()[:10],
                'gravidade': 'Média'
            })
        
        # Dias com chuva torrencial (> 100mm)
        chuva_torrencial = df[df['Precipitacao'] > 100]
        if len(chuva_torrencial) > 0:
            eventos.append({
                'tipo': 'Chuva Torrencial',
                'descricao': f"{len(chuva_torrencial)} dias com precipitação > 100mm",
                'datas': chuva_torrencial['DateTime'].dt.strftime('%Y-%m-%d').tolist()[:10],
                'gravidade': 'Crítica'
            })
    
    # Ventos fortes
    if 'U2' in df.columns:
        ventos_fortes = df[df['U2'] > 10]
        if len(ventos_fortes) > 0:
            eventos.append({
                'tipo': 'Ventos Fortes',
                'descricao': f"{len(ventos_fortes)} dias com vento > 10 m/s",
                'datas': ventos_fortes['DateTime'].dt.strftime('%Y-%m-%d').tolist()[:10],
                'gravidade': 'Média'
            })
    
    # Ondas de calor (3+ dias consecutivos com Tmax acima da média)
    if 'Tmax' in df.columns and 'DateTime' in df.columns:
        df_sorted = df.sort_values('DateTime')
        media_tmax = df_sorted['Tmax'].mean()
        df_sorted['Acima_Media'] = df_sorted['Tmax'] > media_tmax
        
        # Detectar sequências consecutivas
        sequencia = 0
        ondas_calor = []
        for i, val in enumerate(df_sorted['Acima_Media']):
            if val:
                sequencia += 1
            else:
                if sequencia >= 3:
                    ondas_calor.append(sequencia)
                sequencia = 0
        if sequencia >= 3:
            ondas_calor.append(sequencia)
        
        if len(ondas_calor) > 0:
            eventos.append({
                'tipo': 'Onda de Calor',
                'descricao': f"{len(ondas_calor)} ondas de calor identificadas (mínimo 3 dias consecutivos)",
                'datas': [],
                'gravidade': 'Média'
            })
    
    return eventos

# ============================================================================
# FUNÇÕES DE CONSOLIDAÇÃO E INDICADORES
# ============================================================================

def consolidacao_avancada_por_mes(df):
    """Consolidação avançada com tratamento diferenciado por variável"""
    
    df_temp = df.copy()
    
    # Criar período mês-ano
    df_temp['Period'] = df_temp['DateTime'].dt.to_period('M')
    
    # Dicionário de funções de agregação por tipo de variável
    agregacoes = {
        'Tmax': 'max',                    # Temperatura máxima: valor máximo absoluto
        'Tmin': 'min',                    # Temperatura mínima: valor mínimo absoluto
        'Temp_Inst': 'mean',              # Temperatura instantânea: média
        'UR_Inst': 'mean',                # Umidade instantânea: média
        'URmax': 'max',                   # Umidade máxima: valor máximo
        'URmin': 'min',                   # Umidade mínima: valor mínimo
        'Precipitacao': 'sum',            # Precipitação: soma total
        'U2': 'mean',                     # Velocidade do vento: média
        'Press_Inst': 'mean',             # Pressão: média
        'Rad_KJ': 'sum'                   # Radiação: soma total
    }
    
    # Filtrar apenas colunas existentes
    agregacoes_filtradas = {k: v for k, v in agregacoes.items() if k in df.columns}
    
    # Aplicar agregação
    df_consolidado = df_temp.groupby('Period').agg(agregacoes_filtradas).reset_index()
    
    # Converter período para string
    df_consolidado['Mes'] = df_consolidado['Period'].astype(str)
    df_consolidado['Ano'] = df_consolidado['Period'].dt.year
    df_consolidado['Numero_Mes'] = df_consolidado['Period'].dt.month
    
    # Ordenar por data
    df_consolidado = df_consolidado.sort_values('Period').reset_index(drop=True)
    
    # Adicionar médias móveis
    if 'Temp_Inst' in df_consolidado.columns:
        df_consolidado['Temp_Media_Movel_3M'] = df_consolidado['Temp_Inst'].rolling(window=3, min_periods=1).mean()
        df_consolidado['Temp_Media_Movel_12M'] = df_consolidado['Temp_Inst'].rolling(window=12, min_periods=1).mean()
    
    if 'Precipitacao' in df_consolidado.columns:
        df_consolidado['Precip_Acumulada_Ano'] = df_consolidado.groupby('Ano')['Precipitacao'].cumsum()
    
    return df_consolidado

def calcular_indicadores_agricolas_avancados(df_diario, df_mensal, latitude=-16.0):
    """
    Cálculo avançado de indicadores agrícolas.
    
    Inclui:
    - GDD (Graus-Dia de Desenvolvimento) com Tb=10°C
    - Horas de Frio para limiares de 7°C, 10°C e 13°C
    - ETo (Evapotranspiração Potencial) pelo método de Hargreaves com Ra calculado
    - Índice de Aridez (P/ETo)
    - Amplitude Térmica Média
    """
    
    indicadores = df_mensal[['Mes', 'Ano', 'Numero_Mes']].copy()
    
    # ===== GDD (Graus-Dia de Desenvolvimento) =====
    if 'Tmax' in df_diario.columns and 'Tmin' in df_diario.columns:
        t_base = 10  # Temperatura base para culturas tropicais
        t_otima = 30  # Temperatura ótima
        
        # Cálculo diário
        t_media_diaria = (df_diario['Tmax'] + df_diario['Tmin']) / 2
        gdd_diario = np.maximum(0, np.minimum(t_otima, t_media_diaria) - t_base)
        df_diario['GDD'] = gdd_diario
        
        # Acumulado mensal e anual
        gdd_mensal = df_diario.groupby(df_diario['DateTime'].dt.to_period('M'))['GDD'].sum().reset_index()
        gdd_mensal.columns = ['Period', 'GDD_Acumulado']
        gdd_mensal['Mes'] = gdd_mensal['Period'].astype(str)
        
        indicadores = indicadores.merge(gdd_mensal[['Mes', 'GDD_Acumulado']], on='Mes', how='left')
        
        # GDD acumulado no ano
        df_diario['GDD_Acumulado_Ano'] = df_diario.groupby(df_diario['DateTime'].dt.year)['GDD'].cumsum()
        gdd_anual = df_diario.groupby(df_diario['DateTime'].dt.to_period('M'))['GDD_Acumulado_Ano'].last().reset_index()
        gdd_anual.columns = ['Period', 'GDD_Ano_Acumulado']
        gdd_anual['Mes'] = gdd_anual['Period'].astype(str)
        indicadores = indicadores.merge(gdd_anual[['Mes', 'GDD_Ano_Acumulado']], on='Mes', how='left')
    
    # ===== Horas de Frio =====
    if 'Tmin' in df_diario.columns:
        for limiar in [7, 10, 13]:
            horas_frio = (df_diario['Tmin'] <= limiar).astype(int)
            df_diario[f'Horas_Frio_{limiar}C'] = horas_frio
            
            horas_mensal = df_diario.groupby(df_diario['DateTime'].dt.to_period('M'))[f'Horas_Frio_{limiar}C'].sum().reset_index()
            horas_mensal.columns = ['Period', f'Horas_Frio_{limiar}C']
            horas_mensal['Mes'] = horas_mensal['Period'].astype(str)
            indicadores = indicadores.merge(horas_mensal[['Mes', f'Horas_Frio_{limiar}C']], on='Mes', how='left')
    
    # ===== Evapotranspiração Potencial (ETo) - Método Hargreaves melhorado =====
    if all(x in df_diario.columns for x in ['Tmax', 'Tmin']):
        import math
        
        # Radiação extraterrestre (Ra) por latitude
        def calc_ra(lat, dia_ano):
            """
            Calcula a radiação solar no topo da atmosfera (Ra) em MJ/m²/dia.
            
            Baseado na equação da FAO (Allen et al., 1998).
            
            Parâmetros:
                lat: latitude em graus (negativo para Sul)
                dia_ano: dia do ano (1-365/366)
            
            Retorna:
                Ra em MJ/m²/dia
            """
            Gsc = 0.0820  # Constante solar (MJ/m²/min)
            phi = math.radians(lat)
            delta = 0.4093 * math.sin(2 * math.pi / 365 * dia_ano - 1.405)
            ws = math.acos(-math.tan(phi) * math.tan(delta))
            dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * dia_ano)
            ra = (24 * 60 / math.pi) * Gsc * dr * (ws * math.sin(phi) * math.sin(delta) + 
                                                   math.cos(phi) * math.cos(delta) * math.sin(ws))
            return ra
        
        df_diario['Dia_Ano'] = df_diario['DateTime'].dt.dayofyear
        df_diario['Ra'] = df_diario['Dia_Ano'].apply(lambda d: calc_ra(latitude, d))
        
        t_media_diaria = (df_diario['Tmax'] + df_diario['Tmin']) / 2
        amplitude = df_diario['Tmax'] - df_diario['Tmin']
        
        # ETo pelo método Hargreaves
        eto_diario = 0.0023 * df_diario['Ra'] * np.sqrt(amplitude) * (t_media_diaria + 17.8)
        df_diario['ETo'] = np.maximum(0, eto_diario)
        
        # ETo mensal
        eto_mensal = df_diario.groupby(df_diario['DateTime'].dt.to_period('M'))['ETo'].sum().reset_index()
        eto_mensal.columns = ['Period', 'ETo_Total_mm']
        eto_mensal['Mes'] = eto_mensal['Period'].astype(str)
        indicadores = indicadores.merge(eto_mensal[['Mes', 'ETo_Total_mm']], on='Mes', how='left')
    
    # ===== Índice de Aridez =====
    if 'Precipitacao' in df_diario.columns and 'ETo' in df_diario.columns:
        # Aridez mensal (P/ETP)
        df_diario['P_ETo'] = df_diario['Precipitacao'] / df_diario['ETo'].replace(0, 0.1)
        
        aridez_mensal = df_diario.groupby(df_diario['DateTime'].dt.to_period('M'))['P_ETo'].mean().reset_index()
        aridez_mensal.columns = ['Period', 'Indice_Aridade']
        aridez_mensal['Mes'] = aridez_mensal['Period'].astype(str)
        indicadores = indicadores.merge(aridez_mensal[['Mes', 'Indice_Aridade']], on='Mes', how='left')
        
        # Classificação do índice de aridez
        def classificar_aridez(indice):
            if pd.isna(indice):
                return 'N/A'
            elif indice < 0.2:
                return 'Hiperárido'
            elif indice < 0.5:
                return 'Árido'
            elif indice < 0.65:
                return 'Semiárido'
            elif indice < 1.0:
                return 'Subúmido Seco'
            else:
                return 'Úmido'
        
        indicadores['Classificacao_Aridade'] = indicadores['Indice_Aridade'].apply(classificar_aridez)
    
    # ===== Oscilações Térmicas =====
    if 'Tmax' in df_diario.columns and 'Tmin' in df_diario.columns:
        df_diario['Amplitude_Termica'] = df_diario['Tmax'] - df_diario['Tmin']
        
        amplitude_mensal = df_diario.groupby(df_diario['DateTime'].dt.to_period('M'))['Amplitude_Termica'].mean().reset_index()
        amplitude_mensal.columns = ['Period', 'Amplitude_Termica_Media']
        amplitude_mensal['Mes'] = amplitude_mensal['Period'].astype(str)
        indicadores = indicadores.merge(amplitude_mensal[['Mes', 'Amplitude_Termica_Media']], on='Mes', how='left')
    
    return indicadores, df_diario

# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO AVANÇADA
# ============================================================================

def gerar_climograma_streamlit(df_mensal):
    """
    Geração de climograma usando streamlit nativo e Vega-Lite.
    
    Verifica a existência das colunas necessárias e exibe os gráficos disponíveis.
    """
    
    # Verificar se o DataFrame está vazio
    if df_mensal is None or df_mensal.empty:
        st.warning("⚠️ Nenhum dado disponível para gerar o climograma.")
        return
    
    # Criar uma cópia segura
    df = df_mensal.copy()
    
    # Verificar quais colunas existem
    colunas_disponiveis = df.columns.tolist()
    
    # Mapear possíveis nomes de colunas
    mapeamento_mes = ['Mes', 'Mês', 'mes', 'mês', 'Period', 'period', 'ANO_MES', 'ano_mes']
    mapeamento_precip = ['Precipitacao', 'precipitacao', 'Precip', 'precip', 'CHUVA', 'chuva', 'PRECIPITACAO']
    mapeamento_temp = ['Temp_Inst', 'temp_inst', 'Temperatura', 'temperatura', 'TEMP_MEDIA', 'temp_media']
    
    # Encontrar a coluna correta para Mês
    col_mes = None
    for col in mapeamento_mes:
        if col in colunas_disponiveis:
            col_mes = col
            break
    if col_mes is None and len(df) > 0:
        # Usar o índice se não encontrar coluna de mês
        df['Mes'] = df.index.astype(str)
        col_mes = 'Mes'
    
    # Encontrar a coluna correta para Precipitação
    col_precip = None
    for col in mapeamento_precip:
        if col in colunas_disponiveis:
            col_precip = col
            break
    
    # Encontrar a coluna correta para Temperatura
    col_temp = None
    for col in mapeamento_temp:
        if col in colunas_disponiveis:
            col_temp = col
            break
    
    # Se não encontrar temperatura, tentar Tmax ou Tmin
    if col_temp is None and 'Tmax' in colunas_disponiveis:
        col_temp = 'Tmax'
    if col_temp is None and 'Tmin' in colunas_disponiveis:
        col_temp = 'Tmin'
    
    # Preparar DataFrame para exibição
    df_exibicao = df[[col_mes]].copy()
    df_exibicao.rename(columns={col_mes: 'Mes'}, inplace=True)
    
    # Adicionar precipitação se disponível
    if col_precip:
        df_exibicao['Precipitacao'] = df[col_precip]
    else:
        df_exibicao['Precipitacao'] = 0
    
    # Adicionar temperatura se disponível
    if col_temp:
        df_exibicao['Temp_Inst'] = df[col_temp]
    else:
        df_exibicao['Temp_Inst'] = 0
    
    # ===== GRÁFICO 1: Precipitação =====
    st.markdown("### ☔ Precipitação Mensal")
    if col_precip and df_exibicao['Precipitacao'].sum() > 0:
        df_precip = df_exibicao[['Mes', 'Precipitacao']].set_index('Mes')
        st.bar_chart(df_precip, height=350, color='#3498db')
    else:
        st.info("📌 Dados de precipitação não disponíveis para este arquivo")
    
    # ===== GRÁFICO 2: Temperatura =====
    st.markdown("### 🌡️ Temperatura")
    if col_temp and df_exibicao['Temp_Inst'].sum() != 0:
        df_temp = df_exibicao[['Mes', 'Temp_Inst']].set_index('Mes')
        st.line_chart(df_temp, height=350, color='#e74c3c')
    else:
        st.info("📌 Dados de temperatura não disponíveis para este arquivo")
    
    st.markdown("---")
    
    # ===== GRÁFICO COMBINADO INTERATIVO (Vega-Lite) =====
    if col_precip and col_temp and df_exibicao['Precipitacao'].sum() > 0 and df_exibicao['Temp_Inst'].sum() != 0:
        st.markdown("### 📊 Análise Combinada (Precipitação vs Temperatura)")
        
        # Preparar dados para Vega-Lite
        df_vega = df_exibicao[['Mes', 'Precipitacao', 'Temp_Inst']].copy()
        df_vega['Precipitacao'] = df_vega['Precipitacao'].round(1)
        df_vega['Temp_Inst'] = df_vega['Temp_Inst'].round(1)
        
        # Gráfico combinado com Vega-Lite
        chart = {
            "title": "Relação entre Precipitação e Temperatura",
            "mark": {"type": "bar", "tooltip": True},
            "encoding": {
                "x": {"field": "Mes", "type": "ordinal", "title": "Mês", "sort": None},
                "y": {"field": "Precipitacao", "type": "quantitative", "title": "Precipitação (mm)"},
                "tooltip": [
                    {"field": "Mes", "type": "nominal", "title": "Mês"},
                    {"field": "Precipitacao", "type": "quantitative", "title": "Precipitação (mm)", "format": ".1f"},
                    {"field": "Temp_Inst", "type": "quantitative", "title": "Temperatura (°C)", "format": ".1f"}
                ]
            }
        }
        
        line_chart = {
            "mark": {"type": "line", "color": "red", "tooltip": True},
            "encoding": {
                "y": {"field": "Temp_Inst", "type": "quantitative", "title": "Temperatura (°C)", "axis": {"titleColor": "red"}}
            }
        }
        
        chart['layer'] = [{"mark": "bar"}, {"mark": {"type": "line", "color": "red"}}]
        chart['encoding']['y'] = {"field": "Precipitacao", "type": "quantitative", "title": "Precipitação (mm)"}
        chart['encoding']['y2'] = {"field": "Temp_Inst", "type": "quantitative", "title": "Temperatura (°C)"}
        
        st.vega_lite_chart(df_vega, {
            "layer": [
                {"mark": {"type": "bar", "tooltip": True}, "encoding": {"y": {"field": "Precipitacao", "type": "quantitative", "title": "Precipitação (mm)"}}},
                {"mark": {"type": "line", "color": "red", "tooltip": True}, "encoding": {"y": {"field": "Temp_Inst", "type": "quantitative", "title": "Temperatura (°C)", "axis": {"titleColor": "red"}}}}],
            "encoding": {"x": {"field": "Mes", "type": "ordinal", "title": "Mês", "sort": None}},
            "title": "Precipitação e Temperatura"
        }, use_container_width=True)
    
    st.markdown("---")
    
    # ===== GRÁFICO COMBINADO =====
    st.markdown("### 📊 Dados Mensais Consolidados")
    
    # Mostrar apenas colunas que existem
    colunas_para_mostrar = ['Mes']
    if col_precip:
        colunas_para_mostrar.append('Precipitacao')
    if col_temp:
        colunas_para_mostrar.append('Temp_Inst')
    
    # Adicionar outras colunas úteis se disponíveis
    for col in ['GDD_Acumulado', 'ETo_Total_mm', 'Horas_Frio_10C', 'Tmax', 'Tmin', 'UR_Inst', 'U2']:
        if col in df.columns:
            colunas_para_mostrar.append(col)
            df_exibicao[col] = df[col]
    
    st.dataframe(df_exibicao[colunas_para_mostrar].set_index('Mes'), use_container_width=True)
    
    # ===== INDICADORES ACUMULADOS =====
    st.markdown("---")
    st.markdown("### 🌱 Indicadores Acumulados")
    
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    
    with col_i1:
        if col_precip and df_exibicao['Precipitacao'].sum() > 0:
            st.metric("☔ Precipitação Total", f"{df_exibicao['Precipitacao'].sum():.0f} mm")
        else:
            st.metric("☔ Precipitação Total", "N/D")
    
    with col_i2:
        if 'GDD_Acumulado' in df.columns and df['GDD_Acumulado'].sum() > 0:
            st.metric("🌾 GDD Total", f"{df['GDD_Acumulado'].sum():.0f}")
        else:
            st.metric("🌾 GDD Total", "N/D")
    
    with col_i3:
        if 'ETo_Total_mm' in df.columns and df['ETo_Total_mm'].sum() > 0:
            st.metric("💧 ETo Total", f"{df['ETo_Total_mm'].sum():.0f} mm")
        else:
            st.metric("💧 ETo Total", "N/D")
    
    with col_i4:
        if 'Horas_Frio_10C' in df.columns and df['Horas_Frio_10C'].sum() > 0:
            st.metric("❄️ Horas de Frio", f"{df['Horas_Frio_10C'].sum():.0f} h")
        else:
            st.metric("❄️ Horas de Frio", "N/D")
    
    # ===== GRÁFICO GDD =====
    if 'GDD_Acumulado' in df.columns and df['GDD_Acumulado'].sum() > 0:
        st.markdown("---")
        st.markdown("### 📈 Evolução do GDD Acumulado")
        df_gdd = df_exibicao[['Mes', 'GDD_Acumulado']].set_index('Mes')
        st.line_chart(df_gdd, height=300, color='#4caf50')
    
    # ===== GRÁFICO ETo =====
    if 'ETo_Total_mm' in df.columns and df['ETo_Total_mm'].sum() > 0:
        st.markdown("### 💧 Evolução da ETo Mensal")
        df_eto = df_exibicao[['Mes', 'ETo_Total_mm']].set_index('Mes')
        st.line_chart(df_eto, height=300, color='#9c27b0')


def gerar_relatorio_pdf_completo(df_mensal, df_indicadores, qualidade, eventos_extremos, nome_estacao, info_estacao):
    """Geração de relatório PDF completo e profissional"""
    
    data_geracao = datetime.now().strftime('%d/%m/%Y às %H:%M')
    
    # Preparar tabela de dados mensais
    tabela_mensal = ""
    for _, row in df_mensal.iterrows():
        tabela_mensal += f"""
        <tr>
            <td>{row['Mes']}</td>
            <td>{row.get('Temp_Inst', 0):.1f}</td>
            <td>{row.get('Tmax', 0):.1f}</td>
            <td>{row.get('Tmin', 0):.1f}</td>
            <td>{row.get('Precipitacao', 0):.1f}</td>
            <td>{row.get('UR_Inst', 0):.1f}</td>
            <td>{row.get('U2', 0):.1f}</td>
        </tr>
        """
    
    # Preparar tabela de indicadores
    tabela_indicadores = ""
    if df_indicadores is not None and len(df_indicadores) > 0:
        for _, row in df_indicadores.iterrows():
            tabela_indicadores += f"""
            <tr>
                <td>{row.get('Mes', 'N/A')}</td>
                <td>{row.get('GDD_Acumulado', 0):.0f}</td>
                <td>{row.get('Horas_Frio_10C', 0):.0f}</td>
                <td>{row.get('ETo_Total_mm', 0):.1f}</td>
                <td>{row.get('Indice_Aridade', 0):.2f}</td>
                <td>{row.get('Classificacao_Aridade', 'N/A')}</td>
            </tr>
            """
    
    # Preparar tabela de qualidade
    tabela_qualidade = ""
    for var, info in qualidade.items():
        if var != 'temporal' and isinstance(info, dict):
            tabela_qualidade += f"""
            <tr>
                <td>{var}</td>
                <td>{info.get('percentual_completo', 0):.1f}%</td>
                <td>{info.get('classificacao', 'N/A')}</td>
                <td>{info.get('num_outliers', 0)}</td>
            </tr>
            """
    
    # Preparar cards de eventos
    eventos_html = ""
    for evento in eventos_extremos:
        classe = 'critical-card' if evento['gravidade'] == 'Crítica' else 'warning-card' if evento['gravidade'] == 'Alta' else 'success-card'
        eventos_html += f"""
            <div class="{classe}">
                <strong>🔴 {evento['tipo']}</strong><br>
                {evento['descricao']}<br>
                <small>Gravidade: {evento['gravidade']}</small>
            </div>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Relatório Meteorológico Completo - {nome_estacao}</title>
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            .header {{
                background: linear-gradient(135deg, #1a5f7a, #0d3b4f);
                color: white;
                padding: 40px;
                text-align: center;
                border-radius: 10px 10px 0 0;
            }}
            .header h1 {{
                margin: 0;
                font-size: 28px;
            }}
            .header p {{
                margin: 10px 0 0;
                opacity: 0.9;
            }}
            .section {{
                background: white;
                margin: 20px;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                page-break-inside: avoid;
            }}
            .section h2 {{
                color: #1a5f7a;
                border-bottom: 3px solid #4caf50;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 10px;
                text-align: center;
            }}
            th {{
                background: linear-gradient(135deg, #1a5f7a, #0d3b4f);
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #e8f4f8, #d4eaf1);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 28px;
                font-weight: bold;
                color: #1a5f7a;
            }}
            .metric-label {{
                font-size: 12px;
                color: #666;
                margin-top: 5px;
            }}
            .warning-card {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }}
            .critical-card {{
                background: #f8d7da;
                border-left: 4px solid #f44336;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }}
            .success-card {{
                background: #d4edda;
                border-left: 4px solid #28a745;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                font-size: 12px;
                color: #999;
                border-top: 1px solid #eee;
                margin-top: 20px;
            }}
            @media print {{
                body {{
                    background: white;
                    padding: 0;
                }}
                .section {{
                    break-inside: avoid;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌾 RELATÓRIO METEOROLÓGICO COMPLETO</h1>
                <h2>{nome_estacao}</h2>
                <p>Latitude: {info_estacao.get('latitude', 'N/A')}° | 
                   Longitude: {info_estacao.get('longitude', 'N/A')}° | 
                   Altitude: {info_estacao.get('altitude', 'N/A')} m</p>
                <p>Gerado em: {data_geracao}</p>
            </div>
            
            <div class="section">
                <h2>📊 Resumo Estatístico do Período</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{df_mensal['Temp_Inst'].mean():.1f}°C</div>
                        <div class="metric-label">Temperatura Média</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{df_mensal['Tmax'].max():.1f}°C</div>
                        <div class="metric-label">Temperatura Máxima Absoluta</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{df_mensal['Tmin'].min():.1f}°C</div>
                        <div class="metric-label">Temperatura Mínima Absoluta</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{df_mensal['Precipitacao'].sum():.0f} mm</div>
                        <div class="metric-label">Precipitação Total</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>📅 Dados Mensais Consolidados</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Mês</th>
                            <th>Temp Média (°C)</th>
                            <th>Tmax (°C)</th>
                            <th>Tmin (°C)</th>
                            <th>Precipitação (mm)</th>
                            <th>Umidade (%)</th>
                            <th>Vento (m/s)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {tabela_mensal}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>🌱 Indicadores Agrícolas e Climáticos</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Mês</th>
                            <th>GDD Acumulado</th>
                            <th>Horas de Frio (10°C)</th>
                            <th>ETo Total (mm)</th>
                            <th>Índice de Aridez</th>
                            <th>Classificação</th>
                        </tr>
                    </thead>
                    <tbody>
                        {tabela_indicadores}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>⚠️ Eventos Climáticos Extremos Detectados</h2>
                {eventos_html if eventos_extremos else "<p>✅ Nenhum evento climático extremo detectado no período analisado</p>"}
            </div>
            
            <div class="section">
                <h2>📈 Análise de Qualidade dos Dados</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Variável</th>
                            <th>Completude (%)</th>
                            <th>Classificação</th>
                            <th>Outliers</th>
                        </tr>
                    </thead>
                    <tbody>
                        {tabela_qualidade}
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                <p>Relatório gerado automaticamente pelo INMET Smart Processor</p>
                <p>Este relatório pode ser utilizado para fins acadêmicos, científicos e de planejamento agrícola</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

# ============================================================================
# FUNÇÃO PRINCIPAL DE PROCESSAMENTO
# ============================================================================

def pipeline_completo_processamento(arquivo_bytes, config):
    """Pipeline completo de processamento de dados INMET"""
    
    resultados = {}
    
    try:
        # Passo 1: Detecção de formato
        with st.spinner("🔍 Detectando formato do arquivo..."):
            df, info_estacao = ler_arquivo_inteligente(arquivo_bytes)
            resultados['info_estacao'] = info_estacao
        
        # Passo 2: Padronização de colunas
        with st.spinner("📋 Padronizando colunas..."):
            df = padronizar_colunas(df)
        
        # Passo 3: Correção de datas
        with st.spinner("📅 Corrigindo datas..."):
            df = corrigir_datas(df)
        
        # Passo 4: Validação de consistência física
        with st.spinner("🔬 Validando consistência física dos dados..."):
            df, alertas_fisicos = validar_consistencia_fisica(df)
            if alertas_fisicos:
                st.warning(f"⚠️ Foram encontrados {len(alertas_fisicos)} tipos de inconsistências físicas nos dados.")
                with st.expander("📋 Detalhes das inconsistências encontradas"):
                    for alerta in alertas_fisicos:
                        st.write(f"**Variável:** {alerta['variavel']}")
                        st.write(f"**Quantidade de registros inconsistentes:** {alerta['quantidade']}")
                        if alerta['exemplos']:
                            st.write(f"**Exemplos (data/hora):** {', '.join(alerta['exemplos'][:3])}")
                        st.markdown("---")
        
        # Passo 5: Análise de qualidade
        with st.spinner("📊 Analisando qualidade dos dados..."):
            qualidade, outliers = analise_completa_qualidade(df)
            resultados['qualidade'] = qualidade
            resultados['outliers'] = outliers
        
        # Passo 6: Preenchimento de falhas
        with st.spinner("🔄 Preenchendo falhas nos dados..."):
            metodo_falhas = config.get('metodo_falhas', 'multivariado')
            df = preenchimento_inteligente_falhas(df, metodo=metodo_falhas)
        
        # Passo 7: Detecção de eventos extremos
        with st.spinner("⚠️ Detectando eventos climáticos extremos..."):
            eventos_extremos = detectar_eventos_extremos(df)
            resultados['eventos_extremos'] = eventos_extremos
        
        # Passo 8: Consolidação mensal
        with st.spinner("📆 Consolidando dados por mês..."):
            df_mensal = consolidacao_avancada_por_mes(df)
            resultados['dados_mensais'] = df_mensal
        
        # Passo 9: Cálculo de indicadores agrícolas
        with st.spinner("🌱 Calculando indicadores agrícolas..."):
            latitude = config.get('latitude', info_estacao.get('latitude', -16.0))
            df_indicadores, df = calcular_indicadores_agricolas_avancados(df, df_mensal, latitude)
            resultados['indicadores'] = df_indicadores
            resultados['dados_diarios'] = df
        
        return resultados
        
    except Exception as e:
        st.error(f"❌ Erro no processamento: {str(e)}")
        return None

# ============================================================================
# INTERFACE PRINCIPAL DO STREAMLIT
# ============================================================================

def main():
    # Cabeçalho
    st.markdown("""
    <div class="main-header">
        <h1>🌾 INMET Smart Processor</h1>
        <p style="color: #a0c4d6; font-size: 1.1rem;">
            Sistema Profissional de Tratamento, Análise e Visualização de Dados Meteorológicos
        </p>
        <p style="color: #81c784; font-size: 0.9rem;">
            🔬 IA Integrada | 📊 Indicadores Agrícolas | 🌡️ Detecção de Eventos Extremos | 📈 Relatórios Automáticos
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Configurações
    with st.sidebar:
        st.markdown("## ⚙️ Configurações Avançadas")
        
        uploaded_file = st.file_uploader(
            "📂 **Upload do Arquivo INMET**",
            type=['csv', 'txt', 'dat'],
            help="Arquivos CSV, TXT ou DAT baixados do portal do INMET"
        )
        
        st.markdown("---")
        
        st.markdown("### 🔧 Métodos de Preenchimento")
        metodo_falhas = st.selectbox(
            "**Técnica para preenchimento de falhas:**",
            ["multivariado", "interpolacao_linear", "interpolacao_spline", "media_movel"],
            format_func=lambda x: {
                "multivariado": "🧠 Multivariado (Recomendado)",
                "interpolacao_linear": "📈 Interpolação Linear",
                "interpolacao_spline": "📊 Spline Cúbico",
                "media_movel": "📉 Média Móvel"
            }[x]
        )
        
        st.markdown("---")
        
        st.markdown("### 🌍 Parâmetros Regionais")
        latitude = st.number_input(
            "**Latitude da Estação (°):**",
            value=-16.0,
            step=0.1,
            format="%.1f",
            help="Use valores negativos para Sul do Equador"
        )
        
        st.markdown("---")
        
        st.markdown("### 🤖 IA Gemini")
        usar_ia = st.checkbox("**Ativar análise com IA Gemini**", value=True)
        if usar_ia:
            st.info("A IA analisará padrões climáticos e fenômenos como El Niño/La Niña")
        
        st.markdown("---")
        
        st.markdown("### 📄 Relatório")
        incluir_eventos = st.checkbox("**Incluir análise de eventos extremos**", value=True)
        incluir_indicadores = st.checkbox("**Incluir indicadores agrícolas**", value=True)
        
        st.markdown("---")
        st.caption("🔬 Desenvolvido para análise profissional de dados meteorológicos")
        st.caption("Versão 2.0 - Pipeline Completo")
    
    # Área principal
    if uploaded_file is not None:
        
        # Configurações do pipeline
        config = {
            'metodo_falhas': metodo_falhas,
            'latitude': latitude,
            'usar_ia': usar_ia,
            'incluir_eventos': incluir_eventos,
            'incluir_indicadores': incluir_indicadores
        }
        
        # Processar dados
        with st.status("🔄 **Processando dados meteorológicos...**", expanded=True) as status:
            st.write("📁 Lendo e analisando arquivo...")
            resultados = pipeline_completo_processamento(uploaded_file, config)
            
            if resultados:
                status.update(label="✅ Processamento concluído!", state="complete")
        
        if resultados:
            # Extrair resultados
            df_mensal = resultados['dados_mensais']
            df_indicadores = resultados.get('indicadores')
            qualidade = resultados.get('qualidade', {})
            eventos = resultados.get('eventos_extremos', [])
            info_estacao = resultados.get('info_estacao', {})
            df_diario = resultados.get('dados_diarios', pd.DataFrame())
            
            nome_estacao = info_estacao.get('station_name', 'Estação Meteorológica')
            
            # Alertas de qualidade
            alertas_criticos = []
            for var, info in qualidade.items():
                if var != 'temporal' and isinstance(info, dict):
                    if info.get('percentual_completo', 100) < 70:
                        alertas_criticos.append(f"⚠️ {var}: {info['percentual_completo']:.1f}% de dados válidos")
            
            if alertas_criticos:
                with st.expander("⚠️ **Alertas de Qualidade dos Dados**", expanded=True):
                    for alerta in alertas_criticos[:5]:
                        st.warning(alerta)
                    if len(alertas_criticos) > 5:
                        st.info(f"... e mais {len(alertas_criticos) - 5} alertas")
            
            # Tabs principais
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 **Dados Consolidados**",
                "🌱 **Indicadores Agrícolas**",
                "📈 **Visualizações**",
                "⚠️ **Eventos Extremos**",
                "🤖 **Análise IA**",
                "📥 **Downloads**"
            ])
            
            # Tab 1: Dados Consolidados
            with tab1:
                st.markdown("### 📊 Tabela de Dados Mensais Consolidados")
                
                # Formatar para exibição
                df_exibicao = df_mensal.copy()
                colunas_exibicao = ['Mes', 'Temp_Inst', 'Tmax', 'Tmin', 'Precipitacao', 'UR_Inst', 'U2']
                colunas_existentes = [c for c in colunas_exibicao if c in df_exibicao.columns]
                df_exibicao = df_exibicao[colunas_existentes]
                
                # Renomear para exibição
                renomeios_exibicao = {
                    'Mes': 'Mês',
                    'Temp_Inst': 'Temp Média',
                    'Tmax': 'T Max',
                    'Tmin': 'T Min',
                    'Precipitacao': 'Precipitação',
                    'UR_Inst': 'UR Média',
                    'U2': 'Vento'
                }
                df_exibicao = df_exibicao.rename(columns={k: v for k, v in renomeios_exibicao.items() if k in df_exibicao.columns})
                
                st.dataframe(df_exibicao, use_container_width=True)
                
                # Estatísticas descritivas
                st.markdown("### 📈 Estatísticas Descritivas do Período")
                col_est1, col_est2, col_est3, col_est4 = st.columns(4)
                
                with col_est1:
                    if 'Temp_Inst' in df_mensal.columns:
                        st.metric("🌡️ Temperatura Média", f"{df_mensal['Temp_Inst'].mean():.1f}°C")
                        st.caption(f"Min: {df_mensal['Temp_Inst'].min():.1f}°C | Max: {df_mensal['Temp_Inst'].max():.1f}°C")
                    else:
                        st.info("Dados de temperatura não disponíveis")
                
                with col_est2:
                    if 'Precipitacao' in df_mensal.columns:
                        st.metric("☔ Precipitação Total", f"{df_mensal['Precipitacao'].sum():.0f} mm")
                        st.caption(f"Média mensal: {df_mensal['Precipitacao'].mean():.1f} mm")
                    else:
                        st.info("Dados de precipitação não disponíveis")
                
                with col_est3:
                    if 'UR_Inst' in df_mensal.columns:
                        st.metric("💧 Umidade Média", f"{df_mensal['UR_Inst'].mean():.1f}%")
                        st.caption(f"Min: {df_mensal['UR_Inst'].min():.1f}% | Max: {df_mensal['UR_Inst'].max():.1f}%")
                    else:
                        st.info("Dados de umidade não disponíveis")
                
                with col_est4:
                    if 'U2' in df_mensal.columns:
                        st.metric("💨 Velocidade do Vento", f"{df_mensal['U2'].mean():.1f} m/s")
                        st.caption(f"Máxima: {df_mensal['U2'].max():.1f} m/s")
                    else:
                        st.info("Dados de vento não disponíveis")
            
            # Tab 2: Indicadores Agrícolas
            with tab2:
                if incluir_indicadores and df_indicadores is not None and len(df_indicadores) > 0:
                    st.markdown("### 🌱 Indicadores Agrícolas e Climáticos")
                    
                    # Selecionar colunas para exibição
                    colunas_ind = ['Mes', 'GDD_Acumulado', 'Horas_Frio_10C', 'ETo_Total_mm', 'Indice_Aridade', 'Classificacao_Aridade']
                    colunas_existentes_ind = [c for c in colunas_ind if c in df_indicadores.columns]
                    
                    if colunas_existentes_ind:
                        df_ind_exibicao = df_indicadores[colunas_existentes_ind]
                        
                        # Renomear
                        renomeios_ind = {
                            'Mes': 'Mês',
                            'GDD_Acumulado': 'GDD Acumulado',
                            'Horas_Frio_10C': 'Horas de Frio (10°C)',
                            'ETo_Total_mm': 'ETo Total (mm)',
                            'Indice_Aridade': 'Índice de Aridez',
                            'Classificacao_Aridade': 'Classificação'
                        }
                        df_ind_exibicao = df_ind_exibicao.rename(columns={k: v for k, v in renomeios_ind.items() if k in df_ind_exibicao.columns})
                        
                        st.dataframe(df_ind_exibicao, use_container_width=True)
                    else:
                        st.warning("Indicadores não disponíveis para os dados atuais")
                    
                    # Cards explicativos
                    st.markdown("---")
                    st.markdown("### 📖 Interpretação dos Indicadores")
                    
                    col_int1, col_int2, col_int3 = st.columns(3)
                    
                    with col_int1:
                        st.info("""
                        **🌾 GDD (Graus-Dia)**  
                        Soma térmica diária (Tb=10°C).  
                        Essencial para prever:  
                        • Ciclo das culturas  
                        • Época de floração  
                        • Estimativa de colheita
                        """)
                    
                    with col_int2:
                        st.info("""
                        **❄️ Horas de Frio**  
                        Horas com Tmin ≤ 10°C.  
                        Fundamental para:  
                        • Frutíferas temperadas  
                        • Dormência de gemas  
                        • Produção de maçã, uva
                        """)
                    
                    with col_int3:
                        st.info("""
                        **💧 ETo (Evapotranspiração)**  
                        Demanda evaporativa atmosférica.  
                        Aplica-se em:  
                        • Manejo de irrigação  
                        • Balanço hídrico  
                        • Dimensionamento de projetos
                        """)
                else:
                    st.info("ℹ️ Indicadores agrícolas não disponíveis ou desativados nas configurações")
            
            # Tab 3: Visualizações
            with tab3:
                st.markdown("### 📈 Visualizações Interativas")
                
                # Climograma
                if len(df_mensal) > 0:
                    gerar_climograma_streamlit(df_mensal)
                else:
                    st.warning("Dados insuficientes para gerar visualizações")
            
            # Tab 4: Eventos Extremos
            with tab4:
                if incluir_eventos and eventos:
                    st.markdown("### ⚠️ Eventos Climáticos Extremos Detectados")
                    
                    for evento in eventos:
                        if evento['gravidade'] == 'Crítica':
                            st.error(f"🔴 **{evento['tipo']}** - Gravidade: {evento['gravidade']}")
                        elif evento['gravidade'] == 'Alta':
                            st.warning(f"🟠 **{evento['tipo']}** - Gravidade: {evento['gravidade']}")
                        else:
                            st.info(f"🟡 **{evento['tipo']}** - Gravidade: {evento['gravidade']}")
                        
                        st.write(evento['descricao'])
                        if evento.get('datas'):
                            with st.expander(f"📅 Datas dos eventos ({len(evento['datas'])} ocorrências)"):
                                st.write(', '.join(evento['datas'][:10]))
                                if len(evento['datas']) > 10:
                                    st.write(f"... e mais {len(evento['datas']) - 10} ocorrências")
                        st.markdown("---")
                else:
                    st.success("✅ Nenhum evento climático extremo detectado no período analisado")
            
            # Tab 5: Análise com IA Gemini
            with tab5:
                st.markdown("### 🤖 Análise Inteligente com IA Gemini")
                
                if usar_ia:
                    st.markdown("A IA analisará os dados climáticos e fornecerá insights profissionais")
                    
                    col_ia1, col_ia2 = st.columns([2, 1])
                    
                    with col_ia1:
                        tipo_analise = st.selectbox(
                            "**Selecione o tipo de análise:**",
                            ["Análise Climática Geral", "Identificação de Fenômenos (El Niño/La Niña)", 
                             "Impactos na Agricultura", "Previsão de Tendências", "Análise Personalizada"]
                        )
                    
                    with col_ia2:
                        pergunta_personalizada = ""
                        if tipo_analise == "Análise Personalizada":
                            pergunta_personalizada = st.text_input("Sua pergunta específica:")
                    
                    if st.button("🚀 **Executar Análise com IA**", use_container_width=True):
                        with st.spinner("🤖 Consultando IA Gemini. Analisando dados climáticos..."):
                            # Preparar contexto
                            contexto = f"""
                            Dados da estação {nome_estacao}:
                            - Período analisado: {df_mensal['Mes'].iloc[0]} a {df_mensal['Mes'].iloc[-1]}
                            - Temperatura média: {df_mensal['Temp_Inst'].mean():.1f}°C
                            - Precipitação total: {df_mensal['Precipitacao'].sum():.1f} mm
                            - Temperatura máxima registrada: {df_mensal['Tmax'].max():.1f}°C
                            - Temperatura mínima registrada: {df_mensal['Tmin'].min():.1f}°C
                            - Umidade média: {df_mensal['UR_Inst'].mean():.1f}%
                            - Velocidade média do vento: {df_mensal['U2'].mean():.1f} m/s
                            """
                            
                            if df_indicadores is not None:
                                contexto += f"""
                                - GDD acumulado total: {df_indicadores['GDD_Acumulado'].sum():.0f}
                                - ETo total: {df_indicadores['ETo_Total_mm'].sum():.1f} mm
                                - Horas de frio total: {df_indicadores['Horas_Frio_10C'].sum():.0f} horas
                                """
                            
                            if eventos:
                                contexto += f"\n- Eventos extremos detectados: {len(eventos)}"
                            
                            # Montar prompt
                            if tipo_analise == "Análise Personalizada" and pergunta_personalizada:
                                prompt = pergunta_personalizada
                            elif tipo_analise == "Identificação de Fenômenos (El Niño/La Niña)":
                                prompt = f"""Analise os dados climáticos e identifique possíveis evidências de fenômenos como 
                                El Niño, La Niña ou outras oscilações climáticas. Compare com padrões históricos esperados 
                                para a região de latitude {latitude:.1f}°."""
                            elif tipo_analise == "Impactos na Agricultura":
                                prompt = f"""Com base nos dados climáticos, analise os potenciais impactos na agricultura. 
                                Inclua recomendações sobre épocas de plantio, irrigação, escolha de culturas e manejo 
                                fitossanitário. Considere GDD, horas de frio e eventos extremos detectados."""
                            elif tipo_analise == "Previsão de Tendências":
                                prompt = f"""Projete tendências climáticas para os próximos meses baseado nos padrões 
                                identificados na série histórica. Destaque possíveis anomalias e recomendações de adaptação."""
                            else:
                                prompt = f"""Faça uma análise climática completa dos dados fornecidos. Identifique padrões, 
                                anomalias e forneça um diagnóstico profissional sobre o comportamento climático no período."""
                            
                            resposta_ia = consultar_gemini(prompt, contexto)
                            
                            st.markdown("---")
                            st.markdown("### 🧠 Resposta da IA Gemini")
                            st.markdown(resposta_ia)
                else:
                    st.info("ℹ️ Ative a análise com IA Gemini na barra lateral para obter insights inteligentes")
            
            # Tab 6: Downloads
            with tab6:
                st.markdown("### 💾 Download de Relatórios e Dados")
                
                col_down1, col_down2 = st.columns(2)
                
                with col_down1:
                    st.markdown("#### 📊 Dados em CSV")
                    
                    # Dados mensais
                    if df_mensal is not None and not df_mensal.empty:
                        try:
                            csv_mensal = df_mensal.to_csv(index=False, sep=';', decimal=',')
                            st.download_button(
                                label="📥 Dados Mensais Consolidados (CSV)",
                                data=csv_mensal,
                                file_name=f"{nome_estacao.replace(' ', '_')}_dados_mensais.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Erro ao gerar CSV mensal: {str(e)}")
                    else:
                        st.warning("⚠️ Dados mensais não disponíveis")
                    
                    # Indicadores agrícolas
                    if df_indicadores is not None and len(df_indicadores) > 0:
                        try:
                            csv_indicadores = df_indicadores.to_csv(index=False, sep=';', decimal=',')
                            st.download_button(
                                label="🌱 Indicadores Agrícolas (CSV)",
                                data=csv_indicadores,
                                file_name=f"{nome_estacao.replace(' ', '_')}_indicadores_agricolas.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Erro ao gerar CSV de indicadores: {str(e)}")
                    else:
                        st.info("📌 Indicadores agrícolas não disponíveis")
                    
                    # Dados diários (amostra)
                    if not df_diario.empty:
                        try:
                            csv_diario = df_diario.head(1000).to_csv(index=False, sep=';', decimal=',')
                            st.download_button(
                                label="📋 Amostra de Dados Diários (CSV)",
                                data=csv_diario,
                                file_name=f"{nome_estacao.replace(' ', '_')}_dados_diarios_amostra.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Erro ao gerar CSV diário: {str(e)}")
                
                with col_down2:
                    st.markdown("#### 📄 Relatório Completo")
                    
                    # Relatório PDF (HTML)
                    if df_mensal is not None and not df_mensal.empty:
                        try:
                            relatorio_html = gerar_relatorio_pdf_completo(
                                df_mensal, df_indicadores, qualidade, eventos, 
                                nome_estacao, info_estacao
                            )
                            
                            b64 = base64.b64encode(relatorio_html.encode()).decode()
                            href = f'<a href="data:text/html;base64,{b64}" download="{nome_estacao.replace(" ", "_")}_relatorio_completo.html" style="text-decoration: none;">'
                            href += '<button style="background: linear-gradient(135deg, #2e7d32, #1b5e20); border: none; border-radius: 30px; padding: 10px 25px; color: white; font-weight: 600; cursor: pointer; width: 100%; margin-bottom: 10px;">📑 Baixar Relatório HTML</button></a>'
                            st.markdown(href, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Erro ao gerar relatório: {str(e)}")
                    else:
                        st.warning("⚠️ Dados insuficientes para gerar relatório")
                    
                    # Excel com todas as abas
                    if df_mensal is not None and not df_mensal.empty:
                        try:
                            output_excel = BytesIO()
                            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                                df_mensal.to_excel(writer, sheet_name='Dados_Mensais', index=False)
                                if df_indicadores is not None and len(df_indicadores) > 0:
                                    df_indicadores.to_excel(writer, sheet_name='Indicadores_Agricolas', index=False)
                                if not df_diario.empty:
                                    df_diario.head(5000).to_excel(writer, sheet_name='Dados_Diarios', index=False)
                                if qualidade:
                                    df_qualidade = pd.DataFrame([{k: v for k, v in info.items() if isinstance(v, (int, float, str))} 
                                                                for var, info in qualidade.items() if isinstance(info, dict)])
                                    if not df_qualidade.empty:
                                        df_qualidade.to_excel(writer, sheet_name='Qualidade_Dados', index=False)
                            
                            st.download_button(
                                label="📊 Excel Completo (Todas as Abas)",
                                data=output_excel.getvalue(),
                                file_name=f"{nome_estacao.replace(' ', '_')}_dados_completos.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        except Exception as e:
                            st.error(f"Erro ao gerar Excel: {str(e)}")
                    else:
                        st.warning("⚠️ Dados insuficientes para gerar Excel")
                
                st.markdown("---")
                st.markdown("### 📋 Resumo do Processamento")
                
                # Criar resumo para exibição
                if df_mensal is not None and not df_mensal.empty:
                    resumo = {
                        "Estação": nome_estacao,
                        "Período": f"{df_mensal['Mes'].iloc[0]} a {df_mensal['Mes'].iloc[-1]}" if 'Mes' in df_mensal.columns else "N/A",
                        "Meses analisados": len(df_mensal),
                        "Qualidade média dos dados": f"{np.mean([v.get('percentual_completo', 0) for v in qualidade.values() if isinstance(v, dict) and 'percentual_completo' in v]):.1f}%" if qualidade else "N/A",
                        "Eventos extremos": len(eventos),
                        "Indicadores calculados": len([c for c in df_indicadores.columns if c not in ['Mes', 'Ano', 'Numero_Mes']]) if df_indicadores is not None else 0
                    }
                    st.json(resumo)
                else:
                    st.warning("⚠️ Não há dados suficientes para exibir o resumo")
    
    else:
        # Tela inicial sem arquivo
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h2 style="color: #81c784;">🌾 Bem-vindo ao INMET Smart Processor</h2>
            <p style="color: #a0c4d6; font-size: 1.1rem;">
                Sistema completo para tratamento, análise e visualização de dados meteorológicos do INMET
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px;">
                <h3 style="color: #81c784;">📊 1. Upload</h3>
                <p>Faça upload do arquivo CSV/TXT baixado do INMET</p>
                <p style="font-size: 0.9rem;">✅ Auto-detecção de formato</p>
                <p style="font-size: 0.9rem;">✅ Correção automática de separadores</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_info2:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px;">
                <h3 style="color: #81c784;">⚙️ 2. Processamento</h3>
                <p>Tratamento inteligente dos dados</p>
                <p style="font-size: 0.9rem;">✅ Preenchimento de falhas</p>
                <p style="font-size: 0.9rem;">✅ Detecção de outliers</p>
                <p style="font-size: 0.9rem;">✅ Consolidação mensal</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_info3:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px;">
                <h3 style="color: #81c784;">📈 3. Análise</h3>
                <p>Indicadores e relatórios avançados</p>
                <p style="font-size: 0.9rem;">✅ GDD, Horas de Frio, ETo</p>
                <p style="font-size: 0.9rem;">✅ Eventos climáticos extremos</p>
                <p style="font-size: 0.9rem;">✅ Análise com IA Gemini</p>
                <p style="font-size: 0.9rem;">✅ Relatório profissional</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <p style="color: #a0c4d6;">
                🔬 Desenvolvido para profissionais da agricultura, meteorologia, pesquisa acadêmica e gestão de recursos hídricos
            </p>
            <p style="color: #81c784; font-size: 0.9rem;">
                Faça o upload de um arquivo INMET na barra lateral para começar!
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PONTO DE ENTRADA PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Erro na execução do aplicativo: {str(e)}")
        st.info("Verifique se todos os dados necessários foram preenchidos corretamente.")
