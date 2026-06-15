import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
import requests
import json
import re
from datetime import timedelta
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

GEMINI_API_KEY = ""  # Chave fornecida

def consultar_gemini(prompt, contexto=""):
    """Consulta a API Gemini para análises climáticas"""
    try:
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
        # Analisar padrão
        import re
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
        'temp_maxima': 'Tmax', 'TEMP_MAX': 'Tmax',
        'temp_min': 'Tmin', 'temperatura_minima': 'Tmin', 'TEMPERATURA MINIMA': 'Tmin',
        'temp_minima': 'Tmin', 'TEMP_MIN': 'Tmin',
        
        # Umidade
        'umid_ins': 'UR_Inst', 'umidade_ins': 'UR_Inst', 'UMIDADE INST': 'UR_Inst',
        'umid_max': 'URmax', 'umidade_maxima': 'URmax', 'UMIDADE MAXIMA': 'URmax',
        'umid_min': 'URmin', 'umidade_minima': 'URmin', 'UMIDADE MINIMA': 'URmin',
        
        # Pressão
        'pressao_ins': 'Press_Inst', 'pressao_atm': 'Press_Inst',
        'pressao_max': 'Pressmax', 'pressao_min': 'Pressmin',
        
        # Vento
        'vel_vento': 'U2', 'velocidade_vento': 'U2', 'VEL_VENTO': 'U2',
        'vento_medio': 'U2', 'vento_rajada': 'Raj_vento',
        'dir_vento': 'Dir_vento', 'direcao_vento': 'Dir_vento',
        
        # Radiação e precipitação
        'radiacao': 'Rad_KJ', 'radiacao_solar': 'Rad_KJ',
        'precip': 'Precipitacao', 'chuva': 'Precipitacao', 'PRECIPITACAO': 'Precipitacao',
        'precipitacao_total': 'Precipitacao', 'precipitacao_mensal': 'Precipitacao',
        
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
    
    # Análise de outliers
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
            'registros_esperados': None,
            'intervalo_medio': intervalos.mean(),
            'intervalo_mediano': intervalos.median()
        }
    
    return qualidade, outliers_info

def preenchimento_inteligente_falhas(df, metodo='multivariado'):
    """Preenchimento avançado de falhas usando múltiplas técnicas"""
    
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
            
    elif metodo == 'knn':
        # Preparar dados para KNN
        dados_knn = df_filled[colunas_existentes].copy()
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        dados_imputados = imputer.fit_transform(dados_knn)
        
        for i, col in enumerate(colunas_existentes):
            df_filled[col] = dados_imputados[:, i]
            
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
    """Cálculo avançado de indicadores agrícolas"""
    
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
            if indice < 0.2:
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

def gerar_climograma_profissional(df_mensal):
    """Geração de climograma profissional com matplotlib"""
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [0.6, 0.4]})
    
    meses = df_mensal['Mes'].astype(str).tolist()
    x = range(len(meses))
    
    # ===== GRÁFICO 1: Precipitação e Temperatura =====
    
    # Barras de precipitação
    if 'Precipitacao' in df_mensal.columns:
        bars = ax1.bar(x, df_mensal['Precipitacao'], color='#3498db', alpha=0.7, width=0.8, label='Precipitação (mm)')
        ax1.set_ylabel('Precipitação (mm)', color='#3498db', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='#3498db')
        
        # Adicionar valores nas barras
        for i, (bar, valor) in enumerate(zip(bars, df_mensal['Precipitacao'])):
            if valor > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{valor:.0f}', ha='center', va='bottom', fontsize=8, color='#3498db')
    
    # Linha de temperatura
    ax1_twin = ax1.twinx()
    if 'Temp_Inst' in df_mensal.columns:
        line1 = ax1_twin.plot(x, df_mensal['Temp_Inst'], color='#e74c3c', marker='o', 
                              linewidth=2.5, markersize=8, label='Temperatura Média (°C)')
        ax1_twin.set_ylabel('Temperatura (°C)', color='#e74c3c', fontsize=12)
        ax1_twin.tick_params(axis='y', labelcolor='#e74c3c')
        
        # Adicionar valores nos pontos
        for i, temp in enumerate(df_mensal['Temp_Inst']):
            ax1_twin.annotate(f'{temp:.1f}°C', (i, temp), 
                            textcoords="offset points", xytext=(0, 10), 
                            ha='center', fontsize=8, color='#e74c3c')
    
    # Temperaturas máximas e mínimas
    if 'Tmax' in df_mensal.columns and 'Tmin' in df_mensal.columns:
        ax1_twin.plot(x, df_mensal['Tmax'], color='#ff9800', marker='^', 
                     linewidth=1.5, markersize=6, linestyle='--', label='T Máxima')
        ax1_twin.plot(x, df_mensal['Tmin'], color='#2196f3', marker='v', 
                     linewidth=1.5, markersize=6, linestyle='--', label='T Mínima')
    
    # Configurar eixo X do gráfico 1
    ax1.set_xticks(x)
    ax1.set_xticklabels(meses, rotation=45, ha='right', fontsize=10)
    ax1.set_xlabel('Mês', fontsize=12)
    ax1.set_title('🌡️ Precipitação e Temperatura', fontsize=14, fontweight='bold', pad=15)
    
    # ===== GRÁFICO 2: Indicadores Agrícolas =====
    
    # GDD Acumulado
    if 'GDD_Acumulado' in df_mensal.columns:
        bars2 = ax2.bar(x, df_mensal['GDD_Acumulado'], color='#4caf50', alpha=0.7, width=0.4, 
                       label='GDD Acumulado', position=0)
        
        # Adicionar valores
        for i, (bar, valor) in enumerate(zip(bars2, df_mensal['GDD_Acumulado'])):
            if valor > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        f'{valor:.0f}', ha='center', va='bottom', fontsize=8, color='#4caf50')
    
    # ETo Total
    ax2_twin = ax2.twinx()
    if 'ETo_Total_mm' in df_mensal.columns:
        ax2_twin.plot(x, df_mensal['ETo_Total_mm'], color='#9c27b0', marker='s', 
                     linewidth=2, markersize=7, label='ETo Total (mm)')
        ax2_twin.set_ylabel('ETo (mm)', color='#9c27b0', fontsize=12)
        ax2_twin.tick_params(axis='y', labelcolor='#9c27b0')
        
        # Adicionar valores
        for i, eto in enumerate(df_mensal['ETo_Total_mm']):
            ax2_twin.annotate(f'{eto:.0f}', (i, eto), 
                            textcoords="offset points", xytext=(0, 10), 
                            ha='center', fontsize=8, color='#9c27b0')
    
    # Horas de Frio (se disponível)
    if 'Horas_Frio_10C' in df_mensal.columns:
        ax2_bar = ax2.bar([i + 0.35 for i in x], df_mensal['Horas_Frio_10C'], 
                         color='#00bcd4', alpha=0.7, width=0.35, label='Horas de Frio (10°C)')
    
    # Configurar eixo X do gráfico 2
    ax2.set_xticks(x)
    ax2.set_xticklabels(meses, rotation=45, ha='right', fontsize=10)
    ax2.set_xlabel('Mês', fontsize=12)
    ax2.set_ylabel('GDD / Horas de Frio', fontsize=12)
    ax2.set_title('🌱 Indicadores Agrícolas', fontsize=14, fontweight='bold', pad=15)
    
    # ===== LEGENDAS =====
    # Coletar todas as legendas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1_twin, labels1_twin = ax1_twin.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2_twin, labels2_twin = ax2_twin.get_legend_handles_labels()
    
    ax1.legend(lines1 + lines1_twin, labels1 + labels1_twin, loc='upper left', fontsize=9)
    ax2.legend(lines2 + lines2_twin, labels2 + labels2_twin, loc='upper left', fontsize=9)
    
    # ===== AJUSTES FINAIS =====
    plt.suptitle(f'📊 Análise Climática Completa - {df_mensal["Mes"].iloc[0][:4]} a {df_mensal["Mes"].iloc[-1][:4]}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig
    
    # Gráfico 1: Precipitação (barras) e Temperatura (linha)
    fig.add_trace(
        go.Bar(
            x=meses, 
            y=df_mensal['Precipitacao'], 
            name='Precipitação (mm)',
            marker_color='#3498db',
            marker_line_color='#2980b9',
            marker_line_width=1,
            opacity=0.8
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=meses, 
            y=df_mensal['Temp_Inst'], 
            name='Temperatura Média (°C)',
            line=dict(color='#e74c3c', width=3),
            mode='lines+markers',
            marker=dict(size=8, symbol='circle')
        ),
        row=1, col=1
    )
    
    # Adicionar temperaturas máximas e mínimas
    if 'Tmax' in df_mensal.columns and 'Tmin' in df_mensal.columns:
        fig.add_trace(
            go.Scatter(
                x=meses, 
                y=df_mensal['Tmax'], 
                name='Temperatura Máxima',
                line=dict(color='#ff9800', width=2, dash='dash'),
                mode='lines+markers',
                marker=dict(size=6, symbol='triangle-up')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=meses, 
                y=df_mensal['Tmin'], 
                name='Temperatura Mínima',
                line=dict(color='#2196f3', width=2, dash='dash'),
                mode='lines+markers',
                marker=dict(size=6, symbol='triangle-down')
            ),
            row=1, col=1
        )
    
    # Gráfico 2: Indicadores (se disponíveis)
    y_atual = 1
    if 'GDD_Acumulado' in df_mensal.columns:
        fig.add_trace(
            go.Bar(
                x=meses, 
                y=df_mensal['GDD_Acumulado'], 
                name='GDD Acumulado',
                marker_color='#4caf50',
                opacity=0.7
            ),
            row=2, col=1
        )
        y_atual += 1
    
    if 'ETo_Total_mm' in df_mensal.columns:
        fig.add_trace(
            go.Scatter(
                x=meses, 
                y=df_mensal['ETo_Total_mm'], 
                name='ETo Total (mm)',
                line=dict(color='#9c27b0', width=2),
                mode='lines+markers'
            ),
            row=2, col=1
        )
    
    # Atualizar layout
    fig.update_layout(
        title=dict(
            text='<b>Análise Climática Completa</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        template='plotly_dark',
        height=700,
        hovermode='x unified'
    )
    
    # Configurar eixos do primeiro gráfico
    fig.update_yaxes(title_text="Precipitação (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Temperatura (°C)", row=1, col=1, overlaying='y', side='right')
    
    # Configurar eixos do segundo gráfico
    fig.update_yaxes(title_text="Valores", row=2, col=1)
    
    return fig

def gerar_relatorio_pdf_completo(df_mensal, df_indicadores, qualidade, eventos_extremos, nome_estacao, info_estacao):
    """Geração de relatório PDF completo e profissional"""
    
    data_geracao = datetime.now().strftime('%d/%m/%Y às %H:%M')
    
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
            .section h3 {{
                color: #2c3e50;
                margin: 20px 0 10px;
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
    """
    
    # Resumo estatístico
    html_content += """
            <div class="section">
                <h2>📊 Resumo Estatístico do Período</h2>
                <div class="metric-grid">
    """
    
    if 'Temp_Inst' in df_mensal.columns:
        html_content += f"""
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
        """
    
    if 'Precipitacao' in df_mensal.columns:
        html_content += f"""
                    <div class="metric-card">
                        <div class="metric-value">{df_mensal['Precipitacao'].sum():.0f} mm</div>
                        <div class="metric-label">Precipitação Total</div>
                    </div>
        """
    
    if 'GDD_Acumulado' in df_indicadores.columns:
        html_content += f"""
                    <div class="metric-card">
                        <div class="metric-value">{df_indicadores['GDD_Acumulado'].sum():.0f}</div>
                        <div class="metric-label">GDD Total (Tb=10°C)</div>
                    </div>
        """
    
    if 'ETo_Total_mm' in df_indicadores.columns:
        html_content += f"""
                    <div class="metric-card">
                        <div class="metric-value">{df_indicadores['ETo_Total_mm'].sum():.0f} mm</div>
                        <div class="metric-label">Evapotranspiração Total</div>
                    </div>
        """
    
    html_content += """
                </div>
            </div>
    """
    
    # Dados mensais
    html_content += """
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
    """
    
    for _, row in df_mensal.iterrows():
        html_content += f"""
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
    
    html_content += """
                    </tbody>
                </table>
            </div>
    """
    
    # Indicadores agrícolas
    if df_indicadores is not None and len(df_indicadores) > 0:
        html_content += """
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
        """
        
        for _, row in df_indicadores.iterrows():
            html_content += f"""
                        <tr>
                            <td>{row.get('Mes', 'N/A')}</td>
                            <td>{row.get('GDD_Acumulado', 0):.0f}</td>
                            <td>{row.get('Horas_Frio_10C', 0):.0f}</td>
                            <td>{row.get('ETo_Total_mm', 0):.1f}</td>
                            <td>{row.get('Indice_Aridade', 0):.2f}</td>
                            <td>{row.get('Classificacao_Aridade', 'N/A')}</td>
                        </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
            </div>
        """
    
    # Eventos extremos
    if eventos_extremos:
        html_content += """
            <div class="section">
                <h2>⚠️ Eventos Climáticos Extremos Detectados</h2>
        """
        
        for evento in eventos_extremos:
            classe = 'critical-card' if evento['gravidade'] == 'Crítica' else 'warning-card' if evento['gravidade'] == 'Alta' else 'success-card'
            html_content += f"""
                <div class="{classe}">
                    <strong>🔴 {evento['tipo']}</strong><br>
                    {evento['descricao']}<br>
                    <small>Gravidade: {evento['gravidade']}</small>
                </div>
            """
        
        html_content += """
            </div>
        """
    
    # Qualidade dos dados
    if qualidade:
        html_content += """
            <div class="section">
                <h2>📈 Análise de Qualidade dos Dados</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Variável</th>
                            <th>Completude (%)</th>
                            <th>Classificação</th>
                            <th>Outliers (%)</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for var, info in qualidade.items():
            if var != 'temporal' and isinstance(info, dict):
                outliers_pct = info.get('outliers_percent', 0)
                html_content += f"""
                        <tr>
                            <td>{var}</td>
                            <td>{info.get('percentual_completo', 0):.1f}%</td>
                            <td>{info.get('classificacao', 'N/A')}</td>
                            <td>{outliers_pct:.1f}%</td>
                        </tr>
                """
        
        html_content += """
                    </tbody>
                </table>
            </div>
        """
    
    html_content += """
            <div class="footer">
                <p>Relatório gerado automaticamente pelo INMET Smart Processor</p>
                <p>Este relatório pode ser utilizado para fins acadêmicos, científicos e de planejamento agrícola</p>
                <p>© 2024 - Todos os direitos reservados</p>
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
        
        # Passo 4: Análise de qualidade
        with st.spinner("📊 Analisando qualidade dos dados..."):
            qualidade, outliers = analise_completa_qualidade(df)
            resultados['qualidade'] = qualidade
            resultados['outliers'] = outliers
        
        # Passo 5: Preenchimento de falhas
        with st.spinner("🔄 Preenchendo falhas nos dados..."):
            metodo_falhas = config.get('metodo_falhas', 'multivariado')
            df = preenchimento_inteligente_falhas(df, metodo=metodo_falhas)
        
        # Passo 6: Detecção de eventos extremos
        with st.spinner("⚠️ Detectando eventos climáticos extremos..."):
            eventos_extremos = detectar_eventos_extremos(df)
            resultados['eventos_extremos'] = eventos_extremos
        
        # Passo 7: Consolidação mensal
        with st.spinner("📆 Consolidando dados por mês..."):
            df_mensal = consolidacao_avancada_por_mes(df)
            resultados['dados_mensais'] = df_mensal
        
        # Passo 8: Cálculo de indicadores agrícolas
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
            ["multivariado", "interpolacao_linear", "interpolacao_spline", "knn", "media_movel"],
            format_func=lambda x: {
                "multivariado": "🧠 Multivariado (Recomendado)",
                "interpolacao_linear": "📈 Interpolação Linear",
                "interpolacao_spline": "📊 Spline Cúbico",
                "knn": "🤝 KNN (Vizinhos Próximos)",
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
                df_exibicao.columns = ['Mês', 'Temp Média', 'T Max', 'T Min', 'Precipitação', 'UR Média', 'Vento']
                
                st.dataframe(df_exibicao, use_container_width=True)
                
                # Estatísticas descritivas
                st.markdown("### 📈 Estatísticas Descritivas do Período")
                col_est1, col_est2, col_est3, col_est4 = st.columns(4)
                
                with col_est1:
                    if 'Temp_Inst' in df_mensal.columns:
                        st.metric("🌡️ Temperatura Média", f"{df_mensal['Temp_Inst'].mean():.1f}°C")
                        st.caption(f"Min: {df_mensal['Temp_Inst'].min():.1f}°C | Max: {df_mensal['Temp_Inst'].max():.1f}°C")
                
                with col_est2:
                    if 'Precipitacao' in df_mensal.columns:
                        st.metric("☔ Precipitação Total", f"{df_mensal['Precipitacao'].sum():.0f} mm")
                        st.caption(f"Média mensal: {df_mensal['Precipitacao'].mean():.1f} mm")
                
                with col_est3:
                    if 'UR_Inst' in df_mensal.columns:
                        st.metric("💧 Umidade Média", f"{df_mensal['UR_Inst'].mean():.1f}%")
                        st.caption(f"Min: {df_mensal['UR_inst'].min() if 'UR_inst' in df_mensal else 'N/A'}")
                
                with col_est4:
                    if 'U2' in df_mensal.columns:
                        st.metric("💨 Velocidade do Vento", f"{df_mensal['U2'].mean():.1f} m/s")
                        st.caption(f"Máxima: {df_mensal['U2'].max():.1f} m/s")
            
            # Tab 2: Indicadores Agrícolas
            with tab2:
                if incluir_indicadores and df_indicadores is not None:
                    st.markdown("### 🌱 Indicadores Agrícolas e Climáticos")
                    
                    # Selecionar colunas para exibição
                    colunas_ind = ['Mes', 'GDD_Acumulado', 'Horas_Frio_10C', 'ETo_Total_mm', 'Indice_Aridade', 'Classificacao_Aridade']
                    colunas_existentes = [c for c in colunas_ind if c in df_indicadores.columns]
                    df_ind_exibicao = df_indicadores[colunas_existentes]
                    
                    # Renomear
                    renomeios = {
                        'Mes': 'Mês',
                        'GDD_Acumulado': 'GDD Acumulado',
                        'Horas_Frio_10C': 'Horas de Frio (10°C)',
                        'ETo_Total_mm': 'ETo Total (mm)',
                        'Indice_Aridade': 'Índice de Aridez',
                        'Classificacao_Aridade': 'Classificação'
                    }
                    df_ind_exibicao = df_ind_exibicao.rename(columns=renomeios)
                    
                    st.dataframe(df_ind_exibicao, use_container_width=True)
                    
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
                st.plotly_chart(gerar_climograma_profissional(df_mensal), use_container_width=True)
                
                # Séries temporais adicionais
                if df_indicadores is not None and 'GDD_Acumulado' in df_indicadores.columns:
                    st.markdown("### 🌡️ Análise de Tendências")
                    
                    fig_tendencia = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Evolução do GDD Acumulado', 'Horas de Frio por Mês'),
                        vertical_spacing=0.2
                    )
                    
                    # GDD
                    fig_tendencia.add_trace(
                        go.Scatter(
                            x=df_indicadores['Mes'].astype(str),
                            y=df_indicadores['GDD_Acumulado'],
                            mode='lines+markers',
                            name='GDD',
                            line=dict(color='#4caf50', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(76, 175, 80, 0.2)'
                        ),
                        row=1, col=1
                    )
                    
                    # Horas de frio
                    if 'Horas_Frio_10C' in df_indicadores.columns:
                        fig_tendencia.add_trace(
                            go.Bar(
                                x=df_indicadores['Mes'].astype(str),
                                y=df_indicadores['Horas_Frio_10C'],
                                name='Horas de Frio',
                                marker_color='#2196f3'
                            ),
                            row=2, col=1
                        )
                    
                    fig_tendencia.update_layout(
                        height=600,
                        template='plotly_dark',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_tendencia, use_container_width=True)
            
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
                            if tipo_analise == "Análise Personalizada" and 'pergunta_personalizada' in locals() and pergunta_personalizada:
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
                    csv_mensal = df_mensal.to_csv(index=False, sep=';', decimal=',')
                    st.download_button(
                        label="📥 Dados Mensais Consolidados (CSV)",
                        data=csv_mensal,
                        file_name=f"{nome_estacao}_dados_mensais.csv",
                        mime="text/csv"
                    )
                    
                    # Indicadores agrícolas
                    if df_indicadores is not None:
                        csv_indicadores = df_indicadores.to_csv(index=False, sep=';', decimal=',')
                        st.download_button(
                            label="🌱 Indicadores Agrícolas (CSV)",
                            data=csv_indicadores,
                            file_name=f"{nome_estacao}_indicadores_agricolas.csv",
                            mime="text/csv"
                        )
                
                with col_down2:
                    st.markdown("#### 📄 Relatório Completo")
                    
                    # Relatório PDF (HTML)
                    relatorio_html = gerar_relatorio_pdf_completo(
                        df_mensal, df_indicadores, qualidade, eventos, 
                        nome_estacao, info_estacao
                    )
                    
                    b64 = base64.b64encode(relatorio_html.encode()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="{nome_estacao}_relatorio_completo.html" style="text-decoration: none;">'
                    href += '<button style="background: linear-gradient(135deg, #2e7d32, #1b5e20); border: none; border-radius: 30px; padding: 10px 25px; color: white; font-weight: 600; cursor: pointer; width: 100%; margin-bottom: 10px;">📑 Baixar Relatório HTML</button></a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Excel com todas as abas
                    output_excel = BytesIO()
                    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                        df_mensal.to_excel(writer, sheet_name='Dados_Mensais', index=False)
                        if df_indicadores is not None:
                            df_indicadores.to_excel(writer, sheet_name='Indicadores_Agricolas', index=False)
                        if not df_diario.empty:
                            df_diario.head(1000).to_excel(writer, sheet_name='Amostra_Dados_Diarios', index=False)
                    
                    st.download_button(
                        label="📊 Excel Completo (Todas as Abas)",
                        data=output_excel.getvalue(),
                        file_name=f"{nome_estacao}_dados_completos.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                st.markdown("---")
                st.markdown("### 📋 Resumo do Processamento")
                st.json({
                    "Estaçao": nome_estacao,
                    "Período": f"{df_mensal['Mes'].iloc[0]} a {df_mensal['Mes'].iloc[-1]}",
                    "Meses_analisados": len(df_mensal),
                    "Qualidade_media_dados": f"{np.mean([v.get('percentual_completo', 0) for v in qualidade.values() if isinstance(v, dict) and 'percentual_completo' in v]):.1f}%",
                    "Eventos_extremos": len(eventos),
                    "Indicadores_calculados": len([c for c in df_indicadores.columns if c not in ['Mes', 'Ano', 'Numero_Mes']]) if df_indicadores is not None else 0
                })
    
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

if __name__ == "__main__":
    main()
