import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
import base64
import warnings
import requests
import re
import plotly.graph_objs as go
import plotly.express as px
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA E ESTILO
# ============================================================================

st.set_page_config(
    page_title="Smart Meteorological Processor - Sistema Completo de Análise Meteorológica",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Personalizado Avançado
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
    
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
    
    .footer {
        text-align: center;
        padding: 20px;
        font-size: 12px;
        color: #999;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin-top: 30px;
    }
    
    .stDataFrame {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        overflow: hidden;
    }
    
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURAÇÃO DA IA GEMINI
# ============================================================================

GEMINI_API_KEY = "AQ.Ab8RN6J5uzUogvavjQyfFr3wGWEaJbdrW3oByqhWo2bm_mMxmQ"

def consultar_gemini(prompt, contexto=""):
    """Consulta a API Gemini para análises climáticas"""
    try:
        if not GEMINI_API_KEY:
            return "⚠️ API Key do Gemini não configurada."
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
        
        prompt_completo = f"""Você é um meteorologista especialista. Analise os dados climáticos.

        CONTEXTO:
        {contexto}

        PERGUNTA:
        {prompt}

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
                return "Não foi possível obter uma análise detalhada."
        else:
            return f"⚠️ Erro na consulta: Status {response.status_code}"
            
    except Exception as e:
        return f"⚠️ Erro: {str(e)}"

# ============================================================================
# FUNÇÕES DE LEITURA E PROCESSAMENTO
# ============================================================================

def detectar_formato_arquivo(arquivo_bytes):
    """Detecção avançada de formato de arquivo"""
    try:
        conteudo = arquivo_bytes.getvalue().decode('utf-8', errors='ignore')[:5000]
        linhas = conteudo.split('\n')
    except:
        conteudo = ""
        linhas = []
    
    info = {
        'delimiter': None,
        'decimal_separator': ',',
        'skiprows': 0,
        'station_name': None,
        'latitude': None,
        'longitude': None
    }
    
    if linhas:
        delimiters = [';', ',', '\t', '|']
        for delim in delimiters:
            if linhas[0] and delim in linhas[0]:
                info['delimiter'] = delim
                break
    
    if not info['delimiter']:
        info['delimiter'] = ';'
    
    if conteudo:
        if ',' in conteudo and '.' not in conteudo:
            info['decimal_separator'] = ','
        elif '.' in conteudo and ',' not in conteudo:
            info['decimal_separator'] = '.'
        else:
            info['decimal_separator'] = ',' if re.search(r'\d+,\d+', conteudo) else '.'
    
    if linhas:
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
    info = detectar_formato_arquivo(arquivo_bytes)
    
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
            arquivo_bytes.seek(0)
            df_temp = pd.read_csv(
                arquivo_bytes,
                sep=info['delimiter'],
                encoding='utf-8',
                skiprows=estrategia['skiprows'],
                header=estrategia['header'] if isinstance(estrategia['header'], int) else 'infer',
                on_bad_lines='skip'
            )
            
            if len(df_temp.columns) > 3 and len(df_temp) > 10:
                df = df_temp
                info['skiprows'] = estrategia['skiprows']
                break
        except:
            continue
    
    if df is None:
        raise ValueError("Não foi possível ler o arquivo. Verifique o formato.")
    
    if info['decimal_separator'] == ',':
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.replace(',', '.')
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
    
    return df, info

def padronizar_colunas(df):
    """Padronização de nomes de colunas"""
    mapeamento = {
        'data': 'Date', 'DATA': 'Date', 'Data': 'Date', 'dt_data': 'Date',
        'hora': 'Time', 'HORA': 'Time', 'Hora': 'Time', 'hr_data': 'Time',
        'temp_ins': 'Temp_Inst', 'temperatura_ins': 'Temp_Inst', 'temp_inst': 'Temp_Inst',
        'temp_max': 'Tmax', 'temperatura_maxima': 'Tmax', 'temp_maxima': 'Tmax',
        'temp_min': 'Tmin', 'temperatura_minima': 'Tmin', 'temp_minima': 'Tmin',
        'umid_ins': 'UR_Inst', 'umidade_ins': 'UR_Inst', 'umid_inst': 'UR_Inst',
        'umid_max': 'URmax', 'umidade_maxima': 'URmax',
        'umid_min': 'URmin', 'umidade_minima': 'URmin',
        'precip': 'Precipitacao', 'chuva': 'Precipitacao', 'precipitacao': 'Precipitacao',
        'vel_vento': 'U2', 'velocidade_vento': 'U2', 'vento_medio': 'U2',
        'pressao_ins': 'Press_Inst', 'pressao_atm': 'Press_Inst',
        'radiacao': 'Rad_KJ', 'radiacao_solar': 'Rad_KJ'
    }
    
    df_renomeado = df.copy()
    for col in df_renomeado.columns:
        col_lower = col.lower().strip()
        if col_lower in mapeamento:
            df_renomeado.rename(columns={col: mapeamento[col_lower]}, inplace=True)
        else:
            for chave, valor in mapeamento.items():
                if chave in col_lower or col_lower in chave:
                    df_renomeado.rename(columns={col: valor}, inplace=True)
                    break
    
    return df_renomeado

def corrigir_datas(df):
    """Correção robusta de datas"""
    df_corrigido = df.copy()
    
    col_data = None
    for col in df_corrigido.columns:
        if col.lower() in ['date', 'data', 'dt']:
            col_data = col
            break
    
    if not col_data:
        for col in df_corrigido.columns:
            try:
                amostra = df_corrigido[col].dropna().iloc[0] if len(df_corrigido[col].dropna()) > 0 else ''
                if isinstance(amostra, str) and ('/' in amostra or '-' in amostra):
                    if len(amostra) > 6:
                        col_data = col
                        break
            except:
                continue
    
    if col_data:
        formatos = ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y']
        for formato in formatos:
            try:
                df_corrigido['Date'] = pd.to_datetime(df_corrigido[col_data], format=formato, errors='coerce')
                if df_corrigido['Date'].notna().sum() > len(df_corrigido) * 0.8:
                    break
            except:
                continue
        
        if df_corrigido['Date'].isna().sum() > len(df_corrigido) * 0.5:
            df_corrigido['Date'] = pd.to_datetime(df_corrigido[col_data], errors='coerce')
    
    col_hora = None
    for col in df_corrigido.columns:
        if col.lower() in ['time', 'hora', 'hr']:
            col_hora = col
            break
    
    if col_hora and 'Date' in df_corrigido.columns:
        try:
            df_corrigido['DateTime'] = pd.to_datetime(
                df_corrigido['Date'].dt.strftime('%Y-%m-%d') + ' ' + 
                df_corrigido[col_hora].astype(str).str.zfill(4).str[:2] + ':' + 
                df_corrigido[col_hora].astype(str).str.zfill(4).str[2:4],
                errors='coerce'
            )
        except:
            df_corrigido['DateTime'] = df_corrigido['Date']
    elif 'Date' in df_corrigido.columns:
        df_corrigido['DateTime'] = df_corrigido['Date']
    
    if 'DateTime' in df_corrigido.columns:
        df_corrigido['Year'] = df_corrigido['DateTime'].dt.year
        df_corrigido['Month'] = df_corrigido['DateTime'].dt.month
        df_corrigido['Day'] = df_corrigido['DateTime'].dt.day
        df_corrigido['Hour'] = df_corrigido['DateTime'].dt.hour
    
    return df_corrigido

def preenchimento_inteligente_falhas(df, limite_max_falhas=10):
    """
    Preenchimento de falhas - apenas se não houver mais de 10 falhas consecutivas
    """
    df_filled = df.copy()
    substituicoes = []
    
    colunas_numericas = ['Tmax', 'Tmin', 'Temp_Inst', 'UR_Inst', 'URmax', 'URmin', 
                         'Precipitacao', 'U2', 'Press_Inst']
    colunas_existentes = [c for c in colunas_numericas if c in df.columns]
    
    for col in colunas_existentes:
        try:
            # Contar falhas consecutivas
            mask_falhas = df_filled[col].isna()
            max_consecutivas = 0
            consecutivas = 0
            for val in mask_falhas:
                if val:
                    consecutivas += 1
                    max_consecutivas = max(max_consecutivas, consecutivas)
                else:
                    consecutivas = 0
            
            if max_consecutivas > limite_max_falhas:
                substituicoes.append({
                    'variavel': col,
                    'status': 'NÃO PREENCHIDO',
                    'motivo': f'{max_consecutivas} falhas consecutivas (> {limite_max_falhas})'
                })
                continue
            
            num_falhas_original = df_filled[col].isna().sum()
            
            # Interpolação linear
            df_filled[col] = df_filled[col].interpolate(method='linear', limit_direction='both', limit=limite_max_falhas)
            
            # Média móvel para remanescentes
            mascara = df_filled[col].isna()
            if mascara.any():
                media_movel = df_filled[col].rolling(window=24, min_periods=6, center=True).mean()
                df_filled.loc[mascara, col] = media_movel.loc[mascara]
            
            # Preenchimento sazonal
            if 'Month' in df_filled.columns:
                mascara = df_filled[col].isna()
                if mascara.any():
                    medias_mensais = df_filled.groupby('Month')[col].transform('mean')
                    df_filled.loc[mascara, col] = medias_mensais.loc[mascara]
            
            # Último recurso: mediana
            if df_filled[col].isna().any():
                mediana = df_filled[col].median()
                if pd.notna(mediana):
                    df_filled[col].fillna(mediana, inplace=True)
            
            num_falhas_final = df_filled[col].isna().sum()
            num_substituidos = num_falhas_original - num_falhas_final
            
            if num_substituidos > 0:
                substituicoes.append({
                    'variavel': col,
                    'status': 'PREENCHIDO',
                    'quantidade': num_substituidos,
                    'motivo': f'{num_substituidos} valores substituídos'
                })
        except Exception as e:
            substituicoes.append({'variavel': col, 'status': 'ERRO', 'motivo': str(e)})
    
    return df_filled, substituicoes

def detectar_outliers_tratar(df, limite_fator=3):
    """Detecta e trata outliers usando IQR"""
    df_tratado = df.copy()
    outliers_info = {}
    
    colunas_numericas = ['Tmax', 'Tmin', 'Temp_Inst', 'UR_Inst', 'URmax', 'URmin', 
                         'Precipitacao', 'U2', 'Press_Inst']
    colunas_existentes = [c for c in colunas_numericas if c in df.columns]
    
    for col in colunas_existentes:
        try:
            dados_validos = df_tratado[col].dropna()
            if len(dados_validos) > 0:
                Q1 = dados_validos.quantile(0.25)
                Q3 = dados_validos.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    limite_inferior = Q1 - limite_fator * IQR
                    limite_superior = Q3 + limite_fator * IQR
                    
                    mask_outliers = (df_tratado[col] < limite_inferior) | (df_tratado[col] > limite_superior)
                    num_outliers = mask_outliers.sum()
                    
                    if num_outliers > 0:
                        mediana = dados_validos.median()
                        df_tratado.loc[mask_outliers, col] = mediana
                        
                        outliers_info[col] = {
                            'num_outliers': int(num_outliers),
                            'percentual': round((num_outliers / len(dados_validos)) * 100, 2),
                            'limites': f'[{limite_inferior:.2f}, {limite_superior:.2f}]'
                        }
        except Exception as e:
            pass
    
    return df_tratado, outliers_info

def consolidar_medias_mensais(df):
    """
    CRIA UM DATAFRAME SEPARADO com médias mensais.
    NÃO altera os dados originais.
    """
    df_temp = df.copy()
    
    if 'DateTime' not in df_temp.columns:
        if 'Date' in df_temp.columns:
            df_temp['DateTime'] = pd.to_datetime(df_temp['Date'], errors='coerce')
        else:
            return pd.DataFrame()
    
    df_temp['AnoMes'] = df_temp['DateTime'].dt.to_period('M')
    
    agregacoes = {
        'Tmax': 'max',
        'Tmin': 'min',
        'Temp_Inst': 'mean',
        'UR_Inst': 'mean',
        'URmax': 'max',
        'URmin': 'min',
        'Precipitacao': 'sum',
        'U2': 'mean',
        'Press_Inst': 'mean'
    }
    
    agregacoes_filtradas = {k: v for k, v in agregacoes.items() if k in df_temp.columns}
    
    if not agregacoes_filtradas:
        return pd.DataFrame()
    
    df_mensal = df_temp.groupby('AnoMes').agg(agregacoes_filtradas).reset_index()
    df_mensal['Mês'] = df_mensal['AnoMes'].astype(str)
    df_mensal['Ano'] = df_mensal['AnoMes'].dt.year
    df_mensal['Número_Mês'] = df_mensal['AnoMes'].dt.month
    
    df_mensal = df_mensal.sort_values('AnoMes').reset_index(drop=True)
    
    return df_mensal

def analise_completa_qualidade(df):
    """Análise da qualidade dos dados"""
    qualidade = {}
    
    colunas_numericas = ['Tmax', 'Tmin', 'Temp_Inst', 'UR_Inst', 'URmax', 'URmin', 
                         'Precipitacao', 'U2', 'Press_Inst']
    colunas_existentes = [c for c in colunas_numericas if c in df.columns]
    
    for col in colunas_existentes:
        total = len(df)
        presentes = df[col].notna().sum()
        percentual = (presentes / total) * 100 if total > 0 else 0
        
        classificacao = 'Boa' if percentual >= 95 else 'Regular' if percentual >= 80 else 'Crítica'
        
        qualidade[col] = {
            'percentual_completo': round(percentual, 2),
            'classificacao': classificacao,
            'dados_faltantes': total - presentes
        }
    
    if 'DateTime' in df.columns:
        qualidade['periodo'] = {
            'inicio': df['DateTime'].min(),
            'fim': df['DateTime'].max(),
            'total_registros': len(df)
        }
    
    return qualidade

def detectar_eventos_extremos(df):
    """Detecta eventos climáticos extremos"""
    eventos = []
    
    if 'Temp_Inst' in df.columns:
        media_temp = df['Temp_Inst'].mean()
        std_temp = df['Temp_Inst'].std()
        if pd.notna(std_temp) and std_temp > 0:
            limite_calor = media_temp + 2 * std_temp
            mask_calor = df['Temp_Inst'] > limite_calor
            if mask_calor.sum() > 0:
                eventos.append({
                    'tipo': 'Onda de Calor',
                    'gravidade': 'Alta',
                    'descricao': f'{mask_calor.sum()} ocorrências acima de {limite_calor:.1f}°C',
                    'quantidade': int(mask_calor.sum())
                })
    
    if 'Precipitacao' in df.columns:
        media_precip = df['Precipitacao'].mean()
        std_precip = df['Precipitacao'].std()
        if pd.notna(std_precip) and std_precip > 0:
            limite_chuva = media_precip + 2 * std_precip
            mask_chuva = df['Precipitacao'] > limite_chuva
            if mask_chuva.sum() > 0:
                eventos.append({
                    'tipo': 'Chuva Intensa',
                    'gravidade': 'Alta',
                    'descricao': f'{mask_chuva.sum()} ocorrências acima de {limite_chuva:.1f} mm',
                    'quantidade': int(mask_chuva.sum())
                })
    
    if 'U2' in df.columns:
        mask_vento = df['U2'] > 15
        if mask_vento.sum() > 0:
            eventos.append({
                'tipo': 'Ventos Fortes',
                'gravidade': 'Moderada',
                'descricao': f'{mask_vento.sum()} ocorrências acima de 15 m/s',
                'quantidade': int(mask_vento.sum())
            })
    
    return eventos

# ============================================================================
# FUNÇÕES DE GRÁFICOS INTERATIVOS (APENAS PLOTLY)
# ============================================================================

def criar_grafico_temperatura(df, tipo='linha'):
    """Cria gráfico de temperatura interativo"""
    if 'DateTime' not in df.columns or 'Temp_Inst' not in df.columns:
        return None
    
    df_plot = df.dropna(subset=['Temp_Inst']).copy()
    
    if len(df_plot) == 0:
        return None
    
    if tipo == 'linha':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot['DateTime'],
            y=df_plot['Temp_Inst'],
            mode='lines',
            name='Temperatura',
            line=dict(color='#e74c3c', width=2),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.1)
        )
        
        fig.update_layout(
            title='🌡️ Temperatura ao Longo do Tempo',
            xaxis_title='Data/Hora',
            yaxis_title='Temperatura (°C)',
            template='plotly_dark',
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    elif tipo == 'box':
        df_plot['Hora'] = df_plot['DateTime'].dt.hour
        fig = px.box(df_plot, x='Hora', y='Temp_Inst', 
                     title='📊 Distribuição da Temperatura por Hora',
                     labels={'Hora': 'Hora do Dia', 'Temp_Inst': 'Temperatura (°C)'},
                     color_discrete_sequence=['#e74c3c'])
        fig.update_layout(template='plotly_dark', height=400)
        return fig
    
    return None

def criar_grafico_precipitacao(df):
    """Cria gráfico de precipitação"""
    if 'DateTime' not in df.columns or 'Precipitacao' not in df.columns:
        return None
    
    df_plot = df.dropna(subset=['Precipitacao']).copy()
    
    if len(df_plot) == 0:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot['DateTime'],
        y=df_plot['Precipitacao'],
        name='Precipitação',
        marker_color='#3498db',
        marker_line_color='#2980b9',
        marker_line_width=1
    ))
    
    fig.update_layout(
        title='☔ Precipitação ao Longo do Tempo',
        xaxis_title='Data/Hora',
        yaxis_title='Precipitação (mm)',
        template='plotly_dark',
        height=400,
        bargap=0.1
    )
    
    return fig

def criar_grafico_umidade(df):
    """Cria gráfico de umidade"""
    if 'DateTime' not in df.columns or 'UR_Inst' not in df.columns:
        return None
    
    df_plot = df.dropna(subset=['UR_Inst']).copy()
    
    if len(df_plot) == 0:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot['DateTime'],
        y=df_plot['UR_Inst'],
        mode='lines',
        name='Umidade',
        line=dict(color='#2ecc71', width=2),
        fill='tozeroy',
        fillcolor='rgba(46, 204, 113, 0.1)'
    ))
    
    fig.update_layout(
        title='💧 Umidade Relativa ao Longo do Tempo',
        xaxis_title='Data/Hora',
        yaxis_title='Umidade (%)',
        template='plotly_dark',
        height=400,
        yaxis_range=[0, 100]
    )
    
    return fig

def criar_grafico_vento(df):
    """Cria gráfico de velocidade do vento"""
    if 'DateTime' not in df.columns or 'U2' not in df.columns:
        return None
    
    df_plot = df.dropna(subset=['U2']).copy()
    
    if len(df_plot) == 0:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot['DateTime'],
        y=df_plot['U2'],
        mode='lines',
        name='Vento',
        line=dict(color='#f39c12', width=2)
    ))
    
    fig.update_layout(
        title='💨 Velocidade do Vento ao Longo do Tempo',
        xaxis_title='Data/Hora',
        yaxis_title='Velocidade (m/s)',
        template='plotly_dark',
        height=400
    )
    
    return fig

def criar_grafico_comparativo_mensal(df_mensal):
    """Cria gráfico comparativo mensal"""
    if df_mensal is None or df_mensal.empty:
        return None
    
    fig = go.Figure()
    
    if 'Temp_Inst' in df_mensal.columns:
        fig.add_trace(go.Bar(
            x=df_mensal['Mês'],
            y=df_mensal['Temp_Inst'],
            name='Temperatura Média',
            marker_color='#e74c3c',
            yaxis='y'
        ))
    
    if 'Precipitacao' in df_mensal.columns:
        fig.add_trace(go.Scatter(
            x=df_mensal['Mês'],
            y=df_mensal['Precipitacao'],
            name='Precipitação Total',
            marker_color='#3498db',
            line=dict(width=3),
            yaxis='y2'
        ))
    
    fig.update_layout(
        title='📊 Comparativo Mensal',
        template='plotly_dark',
        height=450,
        xaxis_title='Mês',
        yaxis=dict(title='Temperatura (°C)', side='left', color='#e74c3c'),
        yaxis2=dict(title='Precipitação (mm)', overlaying='y', side='right', color='#3498db'),
        hovermode='x unified'
    )
    
    return fig

def criar_heatmap_correlacao(df):
    """Cria heatmap de correlação entre variáveis"""
    colunas_correlacao = ['Temp_Inst', 'UR_Inst', 'U2', 'Press_Inst']
    colunas_existentes = [c for c in colunas_correlacao if c in df.columns]
    
    if len(colunas_existentes) < 2:
        return None
    
    corr_matrix = df[colunas_existentes].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title='📈 Matriz de Correlação entre Variáveis',
        template='plotly_dark',
        height=450,
        width=550
    )
    
    return fig

# ============================================================================
# RELATÓRIO HTML
# ============================================================================

def gerar_relatorio_html(df_original, df_mensal, qualidade, eventos, substituicoes, outliers, nome_estacao):
    """Gera relatório HTML completo"""
    
    data_geracao = datetime.now().strftime('%d/%m/%Y às %H:%M')
    
    # Preparar tabela mensal
    tabela_mensal = ""
    if df_mensal is not None and not df_mensal.empty:
        tabela_mensal = df_mensal.to_html(index=False)
    
    # Preparar tabela de qualidade
    tabela_qualidade = ""
    for var, info in qualidade.items():
        if var != 'periodo' and isinstance(info, dict):
            tabela_qualidade += f"""
            <tr>
                <td>{var}</td>
                <td>{info.get('percentual_completo', 0)}%</td>
                <td>{info.get('classificacao', 'N/A')}</td>
                <td>{info.get('dados_faltantes', 0)}</td>
            </tr>
            """
    
    # Preparar substituições
    html_substituicoes = ""
    for s in substituicoes:
        if s['status'] == 'PREENCHIDO':
            html_substituicoes += f"<div class='success'>✅ {s['variavel']}: {s.get('motivo', '')}</div>"
        elif s['status'] == 'NÃO PREENCHIDO':
            html_substituicoes += f"<div class='warning'>⚠️ {s['variavel']}: {s.get('motivo', '')}</div>"
    
    # Preparar outliers
    html_outliers = ""
    for var, info in outliers.items():
        html_outliers += f"<div class='warning'>🔍 {var}: {info['num_outliers']} outliers ({info['percentual']}%) - Limites: {info['limites']}</div>"
    
    if not html_outliers:
        html_outliers = "<p>✅ Nenhum outlier detectado</p>"
    
    # Preparar eventos
    html_eventos = ""
    for e in eventos:
        html_eventos += f"<div class='warning'><strong>{e['tipo']}</strong>: {e['descricao']}</div>"
    
    if not html_eventos:
        html_eventos = "<p>✅ Nenhum evento extremo detectado</p>"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>Relatório Meteorológico - {nome_estacao}</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; }}
            .header {{ background: linear-gradient(135deg, #1a5f7a, #0d3b4f); color: white; padding: 30px; text-align: center; }}
            .section {{ padding: 20px; border-bottom: 1px solid #eee; }}
            .section h2 {{ color: #1a5f7a; border-left: 4px solid #4caf50; padding-left: 15px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
            th {{ background: #1a5f7a; color: white; }}
            .metric {{ display: inline-block; background: #e8f4f8; padding: 15px; margin: 10px; border-radius: 10px; min-width: 150px; text-align: center; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #1a5f7a; }}
            .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .success {{ background: #d4edda; border-left: 4px solid #28a745; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .footer {{ text-align: center; padding: 20px; font-size: 12px; color: #999; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌾 RELATÓRIO METEOROLÓGICO</h1>
                <h2>{nome_estacao}</h2>
                <p>Gerado em: {data_geracao}</p>
            </div>
            
            <div class="section">
                <h2>📊 Resumo do Processamento</h2>
                <div class="metric">
                    <div class="metric-value">{len(df_original):,}</div>
                    <div>Registros Originais</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(df_mensal) if df_mensal is not None else 0}</div>
                    <div>Meses Consolidados</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(eventos)}</div>
                    <div>Eventos Extremos</div>
                </div>
            </div>
            
            <div class="section">
                <h2>📅 Dados Mensais Consolidados</h2>
                {tabela_mensal if tabela_mensal else "<p>Não disponível</p>"}
            </div>
            
            <div class="section">
                <h2>📈 Qualidade dos Dados</h2>
                <table>
                    <tr><th>Variável</th><th>Completude</th><th>Classificação</th><th>Faltantes</th></tr>
                    {tabela_qualidade if tabela_qualidade else "<tr><td colspan='4'>Nenhum dado disponível</td></tr>"}
                </table>
            </div>
            
            <div class="section">
                <h2>🔄 Substituições Realizadas</h2>
                {html_substituicoes if html_substituicoes else "<p>✅ Nenhuma substituição necessária</p>"}
            </div>
            
            <div class="section">
                <h2>⚠️ Outliers Tratados</h2>
                {html_outliers}
            </div>
            
            <div class="section">
                <h2>🌡️ Eventos Climáticos Extremos</h2>
                {html_eventos}
            </div>
            
            <div class="footer">
                <p>Smart Meteorological Processor - Aplicativo para fins acadêmicos</p>
                <p>⚠️ Este aplicativo não pertence ao INMET</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def pipeline_completo(arquivo_bytes, config):
    """Pipeline completo de processamento"""
    
    resultados = {}
    
    try:
        # Leitura
        df, info_estacao = ler_arquivo_inteligente(arquivo_bytes)
        resultados['info_estacao'] = info_estacao
        resultados['dados_originais'] = df.copy()  # PRESERVA OS DADOS ORIGINAIS
        
        # Processamento
        df = padronizar_colunas(df)
        df = corrigir_datas(df)
        
        # Qualidade
        qualidade = analise_completa_qualidade(df)
        resultados['qualidade'] = qualidade
        
        # Preenchimento de falhas
        df, substituicoes = preenchimento_inteligente_falhas(df, limite_max_falhas=10)
        resultados['substituicoes'] = substituicoes
        
        # Tratamento de outliers
        df, outliers = detectar_outliers_tratar(df)
        resultados['outliers'] = outliers
        
        # Eventos extremos
        eventos = detectar_eventos_extremos(df)
        resultados['eventos'] = eventos
        
        # Dados consolidados (TABELA SEPARADA - NÃO ALTERA OS ORIGINAIS)
        df_mensal = consolidar_medias_mensais(df)
        resultados['dados_mensais'] = df_mensal
        resultados['dados_processados'] = df
        
        # Estatísticas
        resultados['total_registros_original'] = len(resultados['dados_originais'])
        resultados['total_registros_processados'] = len(df)
        resultados['total_meses'] = len(df_mensal) if df_mensal is not None else 0
        
        return resultados
        
    except Exception as e:
        st.error(f"❌ Erro no processamento: {str(e)}")
        return None

# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================

def main():
    # Cabeçalho
    st.markdown("""
    <div class="main-header">
        <h1>🌾 Smart Meteorological Processor</h1>
        <p style="color: #a0c4d6; font-size: 1.1rem;">
            Sistema Profissional de Tratamento, Análise e Visualização de Dados Meteorológicos
        </p>
        <p style="color: #81c784; font-size: 0.9rem;">
            🔬 Preserva dados originais | 📊 Médias mensais separadas | 🌡️ Tratamento inteligente | 📈 Gráficos interativos
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configurações")
        
        uploaded_file = st.file_uploader(
            "📂 **Upload do Arquivo**",
            type=['csv', 'txt', 'dat', 'xlsx', 'xls'],
            help="Arquivos de dados meteorológicos"
        )
        
        st.markdown("---")
        
        st.markdown("### 🔧 Configurações de Processamento")
        limite_falhas = st.slider("Limite máximo de falhas consecutivas", 5, 30, 10)
        fator_outliers = st.slider("Fator para detecção de outliers (IQR)", 2.0, 5.0, 3.0, 0.5)
        
        st.markdown("---")
        
        st.markdown("### 🤖 IA Gemini")
        usar_ia = st.checkbox("Ativar análise com IA", value=True)
        
        st.markdown("---")
        st.caption("🔬 Desenvolvido para análise profissional")
        st.caption("⚠️ Aplicativo para fins acadêmicos")
    
    # Área principal
    if uploaded_file is not None:
        
        config = {
            'limite_falhas': limite_falhas,
            'fator_outliers': fator_outliers,
            'usar_ia': usar_ia
        }
        
        with st.spinner("🔄 Processando dados..."):
            resultados = pipeline_completo(uploaded_file, config)
        
        if resultados:
            df_original = resultados['dados_originais']
            df_processado = resultados['dados_processados']
            df_mensal = resultados['dados_mensais']
            qualidade = resultados.get('qualidade', {})
            eventos = resultados.get('eventos', [])
            substituicoes = resultados.get('substituicoes', [])
            outliers = resultados.get('outliers', {})
            info_estacao = resultados.get('info_estacao', {})
            
            nome_estacao = info_estacao.get('station_name', 'Estação Meteorológica')
            
            # Cards de resumo
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Registros Originais", f"{resultados['total_registros_original']:,}")
            with col2:
                st.metric("✅ Registros Processados", f"{resultados['total_registros_processados']:,}")
            with col3:
                st.metric("📅 Meses Consolidados", resultados['total_meses'])
            with col4:
                st.metric("⚠️ Eventos Extremos", len(eventos))
            
            # Exibir alertas
            if substituicoes:
                with st.expander("📋 **Dados substituídos no preenchimento de falhas**", expanded=False):
                    for sub in substituicoes:
                        if sub['status'] == 'PREENCHIDO':
                            st.success(f"✅ {sub['variavel']}: {sub.get('quantidade', 0)} valores preenchidos")
                        elif sub['status'] == 'NÃO PREENCHIDO':
                            st.warning(f"⚠️ {sub['variavel']}: {sub.get('motivo', '')}")
            
            if outliers:
                with st.expander("📊 **Outliers detectados e tratados**", expanded=False):
                    for var, info in outliers.items():
                        st.info(f"🔍 {var}: {info['num_outliers']} outliers ({info['percentual']}%) - Limites: {info['limites']}")
            
            # TABS
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Dados Originais",
                "📅 Médias Mensais",
                "📈 Gráficos",
                "🔬 Análise de Qualidade",
                "🤖 IA Gemini",
                "📥 Downloads"
            ])
            
            # TAB 1: DADOS ORIGINAIS
            with tab1:
                st.markdown("### 📊 Dados Originais (Preservados na Íntegra)")
                st.caption(f"Total de registros: {len(df_original):,} linhas")
                
                # Amostra dos dados
                st.dataframe(df_original.head(100), use_container_width=True)
                
                # Estatísticas descritivas
                st.markdown("### 📈 Estatísticas Descritivas")
                colunas_stats = ['Temp_Inst', 'UR_Inst', 'Precipitacao', 'U2']
                colunas_existentes = [c for c in colunas_stats if c in df_original.columns]
                if colunas_existentes:
                    st.dataframe(df_original[colunas_existentes].describe(), use_container_width=True)
            
            # TAB 2: MÉDIAS MENSAIS
            with tab2:
                st.markdown("### 📅 Dados Consolidados por Mês")
                st.caption("Médias mensais calculadas a partir dos dados horários")
                
                if df_mensal is not None and not df_mensal.empty:
                    st.dataframe(df_mensal, use_container_width=True)
                    
                    # Gráfico comparativo mensal
                    fig = criar_grafico_comparativo_mensal(df_mensal)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Dados mensais não disponíveis para este arquivo")
            
            # TAB 3: GRÁFICOS
            with tab3:
                st.markdown("### 📈 Visualizações Interativas")
                
                tipo_grafico = st.radio("Selecione o tipo de visualização:", 
                                        ["Série Temporal", "Boxplot por Hora", "Heatmap de Correlação"],
                                        horizontal=True)
                
                if tipo_grafico == "Série Temporal":
                    col_graf1, col_graf2 = st.columns(2)
                    
                    with col_graf1:
                        fig_temp = criar_grafico_temperatura(df_processado, 'linha')
                        if fig_temp:
                            st.plotly_chart(fig_temp, use_container_width=True)
                        else:
                            st.info("Dados de temperatura não disponíveis")
                    
                    with col_graf2:
                        fig_umid = criar_grafico_umidade(df_processado)
                        if fig_umid:
                            st.plotly_chart(fig_umid, use_container_width=True)
                        else:
                            st.info("Dados de umidade não disponíveis")
                    
                    fig_precip = criar_grafico_precipitacao(df_processado)
                    if fig_precip:
                        st.plotly_chart(fig_precip, use_container_width=True)
                    else:
                        st.info("Dados de precipitação não disponíveis")
                    
                    fig_vento = criar_grafico_vento(df_processado)
                    if fig_vento:
                        st.plotly_chart(fig_vento, use_container_width=True)
                    else:
                        st.info("Dados de vento não disponíveis")
                
                elif tipo_grafico == "Boxplot por Hora":
                    fig_box = criar_grafico_temperatura(df_processado, 'box')
                    if fig_box:
                        st.plotly_chart(fig_box, use_container_width=True)
                    else:
                        st.info("Dados insuficientes para boxplot")
                
                else:  # Heatmap
                    fig_heat = criar_heatmap_correlacao(df_processado)
                    if fig_heat:
                        st.plotly_chart(fig_heat, use_container_width=True)
                    else:
                        st.info("Dados insuficientes para matriz de correlação")
            
            # TAB 4: ANÁLISE DE QUALIDADE
            with tab4:
                st.markdown("### 📋 Análise de Qualidade dos Dados")
                
                if qualidade:
                    for var, info in qualidade.items():
                        if var != 'periodo' and isinstance(info, dict):
                            percentual = info.get('percentual_completo', 0)
                            if percentual >= 95:
                                st.success(f"**{var}**: {percentual:.1f}% completo ✅")
                            elif percentual >= 80:
                                st.warning(f"**{var}**: {percentual:.1f}% completo ⚠️")
                            else:
                                st.error(f"**{var}**: {percentual:.1f}% completo ❌")
                    
                    if 'periodo' in qualidade and isinstance(qualidade['periodo'], dict):
                        st.markdown("#### 📅 Período Analisado")
                        st.write(f"Início: {qualidade['periodo'].get('inicio', 'N/A')}")
                        st.write(f"Fim: {qualidade['periodo'].get('fim', 'N/A')}")
                        st.write(f"Total de registros: {qualidade['periodo'].get('total_registros', 0):,}")
            
            # TAB 5: IA GEMINI
            with tab5:
                st.markdown("### 🤖 Análise com IA Gemini")
                
                if usar_ia:
                    tipo_analise = st.selectbox(
                        "Tipo de análise:",
                        ["Análise Climática Geral", "Identificação de Fenômenos", "Impactos na Agricultura"]
                    )
                    
                    if st.button("🚀 Executar Análise", use_container_width=True):
                        with st.spinner("Consultando IA..."):
                            contexto = f"""
                            Estação: {nome_estacao}
                            Registros processados: {resultados['total_registros_processados']:,}
                            Meses analisados: {resultados['total_meses']}
                            """
                            
                            if 'Temp_Inst' in df_processado.columns:
                                contexto += f"Temperatura média: {df_processado['Temp_Inst'].mean():.1f}°C\n"
                            if 'Precipitacao' in df_processado.columns:
                                contexto += f"Precipitação total: {df_processado['Precipitacao'].sum():.1f} mm\n"
                            if 'UR_Inst' in df_processado.columns:
                                contexto += f"Umidade média: {df_processado['UR_Inst'].mean():.1f}%\n"
                            
                            prompt = f"Faça uma {tipo_analise.lower()} baseada nos dados fornecidos."
                            resposta = consultar_gemini(prompt, contexto)
                            
                            st.markdown("### 🧠 Resposta da IA")
                            st.markdown(resposta)
                else:
                    st.info("ℹ️ Ative a IA Gemini na barra lateral")
            
            # TAB 6: DOWNLOADS
            with tab6:
                st.markdown("### 💾 Downloads")
                
                col_down1, col_down2 = st.columns(2)
                
                with col_down1:
                    # CSV dos dados originais
                    csv_original = df_original.to_csv(index=False, sep=';', decimal=',')
                    st.download_button(
                        label="📥 Dados Originais (CSV)",
                        data=csv_original,
                        file_name=f"{nome_estacao.replace(' ', '_')}_dados_originais.csv",
                        mime="text/csv",
                        key="download_original"
                    )
                    
                    # CSV dos dados processados
                    csv_processado = df_processado.to_csv(index=False, sep=';', decimal=',')
                    st.download_button(
                        label="📥 Dados Processados (CSV)",
                        data=csv_processado,
                        file_name=f"{nome_estacao.replace(' ', '_')}_dados_processados.csv",
                        mime="text/csv",
                        key="download_processado"
                    )
                
                with col_down2:
                    # CSV das médias mensais
                    if df_mensal is not None and not df_mensal.empty:
                        csv_mensal = df_mensal.to_csv(index=False, sep=';', decimal=',')
                        st.download_button(
                            label="📥 Médias Mensais (CSV)",
                            data=csv_mensal,
                            file_name=f"{nome_estacao.replace(' ', '_')}_medias_mensais.csv",
                            mime="text/csv",
                            key="download_mensal"
                        )
                    
                    # Relatório HTML
                    relatorio_html = gerar_relatorio_html(
                        df_original, df_mensal, qualidade, eventos, 
                        substituicoes, outliers, nome_estacao
                    )
                    b64 = base64.b64encode(relatorio_html.encode()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="{nome_estacao.replace(" ", "_")}_relatorio.html" style="text-decoration: none;">'
                    href += '<button style="background: linear-gradient(135deg, #2e7d32, #1b5e20); border: none; border-radius: 30px; padding: 10px 25px; color: white; font-weight: 600; cursor: pointer; width: 100%; margin-top: 0px;">📑 Baixar Relatório HTML</button></a>'
                    st.markdown(href, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>⚠️ Smart Meteorological Processor - Aplicativo para fins acadêmicos e de pesquisa</p>
        <p>Este aplicativo não pertence ao INMET. Os dados processados devem ser validados antes de uso profissional.</p>
        <p>Código acadêmico - v2.0</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    main()
