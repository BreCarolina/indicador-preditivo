#@title Script para transformação e criação de features ✅

import os
import pandas as pd
from datetime import datetime, timezone

"""
Script de transformação de dados para o projeto Indicador Preditivo.

Este script:
1. Lê os dados brutos (candles) já coletados.
2. Organiza e valida colunas essenciais.
3. Cria um conjunto de features técnicas e estatísticas:
   - Pressão compradora/vendedora
   - Médias móveis (SMA, EMA)
   - Suportes, resistências e distâncias relativas
   - Variação do fechamento
   - RSI (14 períodos)
   - Volumes médios (5 e 20)
   - Target contínuo: fechamento futuro (regressão)
   - Features temporais (hora, minuto, dia da semana)
   - Retorno percentual
   - Volatilidade (desvio padrão móvel)
4. Salva o dataset transformado no diretório definido.

Resultado: um CSV no diretório /data/transformed com todas as features prontas.
"""

# Configurações principais
ROOT = "/content/indicador-preditivo"
PAR = "ETHUSD"
TIMEFRAME = 300    # em segundos (M5 = 300, M1 = 60, M15 = 900)
DIAS = 30

# Estrutura de diretórios
RAW_DIR = os.path.join(ROOT, "data", "raw")       
CLEANED_DIR = os.path.join(ROOT, "data", "cleaned")   
TRANSFORMED_DIR = os.path.join(ROOT, "data", "transformed") 
LOG_DIR = os.path.join(ROOT, "data", "logs")

# Criação das pastas se não existirem
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(TRANSFORMED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Nome do arquivo bruto esperado
RAW_FILENAME = f"{PAR}_M{TIMEFRAME//60}_{DIAS}d.csv"  
RAW_PATH = os.path.join(RAW_DIR, RAW_FILENAME)

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Arquivo bruto não encontrado em {RAW_PATH}")

def log(msg):
    """Registra mensagem em log (console + arquivo)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(msg)
    with open(os.path.join(LOG_DIR, f"process_log_{datetime.now(timezone.utc).strftime('%Y%m%d')}.txt"), "a") as f:
        f.write(f"{ts} - {msg}\n")

# Períodos para médias móveis
MA_PERIODS = [5, 10, 20, 50, 100, 200]  

# Leitura do CSV bruto
log(f"Lendo arquivo bruto: {RAW_PATH}")
df = pd.read_csv(RAW_PATH)

# Identificar coluna de tempo
time_col = None
for candidate in ["from", "datetime", "timestamp", "time"]:
    if candidate in df.columns:
        time_col = candidate
        break

if time_col is None:
    for c in df.columns:
        try:
            pd.to_datetime(df[c].iloc[0])
            time_col = c
            break
        except Exception:
            continue

if time_col is None:
    raise RuntimeError("Não foi possível identificar coluna de tempo no CSV.")

# Converter coluna de tempo para datetime UTC
df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
if df[time_col].isna().all():
    df[time_col] = pd.to_datetime(df[time_col].astype(str), errors="coerce")

# Validar colunas obrigatórias
required = [time_col, "abertura", "maxima", "minima", "fechamento", "volume"]
for r in required:
    if r not in df.columns:
        raise RuntimeError(f"Coluna obrigatória ausente: {r}")

# Ordenar por tempo e padronizar nome
df = df.sort_values(by=time_col).reset_index(drop=True)
df = df.rename(columns={time_col: "timestamp"})

# -----------------------------------
# Features técnicas
# -----------------------------------

# Pressões compradora e vendedora
df["pressao_compradora"] = df["maxima"] - df["fechamento"]
df["pressao_vendedora"] = df["fechamento"] - df["minima"]

# Médias móveis simples e exponenciais
for p in MA_PERIODS:
    df[f"SMA_{p}"] = df["fechamento"].rolling(window=p, min_periods=1).mean()
    df[f"EMA_{p}"] = df["fechamento"].ewm(span=p, adjust=False).mean()

# Suporte, resistência e distâncias
df['resistencia'] = df['maxima'].rolling(window=20, min_periods=1).max()
df['suporte'] = df['minima'].rolling(window=20, min_periods=1).min()
df['dist_resistencia'] = df['resistencia'] - df['fechamento']
df['dist_suporte'] = df['fechamento'] - df['suporte']

# Variação absoluta do fechamento
df['var_fechamento'] = df['fechamento'].diff()

# RSI (14 períodos)
delta = df['fechamento'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
roll_up = up.rolling(14, min_periods=1).mean()
roll_down = down.rolling(14, min_periods=1).mean()
rs = roll_up / roll_down
df['RSI_14'] = 100 - (100 / (1 + rs))

# Volumes médios
df['vol_media_5'] = df['volume'].rolling(5, min_periods=1).mean()
df['vol_media_20'] = df['volume'].rolling(20, min_periods=1).mean()

# Target contínuo (fechamento futuro)
df['fechamento_futuro'] = df['fechamento'].shift(-1)

# -----------------------------------
# Features adicionais (temporais e estatísticas)
# -----------------------------------

def adicionar_features_temporais(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona hora, minuto e dia da semana."""
    df['hora_num'] = df['timestamp'].dt.hour
    df['minuto'] = df['timestamp'].dt.minute
    df['dia_semana'] = df['timestamp'].dt.dayofweek
    return df

def calcular_retorno(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna de retorno percentual entre candles consecutivos."""
    df['retorno'] = df['fechamento'].pct_change().fillna(0)
    return df

def calcular_volatilidade(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona volatilidade como desvio padrão móvel do retorno (janela=10)."""
    df['volatilidade'] = df['retorno'].rolling(window=10, min_periods=1).std().fillna(0)
    return df

# Aplicar funções de novas features
df = adicionar_features_temporais(df)
df = calcular_retorno(df)
df = calcular_volatilidade(df)

# Remover NaNs gerados por shift
df = df.dropna().reset_index(drop=True)

# -----------------------------------
# Salvar resultado
# -----------------------------------
ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
transformed_fname = f"{PAR}_M{TIMEFRAME//60}_{DIAS}d_transformed_{ts}.csv"
transformed_path = os.path.join(TRANSFORMED_DIR, transformed_fname)
df.to_csv(transformed_path, index=False)

log(f"Dados transformados salvos em {transformed_path}")
log("Transformação concluída com sucesso.")
