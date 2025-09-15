
#@title Script para processamento de dados V1✅

import os
import glob
import pandas as pd
from datetime import datetime, timezone


# Configurações de pastas
ROOT = "/content/indicador-preditivo"
RAW_DIR = os.path.join(ROOT, "data", "raw")
CLEANED_DIR = os.path.join(ROOT, "data", "cleaned")
TRANSFORMED_DIR = os.path.join(ROOT, "data", "transformed")
LOG_DIR = os.path.join(ROOT, "data", "logs")

# Criar pastas se não existirem
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(TRANSFORMED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

RAW_FILENAME = "ETHUSD_candles.csv"

# Função de log
def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(msg)
    with open(os.path.join(LOG_DIR, f"process_log_{datetime.now(timezone.utc).strftime('%Y%m%d')}.txt"), "a") as f:
        f.write(f"{ts} - {msg}\n")


# Localizar arquivo raw
candidates = [
    os.path.join(RAW_DIR, RAW_FILENAME),
    os.path.join(ROOT, RAW_FILENAME),
    os.path.join(os.getcwd(), RAW_FILENAME),
    os.path.join("/content", RAW_FILENAME)
]

candidates += glob.glob(os.path.join(ROOT, "**", RAW_FILENAME), recursive=True)

raw_path = next((p for p in candidates if p and os.path.exists(p)), None)
if raw_path is None:
    raise FileNotFoundError(f"Arquivo {RAW_FILENAME} não encontrado.")


# Parâmetros
MA_PERIODS = [5, 10, 20, 50, 100, 200]  # Períodos para SMA e EMA


# Leitura do CSV
log(f"Lendo arquivo bruto: {raw_path}")
df = pd.read_csv(raw_path)


# Identificar coluna de tempo
time_col = None
for candidate in ["from", "datetime", "timestamp", "time"]:
    if candidate in df.columns:
        time_col = candidate
        break

if time_col is None:
    # Tentar inferir convertendo para datetime
    for c in df.columns:
        try:
            pd.to_datetime(df[c].iloc[0])
            time_col = c
            break
        except Exception:
            continue

if time_col is None:
    raise RuntimeError("Não foi possível identificar coluna de tempo no CSV.")

# Converter para datetime UTC
df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
if df[time_col].isna().all():
    df[time_col] = pd.to_datetime(df[time_col].astype(str), errors="coerce")


# Renomear colunas para padrão
col_map = {}
if "open" in df.columns: col_map["open"] = "abertura"
if "high" in df.columns: col_map["high"] = "maxima"
if "low" in df.columns: col_map["low"] = "minima"
if "close" in df.columns: col_map["close"] = "fechamento"
if col_map:
    df = df.rename(columns=col_map)

# Verificar colunas obrigatórias
required = [time_col, "abertura", "maxima", "minima", "fechamento", "volume"]
for r in required:
    if r not in df.columns:
        raise RuntimeError(f"Coluna obrigatória ausente: {r}")


# Ordenar por tempo
df = df.sort_values(by=time_col).reset_index(drop=True)
df = df.rename(columns={time_col: "timestamp"})


# Calcular pressões compradora e vendedora
df["pressao_compradora"] = df["maxima"] - df["fechamento"]
df["pressao_vendedora"] = df["fechamento"] - df["minima"]


# Médias móveis SMA e EMA
for p in MA_PERIODS:
    df[f"SMA_{p}"] = df["fechamento"].rolling(window=p, min_periods=1).mean()
    df[f"EMA_{p}"] = df["fechamento"].ewm(span=p, adjust=False).mean()


# Suporte, resistência e distâncias
df['resistencia'] = df['maxima'].rolling(window=20, min_periods=1).max()
df['suporte'] = df['minima'].rolling(window=20, min_periods=1).min()
df['dist_resistencia'] = df['resistencia'] - df['fechamento']
df['dist_suporte'] = df['fechamento'] - df['suporte']


# Variação do fechamento
df['var_fechamento'] = df['fechamento'].diff()


# RSI simples
delta = df['fechamento'].diff()
up = delta.clip(lower=0)
down = -1*delta.clip(upper=0)
roll_up = up.rolling(14, min_periods=1).mean()
roll_down = down.rolling(14, min_periods=1).mean()
rs = roll_up / roll_down
df['RSI_14'] = 100 - (100 / (1 + rs))


# Volume médio
df['vol_media_5'] = df['volume'].rolling(5, min_periods=1).mean()
df['vol_media_20'] = df['volume'].rolling(20, min_periods=1).mean()


# Tendência futura (+1 alta, -1 baixa, 0 neutra)
df['tendencia_futura'] = df['fechamento'].shift(-1) - df['fechamento']
df['tendencia_futura'] = df['tendencia_futura'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))


# Salvar CSV limpo
ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
cleaned_fname = f"ETHUSD_cleaned_v1_{ts}.csv"
cleaned_path = os.path.join(CLEANED_DIR, cleaned_fname)
df.to_csv(cleaned_path, index=False)

log(f"Dados limpos salvos em {cleaned_path}")
log("Processamento V1 concluído com sucesso.")
