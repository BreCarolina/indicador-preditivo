#@title Script para transformação e criação de features ✅
"""
Script: transformar_dados.py

Descrição:
-----------
Este script lê os dados brutos de candles (extraídos pelo `extrair_dados.py`),
enriquece com features técnicas e estatísticas, e salva um novo CSV transformado
para uso em treinamento de modelos preditivos.
"""

import os
import pandas as pd
from datetime import datetime, timezone

def transformar_dados(raw_path, par="ETHUSD", timeframe=300, dias=30, root="/content/indicador-preditivo"):
    # Estrutura de diretórios
    transformed_dir = os.path.join(root, "data", "transformed")
    log_dir = os.path.join(root, "data", "logs")
    os.makedirs(transformed_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    def log(msg):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        print(msg)
        with open(os.path.join(log_dir, f"process_log_{datetime.now(timezone.utc).strftime('%Y%m%d')}.txt"), "a") as f:
            f.write(f"{ts} - {msg}\n")

    # Leitura do CSV bruto
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Arquivo bruto não encontrado em {raw_path}")

    log(f"Lendo arquivo bruto: {raw_path}")
    df = pd.read_csv(raw_path)

    # Garantir coluna de tempo
    time_col = None
    for candidate in ["from", "datetime", "timestamp", "time"]:
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        raise RuntimeError("Nenhuma coluna de tempo encontrada no CSV.")

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.rename(columns={time_col: "timestamp"}).sort_values("timestamp").reset_index(drop=True)

    # Features de candle
    df["pressao_compradora"] = df["maxima"] - df["fechamento"]
    df["pressao_vendedora"] = df["fechamento"] - df["minima"]

    # Médias móveis
    for p in [5, 10, 20, 50, 100, 200]:
        df[f"SMA_{p}"] = df["fechamento"].rolling(window=p, min_periods=1).mean()
        df[f"EMA_{p}"] = df["fechamento"].ewm(span=p, adjust=False).mean()

    # Suporte e resistência
    df["resistencia"] = df["maxima"].rolling(window=20, min_periods=1).max()
    df["suporte"] = df["minima"].rolling(window=20, min_periods=1).min()
    df["dist_resistencia"] = df["resistencia"] - df["fechamento"]
    df["dist_suporte"] = df["fechamento"] - df["suporte"]

    # Variação e indicadores técnicos
    df["var_fechamento"] = df["fechamento"].diff()
    delta = df["fechamento"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.rolling(14, min_periods=1).mean() / down.rolling(14, min_periods=1).mean()
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Volume
    df["vol_media_5"] = df["volume"].rolling(5, min_periods=1).mean()
    df["vol_media_20"] = df["volume"].rolling(20, min_periods=1).mean()

    # Variáveis futuras e temporais
    df["fechamento_futuro"] = df["fechamento"].shift(-1)
    df["hora_num"] = df["timestamp"].dt.hour
    df["minuto"] = df["timestamp"].dt.minute
    df["dia_semana"] = df["timestamp"].dt.dayofweek
    df["retorno"] = df["fechamento"].pct_change().fillna(0)
    df["volatilidade"] = df["retorno"].rolling(window=10, min_periods=1).std().fillna(0)

    # Salvar dataset transformado
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    transformed_fname = f"{par}_M{timeframe//60}_{dias}d_transformed_{ts}.csv"
    transformed_path = os.path.join(transformed_dir, transformed_fname)
    df.to_csv(transformed_path, index=False)

    log(f"Dados transformados salvos em {transformed_path}")
    return transformed_path


if __name__ == "__main__":
    par = "ETHUSD"
    timeframe = 300   # M5
    dias = 90
    root = "/content/indicador-preditivo"

    raw_path = os.path.join(root, "data", "raw", f"{par}_M{timeframe//60}_{dias}d.csv")
    transformar_dados(raw_path, par=par, timeframe=timeframe, dias=dias, root=root)
