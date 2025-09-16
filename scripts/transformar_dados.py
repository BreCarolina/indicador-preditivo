#@title Script para transformação e criação de features ✅
"""
Script: transformar_dados.py

Descrição:
-----------
Este script lê os dados brutos de candles (extraídos pelo `extrair_dados.py`),
enriquece com features técnicas e estatísticas, e salva um novo CSV transformado
para uso em treinamento de modelos preditivos.

Fluxo de execução:
-------------------
1. Lê o CSV bruto gerado pelo extrair_dados.py.
2. Padroniza a coluna de tempo como `timestamp` e organiza em ordem cronológica.
3. Cria features derivadas:
   - Pressão compradora/vendedora
   - Médias móveis simples (SMA) e exponenciais (EMA)
   - Suporte, resistência e distâncias
   - Variação de fechamento
   - Indicador RSI (14 períodos)
   - Médias móveis de volume
   - Fechamento futuro (shift -1)
   - Features de tempo (hora, minuto, dia da semana)
   - Retorno percentual e volatilidade
4. Salva dataset transformado em `/data/transformed/` com timestamp no nome.
5. Registra logs em `/data/logs/`.

Parâmetros configuráveis:
--------------------------
raw_path   -> Caminho do CSV bruto já salvo (obrigatório).
par        -> Ativo, ex: "ETHUSD".
timeframe  -> Intervalo em segundos (M1=60, M5=300, ...).
dias       -> Número de dias de histórico usado no nome do arquivo final.
root       -> Caminho raiz do projeto.

Saídas:
--------
- Arquivo CSV salvo em: {root}/data/transformed/{par}_M{timeframe//60}_{dias}d_transformed_{timestamp}.csv
- Caminho do arquivo transformado retornado pela função.

Integração:
------------
Este script pode ser importado em outros módulos (como main.py) e usado via a função
`transformar_dados()`, que retorna o caminho do CSV transformado.
"""

import os
import pandas as pd
from datetime import datetime, timezone

def transformar_dados(raw_path, par="ETHUSD", timeframe=300, dias=30, root="/content/indicador-preditivo"):
    """
    Transforma dados brutos (candles) em dataset enriquecido com features.

    Args:
        raw_path (str): Caminho do CSV bruto já salvo.
        par (str): Ativo, ex: ETHUSD.
        timeframe (int): Intervalo em segundos (M1=60, M5=300, ...).
        dias (int): Número de dias usados no nome do arquivo final.
        root (str): Caminho raiz do projeto.

    Returns:
        str: Caminho do arquivo transformado salvo.
    """

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
    # Exemplo de uso direto
    raw_path = "/content/indicador-preditivo/data/raw/ETHUSD_M5_30d.csv"
    transformar_dados(raw_path)
