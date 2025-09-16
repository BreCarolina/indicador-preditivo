#@title Script para extração de dados IQ OPTION✅

import os
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import sys

# -----------------------------
# Configurações de credenciais
# -----------------------------
load_dotenv("/content/drive/MyDrive/Projetos/indicador-preditivo/.env")
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")

# Ajustar caminho da API local
sys.path.append("/content/indicador-preditivo/iqoptionaapi")
from stable_api import IQ_Option

"""
Timeframes suportados (segundos):
- 60   → M1 (1 minuto)  → 1440 candles/dia
- 300  → M5 (5 minutos) → 288 candles/dia
- 900  → M15 (15 min)   → 96 candles/dia
- 1800 → M30 (30 min)   → 48 candles/dia
- 3600 → H1 (1 hora)    → 24 candles/dia
"""

# -----------------------------
# Parâmetros principais
# -----------------------------
PAR = "ETHUSD"     # Ativo
TIMEFRAME = 300    # Intervalo em segundos (ex: M5 = 300)
DIAS = 30          # Quantidade de dias de histórico

# Converte dias em total de candles
candles_por_dia = int((24 * 60) / (TIMEFRAME / 60))
TOTAL_CANDLES = DIAS * candles_por_dia


# -----------------------------
# Funções auxiliares
# -----------------------------
def conectar_api():
    """Estabelece conexão com a API da IQ Option"""
    api = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
    conectado, _ = api.connect()
    if not conectado:
        raise SystemExit("Erro: não foi possível conectar à IQ Option.")
    print("Conectado com sucesso.")
    return api


def buscar_candles(api, par=PAR, timeframe=TIMEFRAME, total_candles=TOTAL_CANDLES):
    """
    Busca candles históricos da IQ Option em blocos.
    Retorna um DataFrame com colunas: [from, abertura, maxima, minima, fechamento, volume].
    """
    max_por_chamada = 1000
    todos_candles = []
    agora = time.time()

    while total_candles > 0:
        qtd = min(max_por_chamada, total_candles)
        candles = api.get_candles(par, timeframe, qtd, agora)
        if not candles:
            break
        todos_candles.extend(candles)
        total_candles -= qtd
        agora = candles[0]["from"]  # timestamp do candle mais antigo

    df = pd.DataFrame(todos_candles)
    df["from"] = pd.to_datetime(df["from"], unit="s", utc=True)  # mantém em UTC
    df = df.rename(columns={
        "open": "abertura",
        "max": "maxima",
        "min": "minima",
        "close": "fechamento",
        "volume": "volume"
    })

    return df[["from", "abertura", "maxima", "minima", "fechamento", "volume"]]


def extrair_dados(par=PAR, timeframe=TIMEFRAME, dias=DIAS, root="/content/indicador-preditivo"):
    """
    Extrai dados da IQ Option ou usa cache já salvo em /data/raw.
    """
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    caminho_csv = os.path.join(raw_dir, f"{par}_M{timeframe//60}_{dias}d.csv")

    atualizar = True
    if os.path.exists(caminho_csv):
        df = pd.read_csv(caminho_csv, parse_dates=["from"])
        ultimo = df["from"].max().tz_localize("UTC") if df["from"].dt.tz is None else df["from"].max()
        if datetime.now(timezone.utc) - ultimo < timedelta(minutes=timeframe // 60):
            print("Usando dados já salvos.")
            atualizar = False
    else:
        df = pd.DataFrame()

    if atualizar:
        print(f"Baixando {dias} dias de dados ({TOTAL_CANDLES} candles em M{timeframe//60})...")
        api = conectar_api()
        df = buscar_candles(api, par, timeframe, TOTAL_CANDLES)
        df.to_csv(caminho_csv, index=False)
        print(f"Dados salvos em {caminho_csv}.")

    return df, caminho_csv


# -----------------------------
# Execução direta
# -----------------------------
if __name__ == "__main__":
    dados, path = extrair_dados(PAR, TIMEFRAME, DIAS)
    print("\nÚltimos 5 candles:")
    print(dados.tail())
