#@title Script para extração de dados IQ OPTION V2✅


import os
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import sys

# Carregar credenciais
load_dotenv("/content/drive/MyDrive/Projetos/indicador-preditivo/.env")
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")

# Ajustar caminho da API
sys.path.append("/content/indicador-preditivo/iqoptionaapi")
from stable_api import IQ_Option

"""
O timeframe é o intervalo em segundos:
60 → 1 minuto (M1)
300 → 5 minutos (M5)
900 → 15 minutos (M15)
1800 → 30 minutos (M30)
3600 → 1 hora (H1)

M1 (1 minuto) → 60 candles por hora × 24 horas = 1440 candles por dia.
M5 (5 minutos) → 12 candles por hora × 24 horas = 288 candles por dia.
M15 (15 minutos) → 4 candles por hora × 24 horas = 96 candles por dia.
M30 (30 minutos) → 2 candles por hora × 24 horas = 48 candles por dia.
H1 (1 hora) → 1 candle por hora × 24 horas = 24 candles por dia.

"""

# Configurações principais
PAR = "ETHUSD"     # Ativo
TIMEFRAME = 300    # em segundos (M5 = 300, M1 = 60, M15 = 900)
DIAS = 30         # Quantos dias de histórico coletar

# Converte DIAS + TIMEFRAME em quantidade de candles
candles_por_dia = int((24 * 60) / (TIMEFRAME / 60))
TOTAL_CANDLES = DIAS * candles_por_dia

def conectar_api():
    """Estabelece conexão com a API da IQ Option"""
    api = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
    conectado, _ = api.connect()
    if not conectado:
        raise SystemExit("Erro: não foi possível conectar à IQ Option.")
    print("Conectado com sucesso.")
    return api

def buscar_candles(api, par=PAR, timeframe=TIMEFRAME, total_candles=TOTAL_CANDLES):
    """Busca candles históricos da IQ Option em blocos"""
    max_por_chamada = 1000  # limite da API
    todos_candles = []
    agora = time.time()

    while total_candles > 0:
        qtd = min(max_por_chamada, total_candles)
        candles = api.get_candles(par, timeframe, qtd, agora)
        if not candles:
            break
        todos_candles.extend(candles)
        total_candles -= qtd
        agora = candles[0]['from']  # timestamp do candle mais antigo

    df = pd.DataFrame(todos_candles)
    df["from"] = pd.to_datetime(df["from"], unit="s", utc=True).dt.tz_convert("America/Sao_Paulo")
    df = df.rename(columns={
        "open": "abertura",
        "max": "maxima",
        "min": "minima",
        "close": "fechamento",
        "volume": "volume"
    })

    return df[["from", "abertura", "maxima", "minima", "fechamento", "volume"]]


def carregar_dados(par=PAR):
    """Carrega dados já existentes ou baixa novos"""
    caminho_csv = f"/content/indicador-preditivo/data/raw/{par}_M{TIMEFRAME//60}_{DIAS}d.csv"

    atualizar = True

    if os.path.exists(caminho_csv):
        df = pd.read_csv(caminho_csv, parse_dates=["from"])
        ultimo = df["from"].max().to_pydatetime()
        if datetime.now(timezone.utc) - ultimo < timedelta(minutes=TIMEFRAME//60):
            print("Usando dados salvos.")
            atualizar = False
    else:
        df = pd.DataFrame()

    if atualizar:
        print(f"Buscando {DIAS} dias de dados ({TOTAL_CANDLES} candles em M{TIMEFRAME//60})...")
        api = conectar_api()
        df = buscar_candles(api, par)
        df.to_csv(caminho_csv, index=False)
        print(f"Dados salvos em {caminho_csv}.")

    return df


if __name__ == "__main__":
    dados = carregar_dados(PAR)
    print("\nÚltimos 5 candles:")
    print(dados.tail())
    print(dados.head())
