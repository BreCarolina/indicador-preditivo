#@title Script para coleta de dados IQ OPTION V1.1✅

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

# Conectar na IQ Option
def conectar_api():
    api = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
    conectado, _ = api.connect()
    if not conectado:
        raise SystemExit("Erro: não foi possível conectar à IQ Option.")
    print("Conectado com sucesso.")
    return api

def buscar_candles(api, par="ETHUSD", timeframe=900, total_candles=30000):

    max_por_chamada = 1000  # ajustar conforme limite da API
    todos_candles = []
    agora = time.time()

    while total_candles > 0:
        qtd = min(max_por_chamada, total_candles)
        candles = api.get_candles(par, timeframe, qtd, agora)
        if not candles:
            break  # caso a API retorne vazio
        todos_candles.extend(candles)
        total_candles -= qtd
        # retroceder o tempo para pegar candles anteriores
        agora = candles[0]['from']  # pegar timestamp do candle mais antigo retornado

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

# Carregar ou atualizar CSV
def carregar_dados(par="ETHUSD"):
    caminho_csv = f"/content/indicador-preditivo/data/raw/{par}_candles.csv"

    atualizar = True

    if os.path.exists(caminho_csv):
        df = pd.read_csv(caminho_csv, parse_dates=["from"])
        ultimo = df["from"].max().to_pydatetime()
        if datetime.now(timezone.utc) - ultimo < timedelta(minutes=15):
            print("Usando dados salvos.")
            atualizar = False
    else:
        df = pd.DataFrame()

    if atualizar:
        print("Buscando dados atualizados da API...")
        api = conectar_api()
        df = buscar_candles(api, par)
        df.to_csv(caminho_csv, index=False)
        print(f"Dados salvos em {caminho_csv}.")

    return df

# Execução
if __name__ == "__main__":
    dados = carregar_dados("ETHUSD")
    print("\nÚltimos 5 candles:")
    print(dados.tail())
