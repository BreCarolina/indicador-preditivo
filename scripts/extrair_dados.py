#@title Script de Extração de dados IqOption✅
"""
Script: extrair_dados.py

Descrição:
-----------
Este script conecta-se à API da IQ Option e extrai candles históricos do par configurado,
coletando sempre até o minuto atual. Os dados são organizados em um DataFrame pandas e
armazenados em CSV na pasta /data/raw dentro do projeto.

Fluxo de execução:
-------------------
1. Carrega credenciais da conta IQ Option a partir do arquivo .env.
2. Conecta à API da IQ Option.
3. Busca candles em blocos de até 1000, retrocedendo até atingir a quantidade desejada.
4. Converte os dados em DataFrame, padronizando nomes de colunas (abertura, maxima, minima, fechamento, volume).
5. Remove duplicados, organiza em ordem cronológica e salva em CSV.
6. Se já existir um CSV anterior, mescla dados antigos com novos e mantém apenas candles únicos.

Parâmetros configuráveis:
--------------------------
PAR        -> Ativo a ser extraído (ex: "ETHUSD").
TIMEFRAME  -> Timeframe em segundos (ex: 300 = 5 minutos).
DIAS       -> Quantidade de dias históricos a coletar.
ROOT_DIR   -> Diretório raiz do projeto onde será criada a pasta data/raw.

Saídas:
--------
- DataFrame pandas contendo os candles ordenados do mais antigo ao mais recente.
- Arquivo CSV salvo em: {ROOT_DIR}/data/raw/{PAR}_M{TIMEFRAME//60}_{DIAS}d.csv

Integração:
------------
Este script foi criado para ser importando em main.py e usado via a função
`extrair_dados()`, que retorna o DataFrame e o caminho do CSV.
"""

import os
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import sys

# Carregar credenciais do arquivo .env
load_dotenv("/content/drive/MyDrive/Projetos/indicador-preditivo/.env")
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")

# Caminho da API da IQ Option
sys.path.append("/content/indicador-preditivo/iqoptionaapi")
from stable_api import IQ_Option

# Parâmetros configuráveis
PAR = "ETHUSD"        # Ativo
TIMEFRAME = 300       # Timeframe em segundos (ex: 300 = 5 minutos)
DIAS = 30             # Quantos dias de histórico buscar
ROOT_DIR = "/content/indicador-preditivo"

# Cálculo do total de candles
candles_por_dia = int((24 * 60) / (TIMEFRAME / 60))
TOTAL_CANDLES = DIAS * candles_por_dia

def conectar_api():
    """
    Conecta à API da IQ Option usando as credenciais fornecidas.
    Retorna o objeto de conexão ativo.
    """
    api = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
    conectado, _ = api.connect()
    if not conectado:
        raise SystemExit("Erro: não foi possível conectar à IQ Option.")
    print("Conectado com sucesso.")
    return api

def buscar_candles(api, par=PAR, timeframe=TIMEFRAME, total_candles=TOTAL_CANDLES):
    """
    Busca candles do ativo especificado, em blocos de até 1000 candles,
    garantindo que os dados cheguem até o minuto atual.
    Retorna um DataFrame com as colunas: from, abertura, maxima, minima, fechamento, volume.
    """
    max_por_chamada = 1000
    todos_candles = []
    agora = int(time.time())  # timestamp atual em segundos UTC

    while total_candles > 0:
        qtd = min(max_por_chamada, total_candles)
        candles = api.get_candles(par, timeframe, qtd, agora)

        if not candles:
            print("Aviso: não retornaram candles nesta chamada.")
            break

        todos_candles.extend(candles)
        agora = candles[0]["from"] - 1  # retrocede para evitar sobreposição
        total_candles -= qtd

    df = pd.DataFrame(todos_candles)
    df["from"] = pd.to_datetime(df["from"], unit="s", utc=True)
    df = df.rename(columns={
        "open": "abertura",
        "max": "maxima",
        "min": "minima",
        "close": "fechamento"
    })

    df = df[["from", "abertura", "maxima", "minima", "fechamento", "volume"]]
    df = df.drop_duplicates(subset=["from"]).sort_values("from").reset_index(drop=True)

    return df

def extrair_dados(par=PAR, timeframe=TIMEFRAME, dias=DIAS, root=ROOT_DIR):
    """
    Extrai candles da IQ Option e salva em CSV.
    Se já existir arquivo, atualiza com novos dados até o minuto atual.
    Retorna o DataFrame final e o caminho do CSV salvo.
    """
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    caminho_csv = os.path.join(raw_dir, f"{par}_M{timeframe//60}_{dias}d.csv")

    print(f"Baixando {dias} dias de dados ({TOTAL_CANDLES} candles em M{timeframe//60})...")

    api = conectar_api()
    df = buscar_candles(api, par, timeframe, TOTAL_CANDLES)

    if os.path.exists(caminho_csv):
        antigo = pd.read_csv(caminho_csv, parse_dates=["from"])
        df = pd.concat([antigo, df]).drop_duplicates(subset=["from"]).sort_values("from").reset_index(drop=True)

    df.to_csv(caminho_csv, index=False)
    print(f"Dados salvos em {caminho_csv} (até {df['from'].max()}).")

    return df, caminho_csv

if __name__ == "__main__":
    dados, path = extrair_dados(PAR, TIMEFRAME, DIAS, ROOT_DIR)
    print(dados.tail())
    print(dados.head())
