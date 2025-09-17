#@title Script de Extração de dados IqOption✅ (v2)

import os
import time
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import sys

"""
Script: extrair_dados.py (v2)

Descrição:
-----------
Conecta-se à API da IQ Option e extrai candles históricos do par configurado,
sempre até o último candle fechado (sem puxar dados futuros).  

Fluxo de execução:
-------------------
1. Carrega credenciais da conta IQ Option a partir do arquivo .env.
2. Conecta à API da IQ Option.
3. Busca candles em blocos de até 1000, retrocedendo até atingir a quantidade desejada.
4. Converte os dados em DataFrame pandas, padroniza colunas e remove duplicados.
5. Descarta qualquer candle futuro (em aberto).
6. Salva os dados em CSV dentro de /data/raw.
7. Se já existir CSV anterior, mescla e mantém apenas candles únicos.

Parâmetros configuráveis:
--------------------------
PAR        -> Ativo a ser extraído (ex: "ETHUSD").
TIMEFRAME  -> Timeframe em segundos (ex: 300 = 5 minutos).
DIAS       -> Quantidade de dias históricos a coletar.
ROOT   -> Diretório raiz do projeto.

Saídas:
--------
- DataFrame pandas contendo candles ordenados.
- Arquivo CSV salvo em: {ROOT_DIR}/data/raw/{PAR}_M{TIMEFRAME//60}_{DIAS}d.csv
"""


#--------------------Configurações-------------------------#

# Carrega credenciais do arquivo .env
load_dotenv("/content/drive/MyDrive/Projetos/indicador-preditivo/.env")
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")

# Caminho da API da IQ Option
sys.path.append("/content/indicador-preditivo/iqoptionaapi")
from stable_api import IQ_Option

# Parâmetros padrão
PAR = "ETHUSD"
TIMEFRAME = 300   # 5 minutos
DIAS = 120
ROOT = "/content/indicador-preditivo"



# Cálculo de quantos candles coletar
candles_por_dia = int((24 * 60) / (TIMEFRAME / 60))
TOTAL_CANDLES = DIAS * candles_por_dia

def conectar_api():
    # Conecta à IQ Option usando credenciais
    api = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
    conectado, _ = api.connect()
    if not conectado:
        raise SystemExit("Erro: não foi possível conectar à IQ Option.")
    print("Conectado com sucesso.")
    return api


def buscar_candles(api, par=PAR, timeframe=TIMEFRAME, total_candles=TOTAL_CANDLES):
    # Busca candles em blocos de até 1000, alinhados ao último candle fechado
    max_por_chamada = 1000
    todos_candles = []

    # timestamp atual arredondado para baixo no múltiplo do timeframe
    agora = int(time.time())
    agora -= agora % timeframe

    while total_candles > 0:
        qtd = min(max_por_chamada, total_candles)
        candles = api.get_candles(par, timeframe, qtd, agora)

        if not candles:
            print("Aviso: não retornaram candles nesta chamada.")
            break

        todos_candles.extend(candles)
        agora = candles[0]["from"] - 1  # retrocede para evitar sobreposição
        total_candles -= qtd

    # Cria DataFrame com candles
    df = pd.DataFrame(todos_candles)
    df["from"] = pd.to_datetime(df["from"], unit="s", utc=True)

    # Padroniza nomes das colunas
    df = df.rename(columns={
        "open": "abertura",
        "max": "maxima",
        "min": "minima",
        "close": "fechamento"
    })
    df = df[["from", "abertura", "maxima", "minima", "fechamento", "volume"]]

    # Remove duplicados e ordena
    df = df.drop_duplicates(subset=["from"]).sort_values("from").reset_index(drop=True)

    # Descarta candles futuros (em aberto)
    agora_ts = pd.Timestamp.now(tz="UTC")  # já retorna tz-aware
    antes = len(df)
    df = df[df["from"] + pd.to_timedelta(timeframe, unit="s") <= agora_ts]
    descartados = antes - len(df)

    if descartados > 0:
        print(f"[INFO] {descartados} candles futuros descartados.")

    return df


def extrair_dados(par=PAR, timeframe=TIMEFRAME, dias=DIAS, root=ROOT):
    # Define diretório e nome do CSV
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    caminho_csv = os.path.join(raw_dir, f"{par}_M{timeframe//60}_{dias}d.csv")

    print(f"Baixando {dias} dias de dados ({TOTAL_CANDLES} candles em M{timeframe//60})...")

    # Conecta e busca dados
    api = conectar_api()
    df = buscar_candles(api, par, timeframe, TOTAL_CANDLES)

    # Se já existir CSV, mescla com novos dados
    if os.path.exists(caminho_csv):
        antigo = pd.read_csv(caminho_csv, parse_dates=["from"])
        df = pd.concat([antigo, df]).drop_duplicates(subset=["from"]).sort_values("from").reset_index(drop=True)

    # Salva CSV final
    df.to_csv(caminho_csv, index=False)
    print(f"Dados salvos em {caminho_csv} (até {df['from'].max()}).")

    return df, caminho_csv


if __name__ == "__main__":
    dados, path = extrair_dados(PAR, TIMEFRAME, DIAS, ROOT)
    print(dados.tail())
    print(dados.head())
