# Main do Indicador Preditivo ✅

import os
from datetime import datetime

# ------------------------
# PARÂMETROS GLOBAIS
# ------------------------
PARAMS = {
    "ROOT": "/content/indicador-preditivo",
    "PAR": "ETHUSD",
    "TIMEFRAME": 300,   # em segundos (M5 = 300, M1 = 60, M15 = 900)
    "DIAS": 30,         # dias de histórico para coletar
    "SEQ_LEN": 288,     # tamanho da sequência (lookback)
    "TEST_SIZE": 0.15,  # % para teste
    "MODELO": {
        "unidades_lstm_camada1": 128,
        "unidades_lstm_camada2": 64,
        "unidades_dense": 64,
        "taxa_dropout": 0.01,
        "taxa_aprendizado": 0.0035,
        "funcao_perda": "Huber",
        "epocas_maximas": 100,
        "tamanho_lote": 128,
        "paciencia": 10
    }
}

# ------------------------
# IMPORTS DOS SCRIPTS
# ------------------------
import sys
sys.path.append("/content/indicador-preditivo")

from scripts.extrair_dados import extrair_dados
from scripts.transformar_dados import transformar_dados
from scripts.preparar_dados_LSTM import preparar_dados
from scripts.treinar_modelo_LSTM import treinar_modelo

# ------------------------
# EXECUÇÃO DO PIPELINE
# ------------------------
def main():
    print("\n=== Indicador Preditivo - Pipeline Completo ===")

    # Etapa 1: Extração de dados
    raw_path = extrair_dados(
        par=PARAMS["PAR"],
        timeframe=PARAMS["TIMEFRAME"],
        dias=PARAMS["DIAS"],
        root=PARAMS["ROOT"]
    )
    print(f"[OK] Dados brutos salvos em: {raw_path}")

    # Etapa 2: Transformação e features
    transformed_path = transformar_dados(
        raw_path=raw_path,
        par=PARAMS["PAR"],
        timeframe=PARAMS["TIMEFRAME"],
        dias=PARAMS["DIAS"],
        root=PARAMS["ROOT"]
    )
    print(f"[OK] Dados transformados salvos em: {transformed_path}")

    # Etapa 3: Preparação (sequências LSTM)
    prepared_paths = preparar_dados(
        transformed_path=transformed_path,
        seq_len=PARAMS["SEQ_LEN"],
        test_size=PARAMS["TEST_SIZE"],
        root=PARAMS["ROOT"]
    )
    print(f"[OK] Dados preparados salvos em: {prepared_paths}")

    # Etapa 4: Treinamento do modelo
    modelo, historico = treinar_modelo(
        prepared_dir=os.path.join(PARAMS["ROOT"], "data", "prepared"),
        models_dir=os.path.join(PARAMS["ROOT"], "models"),
        parametros=PARAMS["MODELO"]
    )
    print("[OK] Treinamento concluído.")


if __name__ == "__main__":
    main()
