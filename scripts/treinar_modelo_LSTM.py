#@title Script para treinamento de modelo LSTM ✅

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

def treinar_modelo(
    prepared_dir="/content/indicador-preditivo/data/prepared",
    models_dir="/content/indicador-preditivo/models",
    parametros=None
):
    """
    Treina uma rede LSTM com os dados preparados.

    Args:
        prepared_dir (str): Diretório onde estão os .npy preparados.
        models_dir (str): Diretório onde salvar modelos treinados.
        parametros (dict): Hiperparâmetros do modelo.

    Returns:
        modelo, historico (obj): Modelo treinado e histórico do treinamento.
    """
    os.makedirs(models_dir, exist_ok=True)

    # Localizar o dataset mais recente
    arquivos = sorted(glob.glob(os.path.join(prepared_dir, "X_train_*.npy")))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo X_train_*.npy encontrado em {prepared_dir}. Execute preparar_dados primeiro.")

    latest_X_train = arquivos[-1]
    versao_dados = os.path.basename(latest_X_train).replace("X_train_", "").replace(".npy", "")

    latest_y_train = latest_X_train.replace("X_train", "y_train")
    latest_X_test  = latest_X_train.replace("X_train", "X_test")
    latest_y_test  = latest_X_train.replace("X_train", "y_test")

    X_train = np.load(latest_X_train)
    y_train = np.load(latest_y_train)
    X_test  = np.load(latest_X_test)
    y_test  = np.load(latest_y_test)

    print(f"Carregado dataset: {versao_dados}")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # Modelo
    if parametros is None:
        parametros = {
            "SEQ_LEN": X_train.shape[1],
            "features": X_train.shape[2],
            "unidades_lstm_camada1": 256,
            "unidades_lstm_camada2": 128,
            "unidades_dense": 64,
            "taxa_dropout": 0.01,
            "taxa_aprendizado": 0.003,
            "funcao_perda": "Huber",
            "paciencia": 10,
            "epocas_maximas": 100,
            "tamanho_lote": 128
        }

    otimizador = Adam(learning_rate=parametros["taxa_aprendizado"])
    funcao_perda = Huber()

    modelo = Sequential([
        LSTM(parametros["unidades_lstm_camada1"], return_sequences=True,
             input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(parametros["taxa_dropout"]),
        LSTM(parametros["unidades_lstm_camada2"], return_sequences=False),
        Dropout(parametros["taxa_dropout"]),
        Dense(parametros["unidades_dense"], activation="relu"),
        Dense(1)
    ])
    modelo.compile(optimizer=otimizador, loss=funcao_perda)
    modelo.summary()

    path_modelo = os.path.join(models_dir, f"modelo_LSTM_seq{parametros['SEQ_LEN']}_{versao_dados}.h5")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=parametros["paciencia"], restore_best_weights=True),
        ModelCheckpoint(path_modelo, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    print("[INFO] Iniciando treinamento...")
    historico = modelo.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=parametros["epocas_maximas"],
        batch_size=parametros["tamanho_lote"],
        callbacks=callbacks,
        verbose=1
    )

    # Avaliação
    y_pred = modelo.predict(X_test, verbose=0)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print("\n[✓] Métricas de validação:")
    print(f"MAE    = {mae:.6f}")
    print(f"RMSE   = {rmse:.6f}")
    print(f"R²     = {r2:.6f}")

    # Relatório
    relatorio_path = os.path.join(models_dir, "relatorio_modelos.csv")
    nova_linha = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "versao_modelo": f"LSTM_seq{parametros['SEQ_LEN']}",
        "versao_dados": versao_dados,
        "parametros": str(parametros),
        "loss": float(historico.history["loss"][-1]),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }])

    if os.path.exists(relatorio_path):
        df_rel = pd.read_csv(relatorio_path)
        df_rel = pd.concat([df_rel, nova_linha], ignore_index=True)
    else:
        df_rel = nova_linha

    df_rel.to_csv(relatorio_path, index=False)
    print(f"[OK] Relatório atualizado em {relatorio_path}")

    return modelo, historico
