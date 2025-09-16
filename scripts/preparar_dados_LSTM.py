#@title Script de preparação de dados para LSTM ✅

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler

def preparar_dados(transformed_path=None, seq_len=300, test_size=0.15, root="/content/indicador-preditivo"):
    """
    Prepara dados sequenciais para treino da LSTM.

    Args:
        transformed_path (str, opcional): Caminho para o CSV transformado. 
                                          Se None, usa o mais recente da pasta /data/transformed.
        seq_len (int): Tamanho da janela (lookback).
        test_size (float): Proporção reservada para teste.
        root (str): Caminho raiz do projeto.

    Returns:
        dict: Contendo nomes dos arquivos salvos (X_train, y_train, X_test, y_test).
    """
    TRANSFORMED_DIR = os.path.join(root, "data", "transformed")
    PREPARED_DIR = os.path.join(root, "data", "prepared")
    os.makedirs(PREPARED_DIR, exist_ok=True)

    # Se não passar caminho, pega o mais recente
    if transformed_path is None:
        transformed_files = sorted(glob.glob(os.path.join(TRANSFORMED_DIR, "ETHUSD_*_transformed_*.csv")))
        if not transformed_files:
            raise FileNotFoundError("Nenhum arquivo encontrado em TRANSFORMED_DIR.")
        transformed_path = transformed_files[-1]

    print(f"[INFO] Carregando arquivo: {transformed_path}")
    df = pd.read_csv(transformed_path, parse_dates=["timestamp"])

    # --------------------------
    # Definição de features
    # --------------------------
    features_normalizar = [
        "abertura", "maxima", "minima", "fechamento",
        "pressao_compradora", "pressao_vendedora",
        "var_fechamento",
        "resistencia", "suporte", "dist_resistencia", "dist_suporte",
    ] + [c for c in df.columns if c.startswith("SMA_") or c.startswith("EMA_")]

    features_padronizar = ["volume", "vol_media_5", "vol_media_20", "retorno", "volatilidade"]
    features_nao_normalizar = ["RSI_14"]
    features_escala_fixa = ["hora_num", "minuto", "dia_semana"]

    # Target: valor de fechamento futuro
    target = "fechamento_futuro"
    all_features = features_normalizar + features_padronizar + features_nao_normalizar + features_escala_fixa

    # Escalas fixas
    df["hora_num"] = df["hora_num"] / 23.0
    df["minuto"] = df["minuto"] / 59.0
    df["dia_semana"] = df["dia_semana"] / 6.0

    # --------------------------
    # Criação de sequências
    # --------------------------
    X, y = [], []
    print("[INFO] Criando sequências deslizantes...")
    for i in range(len(df) - seq_len):
        seq = df.iloc[i:i+seq_len]
        target_val = df.iloc[i+seq_len][target]

        seq_array = seq[all_features].values.astype(np.float32)
        seq_norm = seq_array.copy()

        for idx, col in enumerate(all_features):
            if col in features_normalizar:
                col_min, col_max = seq_array[:, idx].min(), seq_array[:, idx].max()
                seq_norm[:, idx] = (seq_array[:, idx] - col_min) / (col_max - col_min + 1e-8)
            elif col in features_padronizar:
                scaler = StandardScaler()
                seq_norm[:, idx] = scaler.fit_transform(seq_array[:, idx].reshape(-1, 1)).flatten()
            else:
                seq_norm[:, idx] = seq_array[:, idx]

        X.append(seq_norm)
        y.append(target_val)

    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    print(f"[INFO] Formato final -> X: {X.shape}, y: {y.shape}")

    # --------------------------
    # Split treino/teste
    # --------------------------
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"[INFO] Split -> Treino: {X_train.shape}, Teste: {X_test.shape}")

    # --------------------------
    # Salvamento
    # --------------------------
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    np.save(os.path.join(PREPARED_DIR, f"X_train_{ts}.npy"), X_train)
    np.save(os.path.join(PREPARED_DIR, f"y_train_{ts}.npy"), y_train)
    np.save(os.path.join(PREPARED_DIR, f"X_test_{ts}.npy"), X_test)
    np.save(os.path.join(PREPARED_DIR, f"y_test_{ts}.npy"), y_test)

    print(f"[OK] Dados preparados salvos em {PREPARED_DIR}")
    return {
        "X_train": f"X_train_{ts}.npy",
        "y_train": f"y_train_{ts}.npy",
        "X_test": f"X_test_{ts}.npy",
        "y_test": f"y_test_{ts}.npy"
    }

if __name__ == "__main__":
    preparar_dados()
