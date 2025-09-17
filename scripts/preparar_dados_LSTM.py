
"""
Script: preparar_dados_LSTM.py

Descrição:
-----------
Prepara dados transformados em janelas temporais para treino de modelos LSTM.
Normaliza/padroniza features por janela, escalona o target com StandardScaler
ajustado apenas no treino, e salva X/y em .npy e o scaler do target em .npz.
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler

# Carrega o CSV transformado mais recente ou o caminho informado
def carregar_csv(transformed_path, transformed_dir):    
    if transformed_path is None:
        arquivos = sorted(glob.glob(os.path.join(transformed_dir, "ETHUSD_*_transformed_*.csv")))
        if not arquivos:
            raise FileNotFoundError("Nenhum arquivo transformado encontrado.")
        transformed_path = arquivos[-1]

    df = pd.read_csv(transformed_path, parse_dates=["timestamp"])

    # Garante que a coluna de target exista
    if "fechamento_futuro" not in df.columns:
        raise RuntimeError("Coluna 'fechamento_futuro' não encontrada. Rode transformar_dados antes.")

    # Remove linhas sem target (última linha do shift)
    return df.dropna(subset=["fechamento_futuro"]).reset_index(drop=True)


# Normaliza ou padroniza cada feature dentro de uma sequência
def normalizar_seq(seq_array, features, features_norm, features_std):
        seq_norm = seq_array.copy()
    for idx, col in enumerate(features):
        col_vals = seq_array[:, idx]

        # Normalização min-max dentro da janela
        if col in features_norm:
            col_min, col_max = col_vals.min(), col_vals.max()
            denom = col_max - col_min if col_max > col_min else 1.0
            seq_norm[:, idx] = (col_vals - col_min) / denom

        # Padronização z-score (média 0, desvio 1)
        elif col in features_std:
            scaler = StandardScaler()
            seq_norm[:, idx] = scaler.fit_transform(col_vals.reshape(-1, 1)).flatten()

    return seq_norm

# Cria janelas deslizantes (X) e o target correspondente (y)
def criar_sequencias(df, seq_len, features, features_norm, features_std, target):    
    X, y = [], []
    descartadas = 0
    limite = len(df) - seq_len  # garante apenas janelas completas

    for i in range(limite):
        seq = df.iloc[i:i+seq_len][features].values.astype(np.float32)
        target_val = df.iloc[i+seq_len][target]

        # Descarta se houver valores inválidos
        if not np.isfinite(seq).all() or not np.isfinite(target_val):
            descartadas += 1
            continue

        # Normaliza a sequência
        seq_norm = normalizar_seq(seq, features, features_norm, features_std)

        # Descarta se após normalização ainda houver valores inválidos
        if not np.isfinite(seq_norm).all():
            descartadas += 1
            continue

        X.append(seq_norm)
        y.append(target_val)

    print(f"[INFO] Sequências criadas: {len(X)} | Descartadas: {descartadas}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# Define diretórios de entrada (transformados) e saída (preparados)
def preparar_dados(transformed_path=None, seq_len=300, test_size=0.15, root="/content/indicador-preditivo"):  
    transformed_dir = os.path.join(root, "data", "transformed")
    prepared_dir = os.path.join(root, "data", "prepared")
    os.makedirs(prepared_dir, exist_ok=True)

    # Carrega o CSV transformado
    df = carregar_csv(transformed_path, transformed_dir)

    # Define os grupos de features
    features_norm = [
        "abertura", "maxima", "minima", "fechamento",
        "pressao_compradora", "pressao_vendedora", "var_fechamento",
        "resistencia", "suporte", "dist_resistencia", "dist_suporte",
    ] + [c for c in df.columns if c.startswith(("SMA_", "EMA_"))]

    features_std = ["volume", "vol_media_5", "vol_media_20", "retorno", "volatilidade"]
    features_keep = ["RSI_14"]  # já está em escala padronizada
    features_fixed = ["hora_num", "minuto", "dia_semana"]  # escalas fixas

    # Normaliza as variáveis temporais entre 0 e 1
    df["hora_num"] /= 23.0
    df["minuto"] /= 59.0
    df["dia_semana"] /= 6.0

    # Lista final de features e definição do target
    features = features_norm + features_std + features_keep + features_fixed
    target = "fechamento_futuro"

    # Cria sequências para treino/teste
    X, y = criar_sequencias(df, seq_len, features, features_norm, features_std, target)

    # Split treino/teste preservando ordem temporal
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train_raw, y_test_raw = y[:split_idx], y[split_idx:]

    # Escalonamento do target (fit no treino, transform no teste)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).astype(np.float32).flatten()
    y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).astype(np.float32).flatten()

    # Salva datasets e scaler com timestamp
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    np.save(os.path.join(prepared_dir, f"X_train_{ts}.npy"), X_train)
    np.save(os.path.join(prepared_dir, f"y_train_{ts}.npy"), y_train)
    np.save(os.path.join(prepared_dir, f"X_test_{ts}.npy"), X_test)
    np.save(os.path.join(prepared_dir, f"y_test_{ts}.npy"), y_test)
    np.savez(os.path.join(prepared_dir, f"y_scaler_{ts}.npz"),
             mean=y_scaler.mean_.astype(np.float32), scale=y_scaler.scale_.astype(np.float32))

    print(f"[OK] Dados preparados salvos em {prepared_dir}")
    return {
        "X_train": f"X_train_{ts}.npy",
        "y_train": f"y_train_{ts}.npy",
        "X_test": f"X_test_{ts}.npy",
        "y_test": f"y_test_{ts}.npy",
        "y_scaler": f"y_scaler_{ts}.npz"
    }


# Carrega os arquivos recém-criados e imprime estatísticas resumidas
def inspecionar_datasets(prepared_dir, arquivos):    
    def resumo(arr):
        return {
            "shape": arr.shape,
            "dtype": arr.dtype,
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "nan?": bool(np.isnan(arr).any()),
            "inf?": bool(np.isinf(arr).any())
        }

    X_train = np.load(os.path.join(prepared_dir, arquivos["X_train"]))
    y_train = np.load(os.path.join(prepared_dir, arquivos["y_train"]))
    X_test = np.load(os.path.join(prepared_dir, arquivos["X_test"]))
    y_test = np.load(os.path.join(prepared_dir, arquivos["y_test"]))
    s = np.load(os.path.join(prepared_dir, arquivos["y_scaler"]))

    print("\n[INSPEÇÃO] Estatísticas dos datasets:")
    print("X_train:", resumo(X_train))
    print("y_train:", resumo(y_train))
    print("X_test :", resumo(X_test))
    print("y_test :", resumo(y_test))
    print(f"Scaler -> mean={float(s['mean'])}, scale={float(s['scale'])}")


if __name__ == "__main__":
    arquivos = preparar_dados()
    prepared_dir = "/content/indicador-preditivo/data/prepared"
    inspecionar_datasets(prepared_dir, arquivos)
