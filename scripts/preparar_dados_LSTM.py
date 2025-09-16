#@title Script de preparação de dados para LSTM ✅

"""
Script de preparação de dados para LSTM (Regressão).

Fluxo do pipeline:
1. Carregar o CSV mais recente da pasta transformed.
2. Selecionar features e definir target como fechamento futuro (valor contínuo).
3. Tratar valores ausentes e organizar colunas.
4. Criar sequências deslizantes (lookback = SEQ_LEN).
5. Normalizar ou padronizar cada feature de acordo com sua natureza:
   - Normalização por janela (0–1): preços, SMAs, EMAs, suporte, resistência,
     distâncias, pressões, variação do fechamento.
   - Padronização (Z-score): volume, médias de volume, retorno, volatilidade.
   - Sem transformação: RSI.
   - Escala fixa: hora/23, minuto/59, dia_semana/6.
6. Dividir os dados em treino e teste respeitando a ordem temporal.
7. Salvar X_train, y_train, X_test, y_test em arquivos .npy na pasta prepared.
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler

# Configurações principais
ROOT = "/content/indicador-preditivo"
TRANSFORMED_DIR = os.path.join(ROOT, "data", "transformed")
PREPARED_DIR = os.path.join(ROOT, "data", "prepared")
os.makedirs(PREPARED_DIR, exist_ok=True)

SEQ_LEN = 288     # tamanho da janela (lookback) (Coloquei 1dia)
TEST_SIZE = 0.15   # proporção dos dados para teste (15%)

# Localizar o arquivo transformado mais recente
transformed_files = sorted(glob.glob(os.path.join(TRANSFORMED_DIR, "ETHUSD_*_transformed_*.csv")))
if not transformed_files:
    raise FileNotFoundError("Nenhum arquivo encontrado em TRANSFORMED_DIR.")

latest_file = transformed_files[-1]
print(f"[INFO] Carregando arquivo: {latest_file}")
df = pd.read_csv(latest_file, parse_dates=["timestamp"])

# Definição de grupos de features
features_normalizar = [
    "abertura", "maxima", "minima", "fechamento",
    "pressao_compradora", "pressao_vendedora",
    "var_fechamento",
    "resistencia", "suporte", "dist_resistencia", "dist_suporte",
] + [c for c in df.columns if c.startswith("SMA_") or c.startswith("EMA_")]

features_padronizar = [
    "volume", "vol_media_5", "vol_media_20", "retorno", "volatilidade"
]

features_nao_normalizar = [
    "RSI_14"  # já limitado em [0,100]
]

features_escala_fixa = [
    "hora_num", "minuto", "dia_semana"
]

target = "fechamento_futuro"

# Criar coluna de target contínuo (regressão)
df[target] = df["fechamento"].shift(-1)
df = df.dropna().reset_index(drop=True)

all_features = features_normalizar + features_padronizar + features_nao_normalizar + features_escala_fixa

# Aplicar escalas fixas diretamente (0–1)
df["hora_num"] = df["hora_num"] / 23.0
df["minuto"] = df["minuto"] / 59.0
df["dia_semana"] = df["dia_semana"] / 6.0

# Criação das sequências deslizantes
X, y = [], []
print("[INFO] Criando sequências deslizantes...")
for i in range(len(df) - SEQ_LEN):
    # Seleciona a janela de tamanho SEQ_LEN
    seq = df.iloc[i:i+SEQ_LEN]
    # Seleciona o valor alvo correspondente (após a janela)
    target_val = df.iloc[i+SEQ_LEN][target]

    # Converte a janela para array NumPy
    seq_array = seq[all_features].values.astype(np.float32)
    seq_norm = seq_array.copy()

    # Normalização/padronização coluna a coluna
    for idx, col in enumerate(all_features):
        if col in features_normalizar:
            col_min = seq_array[:, idx].min()
            col_max = seq_array[:, idx].max()
            seq_norm[:, idx] = (seq_array[:, idx] - col_min) / (col_max - col_min + 1e-8)

        elif col in features_padronizar:
            scaler = StandardScaler()
            seq_norm[:, idx] = scaler.fit_transform(seq_array[:, idx].reshape(-1, 1)).flatten()

        elif col in features_nao_normalizar or col in features_escala_fixa:
            seq_norm[:, idx] = seq_array[:, idx]  # mantém os valores originais

    X.append(seq_norm)
    y.append(target_val)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)  # agora é regressão, valores contínuos

print(f"[INFO] Formato final -> X: {X.shape}, y: {y.shape}")

# Divisão treino/teste mantendo ordem temporal
split_idx = int(len(X) * (1 - TEST_SIZE))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"[INFO] Split -> Treino: {X_train.shape}, Teste: {X_test.shape}")

# Salvamento dos conjuntos de dados
ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
np.save(os.path.join(PREPARED_DIR, f"X_train_{ts}.npy"), X_train)
np.save(os.path.join(PREPARED_DIR, f"y_train_{ts}.npy"), y_train)
np.save(os.path.join(PREPARED_DIR, f"X_test_{ts}.npy"), X_test)
np.save(os.path.join(PREPARED_DIR, f"y_test_{ts}.npy"), y_test)

print(f"[OK] Dados preparados salvos em {PREPARED_DIR}")
