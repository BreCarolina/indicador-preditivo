#@title Script de treinamento do modelo LSTM (Regressão) ✅

"""
Treina LSTM para prever fechamento futuro (regressão) com alvo padronizado (Z-score).
- Entrada: X normalizado/padronizado por janela
- Alvo: y padronizado (fit no treino), despadronizado para métricas/plots
- Callbacks: EarlyStopping + Checkpoint (.keras)
- Relatório: loss (escala padronizada), MAE/RMSE/R² (em preço real)
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

# ==================== CONFIG ==================== #
PREPARED_DIR = "/content/indicador-preditivo/data/prepared"
MODELS_DIR   = "/content/indicador-preditivo/models"
os.makedirs(MODELS_DIR, exist_ok=True)

parametros = {
    "SEQ_LEN": 288,                # 1 dia em M5
    "features": 32,
    "unidades_lstm_camada1": 128,
    "unidades_lstm_camada2": 128,
    "unidades_dense": 64,
    "taxa_dropout": 0.01,
    "taxa_aprendizado": 0.0035,
    "paciencia": 10,
    "epocas_maximas": 100,
    "tamanho_lote": 128,
    "funcao_perda": "Huber",      # (robusta a outliers)
}

# ==================== LOAD DATA ==================== #
print(f"Carregando dataset mais recente de {PREPARED_DIR}...")

latest_X_train = sorted(glob.glob(os.path.join(PREPARED_DIR, "X_train_*.npy")))[-1]
latest_y_train = latest_X_train.replace("X_train", "y_train")
latest_X_test  = latest_X_train.replace("X_train", "X_test")
latest_y_test  = latest_X_train.replace("X_train", "y_test")

X_train = np.load(latest_X_train)
y_train = np.load(latest_y_train)
X_test  = np.load(latest_X_test)
y_test  = np.load(latest_y_test)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

# Padroniza y (fit no treino, aplica no teste)
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype(np.float32).ravel()
y_test_scaled  = y_scaler.transform(y_test.reshape(-1, 1)).astype(np.float32).ravel()

# ==================== MODEL ==================== #
print("[INFO] Construindo modelo...")

otimizador  = Adam(learning_rate=parametros["taxa_aprendizado"])
funcao_perda = Huber() if parametros["funcao_perda"].lower() == "huber" else "mse"

modelo = Sequential([
    tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(parametros["unidades_lstm_camada1"], return_sequences=True),
    Dropout(parametros["taxa_dropout"]),
    LSTM(parametros["unidades_lstm_camada2"], return_sequences=False),
    Dropout(parametros["taxa_dropout"]),
    Dense(parametros["unidades_dense"], activation="relu"),
    Dense(1)  # saída contínua
])
modelo.compile(optimizer=otimizador, loss=funcao_perda)
modelo.summary()

versao_dados = os.path.basename(latest_X_train).replace("X_train_", "").replace(".npy", "")
path_modelo  = os.path.join(MODELS_DIR, f"modelo_LSTM_seq{parametros['SEQ_LEN']}_{versao_dados}.keras")
path_scaler  = os.path.join(MODELS_DIR, f"y_scaler_{versao_dados}.pkl")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=parametros["paciencia"], restore_best_weights=True),
    ModelCheckpoint(path_modelo, monitor="val_loss", save_best_only=True, verbose=1)
]

# ==================== TRAIN ==================== #
print("[INFO] Iniciando treinamento...")
historico = modelo.fit(
    X_train, y_train_scaled,
    validation_data=(X_test, y_test_scaled),
    epochs=parametros["epocas_maximas"],
    batch_size=parametros["tamanho_lote"],
    callbacks=callbacks,
    verbose=1
)

# ==================== EVAL ==================== #
# Predições na escala padronizada e inversão para preço real
y_pred_scaled = modelo.predict(X_test, verbose=0).ravel()
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2   = r2_score(y_test, y_pred)

print("\n[✓] Métricas de validação (em preço real):")
print(f"MAE    = {mae:.6f}")
print(f"RMSE   = {rmse:.6f}")
print(f"R²     = {r2:.6f}")

# ==================== VISUALIZAÇÃO ==================== #
def plotar_predicoes(y_true, y_pred, n=200, titulo="Previsão vs Real"):
    plt.figure(figsize=(16,6))
    plt.plot(y_true[:n], label="Real", color="darkgreen")
    plt.plot(y_pred[:n], label="Previsto", color="darkorange", linestyle="--")
    plt.title(titulo)
    plt.xlabel("Amostras")
    plt.ylabel("Preço")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()

print("\n[INFO] Comparação das 10 primeiras previsões (preço real):")
for i in range(10):
    print(f"Obs {i+1:2d} | Real: {y_test[i]:.2f} | Previsto: {y_pred[i]:.2f}")

plotar_predicoes(y_test, y_pred)

# ==================== SAVE SCALER + REPORT ==================== #
joblib.dump(y_scaler, path_scaler)

relatorio_path = os.path.join(MODELS_DIR, "relatorio_modelos.csv")
params_iniciais = {
    "SEQ_LEN": parametros["SEQ_LEN"],
    "EPOCHS": parametros["epocas_maximas"],
    "BATCH_SIZE": parametros["tamanho_lote"]
}
params_finais = {
    "Loss(val)": float(min(historico.history["val_loss"])),
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2)
}

nova_linha = pd.DataFrame([{
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "versao_modelo": f"LSTM_seq{parametros['SEQ_LEN']}",
    "versao_dados": versao_dados,
    "checkpoint": os.path.basename(path_modelo),
    "y_scaler":    os.path.basename(path_scaler),
    "parametros_iniciais": str(params_iniciais),
    "parametros_finais":   str(params_finais),
    "loss_val": params_finais["Loss(val)"],
    "mae": params_finais["MAE"],
    "rmse": params_finais["RMSE"],
    "r2":  params_finais["R2"],
}])

if os.path.exists(relatorio_path):
    df_rel = pd.read_csv(relatorio_path)
    df_rel = pd.concat([df_rel, nova_linha], ignore_index=True)
else:
    df_rel = nova_linha

for col in ["loss_val", "mae", "rmse", "r2"]:
    df_rel[col] = pd.to_numeric(df_rel[col], errors="coerce")

df_rel = df_rel.sort_values(by="rmse", ascending=True).reset_index(drop=True).head(50)
df_rel.to_csv(relatorio_path, index=False)

print(f"[OK] Modelo salvo em: {path_modelo}")
print(f"[OK] y_scaler salvo em: {path_scaler}")
print(f"[OK] Relatório atualizado em {relatorio_path}")
