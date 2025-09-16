#@title README.md

# Indicador Preditivo de Opções Binárias

## Visão Geral
Este projeto tem como objetivo construir um **indicador preditivo para o mercado financeiro de opções binárias**, utilizando dados históricos e em tempo real da corretora (via WebSocket).
O sistema implementa um pipeline completo de **coleta, transformação, preparação e modelagem sequencial com LSTM**, para prever o fechamento futuro dos candles (regressão).

Na fase final, será implementada **semi-automação** e, posteriormente, **automação total** das operações, incluindo gerenciamento de ganhos.


## Objetivos do Projeto
- Coletar e armazenar candles históricos da corretora.
- Transformar dados brutos em features técnicas e estatísticas.
- Preparar sequências temporais normalizadas/padronizadas para LSTM.
- Treinar, avaliar e versionar modelos preditivos.
- Registrar métricas (MAE, RMSE, R²) em relatórios de modelos.
- Desenvolver visualizações de comparações previsão vs. real.
- Evoluir para semi-automação e automação total das operações.


## Estrutura do Repositório

indicador-preditivo/
│
├── data/
│   ├── raw/                  # Dados crus (candles coletados)
│   ├── cleaned/              # Dados limpos
│   ├── transformed/          # Dados transformados com features
│   ├── prepared/             # Dados preparados em sequências para LSTM
│   └── logs/                 # Logs de execução
│
├── models/                   # Modelos e relatórios
│   ├── modelo\_LSTM\_seq\*.keras
│   ├── y\_scaler\_\*.pkl
│   └── relatorio\_modelos.csv
│
├── scripts/                  # Scripts principais do pipeline
│   ├── coletar\_dados.py
│   ├── transformar\_dados.py
│   ├── preparar\_dados\_LSTM.py
│   └── treinar\_modelo\_LSTM.py
│
├── realtime/                 # Integração em tempo real (futuro)
│   ├── websocket\_client.py
│   ├── predictor.py
│   ├── signal\_manager.py
│   └── order\_executor.py
│
├── dashboard/                # Visualização dos sinais e métricas
│   └── app.py
│
├── config/                   # Configurações e credenciais
│   └── settings.yaml
│
├── sprints/                  # Notebooks organizados por sprint
│
├── main.py                   # Arquivo principal (orquestração)
├── backlog.csv               # Backlog do projeto
└── requirements.txt          # Dependências do projeto

---

## Sprints e Milestones
O projeto está organizado em **6 sprints**:

1. **Sprint 1 – Setup Inicial e Dados Históricos**
   - Estruturação do repositório
   - Coleta de candles
   - Pré-processamento inicial

2. **Sprint 2 – Transformação e Features**
   - Criação de features técnicas/estatísticas
   - Validação de consistência
   - Logs estruturados

3. **Sprint 3 – Preparação de Dados**
   - Normalização/padronização por sequência
   - Divisão treino/teste temporal
   - Criação da pasta `prepared/`

4. **Sprint 4 – Modelo LSTM**
   - Prototipagem e treinamento da LSTM
   - Ajustes de hiperparâmetros
   - Salvamento de modelos e `y_scaler`

5. **Sprint 5 – Semi-Automação e Dashboard**
   - Painel em tempo real
   - Integração de sinais com corretora
   - Relatórios de simulação

6. **Sprint 6 – Automação Total**
   - Execução automática de ordens
   - Split de ganhos (MetaMask)
   - Relatórios completos (diário, semanal, mensal)

---

## Tecnologias
- Python 3.10+
- Pandas, NumPy (manipulação de dados)
- Matplotlib, Plotly (visualização)
- Scikit-learn, TensorFlow/Keras (modelagem)
- WebSocket (dados em tempo real)
- GitHub Projects (gerenciamento de backlog)

---

## Roadmap
- [x] Setup inicial do repositório
- [x] Coleta de dados brutos
- [x] Transformação e criação de features
- [x] Preparação de sequências para LSTM
- [x] Treinamento inicial de LSTM
- [ ] Ajustes avançados de hiperparâmetros
- [ ] Desenvolvimento de dashboard
- [ ] Integração com corretora em tempo real
- [ ] Semi-automação de ordens
- [ ] Automação total com relatórios

---

## Variáveis de Ambiente
O projeto utiliza um arquivo `.env` (não versionado) com credenciais da corretora e do GitHub.

Exemplo:

IQ\_EMAIL=<email>
IQ\_PASSWORD=<senha>
GITHUB\_OWNER=\<usuário>
REPO=<nome-repo>
GITHUB\_TOKEN=<token>

---

## Contribuição
Pull requests são bem-vindos. Abra uma issue para discussão antes de mudanças grandes.

---

## Licença
Projeto privado durante desenvolvimento. Licenciamento público será definido na versão estável.

