
# Indicador Preditivo de Opções Binárias

## Visão Geral
Este projeto tem como objetivo construir um **indicador preditivo para o mercado financeiro de opções binárias**, integrando dados de uma corretora.  
O sistema consumirá dados históricos e em tempo real (via WebSocket), aplicará modelos de machine learning e deep learning, e gerará previsões sobre a direção do próximo candle.  
Na fase final, será implementada **semi-automação** e posteriormente **automação total** das operações, incluindo gerenciamento de ganhos.

---

## Objetivos do Projeto
- Coletar e processar dados históricos de candles da corretora.
- Criar modelos baseline e evoluir para redes neurais (LSTM).
- Implementar simulações automáticas com métricas de performance.
- Desenvolver visualizações interativas em tempo real.
- Integrar execução semi-automática de ordens.
- Implementar automação total com split de ganhos e relatórios completos.

---

## Estrutura do Repositório
```

indicador-preditivo/
│
├── data/                     # Histórico de candles e logs
│   ├── raw/                  # Dados crus
│   ├── cleaned/
│   │    └── transformed      # Dados pré-processados
│   └── logs/                 # Logs de execução
│
├── models/                   # Modelos preditivos treinados
│   ├── regressao.pkl
│   └── rede\_neural.pkl
│
├── scripts/                  # Scripts auxiliares
│   ├── preprocess.py         # Pré-processamento de dados
│   ├── train\_model.py        # Treino do modelo
│   ├── backtest.py           # Backtesting do modelo
│   └── utils.py              # Funções genéricas
│
├── realtime/                 # Parte em tempo real
│   ├── websocket\_client.py   # Conexão com a corretora (WebSocket)
│   ├── predictor.py          # Carrega modelo e gera previsões
│   ├── signal\_manager.py     # Lógica de CALL/PUT baseada nas previsões
│   └── order\_executor.py     # Executa ordens na corretora
│
├── dashboard/                # Visualização dos sinais
│   └── app.py                # Dashboard (Plotly/Dash ou Streamlit)
│
├── config/                   # Configurações
│   └── settings.yaml         # Credenciais, parâmetros do modelo, ativos
│
├── sprints/                  # Notebooks organizados por sprint
│   ├── sprint1\_notebooks/
│   ├── sprint2\_notebooks/
│   ├── sprint3\_notebooks/
│   ├── sprint4\_notebooks/
│   ├── sprint5\_notebooks/
│   └── sprint6\_notebooks/
│
├── main.py                   # Arquivo principal: orquestra tudo
├── backlog.csv               # Backlog do projeto (issues e milestones)
└── requirements.txt          # Dependências do projeto

```

---

## Sprints e Milestones
O backlog foi organizado em **6 sprints**, correspondendo aos milestones no GitHub:

1. **Sprint 1 – Setup Inicial e Dados Históricos**
   - Criar repositório e estrutura de pastas
   - Configurar ambiente Colab
   - Coletar dados históricos da corretora
   - Pré-processar dados  

2. **Sprint 2 – Baseline e Visualização**
   - Modelo baseline (Regressão Logística)
   - Visualização mínima de candles e previsão
   - Implementação de logs de simulação

3. **Sprint 3 – WebSocket e Modelo Robusto**
   - Integração WebSocket com a corretora
   - Treinar modelo XGBoost
   - Simulação semi-automática de ordens
   - Visualização interativa em tempo real

4. **Sprint 4 – Rede Neural LSTM**
   - Preparar dados sequenciais para LSTM
   - Treinar e avaliar LSTM
   - Testar LSTM em simulação e comparar com XGBoost

5. **Sprint 5 – Semi-Automação e Painel**
   - Implementar semi-automação de ordens
   - Criar painel de métricas em tempo real
   - Gerar relatórios básicos diários

6. **Sprint 6 – Automação Total e Versão Final**
   - Automação total de ordens
   - Implementar split de ganhos (MetaMask)
   - Gerar relatórios completos (diário, semanal, mensal)
   - Funcionalidades extras: backtesting, alertas, painel de saúde do modelo

---

## Tecnologias
- Python 3.10+
- Pandas, NumPy (manipulação de dados)
- Matplotlib, Plotly (visualização)
- Scikit-learn, XGBoost, TensorFlow/Keras (modelagem)
- WebSocket (corretora) para dados em tempo real
- GitHub Projects para gerenciamento do backlog

---

## Roadmap
- [x] Setup inicial do repositório e criação de milestones
- [x] Configuração de ambiente
- [ ] Coleta e pré-processamento de dados históricos
- [ ] Desenvolvimento de modelo baseline
- [ ] Integração com WebSocket e modelo XGBoost
- [ ] Implementação de simulação semi-automática
- [ ] Treinamento de LSTM e comparação de modelos
- [ ] Desenvolvimento de painel de métricas
- [ ] Automação total de ordens e split de ganhos
- [ ] Relatórios completos e funcionalidades extras

---

## Variáveis de Ambiente
O projeto utiliza um arquivo `.env` (não versionado) contendo:
```

GITHUB\_OWNER=\<usuário ou organização>
REPO=\<nome-do-repositório>
GITHUB\_TOKEN=\<token-com-permissões>

```

---

## Contribuição
Pull requests são bem-vindos. Para grandes alterações, abra uma issue antes para discussão.

---

## Licença
Este projeto é privado durante desenvolvimento. Licenciamento público será definido após versão estável.
