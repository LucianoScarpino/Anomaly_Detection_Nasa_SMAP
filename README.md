# Anomaly Detection in Spacecraft Telemetry

Progetto di **Geometric Learning — Politecnico di Torino** dedicato al rilevamento di anomalie nelle serie temporali multivariate del dataset NASA SMAP.

Il progetto confronta sei modelli, dalle baseline classiche alle reti neurali ricorrenti, per studiare due approcci: **prevedere il comportamento futuro** del segnale oppure **ricostruire il comportamento osservato**. Gli errori vengono trasformati in segnalazioni di anomalie attraverso una pipeline comune di smoothing e soglie dinamiche.

**Notebook principale:** [Anomaly_Detection_V5.ipynb](Anomaly_Detection_V5.ipynb)

## Dataset

Il notebook utilizza la telemetria operativa del satellite **SMAP (Soil Moisture Active Passive)**, distribuita con il progetto [Telemanom](https://github.com/khundman/telemanom#data). Il dataset comprende segnali di telemetria, informazioni sui comandi e intervalli di anomalie annotati.

Nell'esecuzione salvata nel notebook vengono analizzati **54 canali di telemetria**, con **25 feature per canale** e **68 sequenze anomale valutate**. Questi conteggi descrivono il sottoinsieme effettivamente utilizzato nel progetto.

Ogni canale è rappresentato da due file `.npy`, uno per il training e uno per il test, con forma `(timesteps, 25)`. La feature `0` è il segnale principale da monitorare; le altre feature forniscono il contesto operativo. Il file `labeled_anomalies.csv` contiene gli identificativi dei canali, la missione e gli intervalli anomali nel test set.

Il notebook seleziona le righe con `spacecraft == "SMAP"`. I dati vengono utilizzati nella scala fornita, senza ulteriore standardizzazione, e le feature costanti vengono mantenute.

## Modelli confrontati

Ogni modello viene addestrato separatamente per ciascun canale.

| Modello | Approccio | Configurazione principale |
| :--- | :--- | :--- |
| PCA | Ricostruzione lineare | Componenti che spiegano il 95% della varianza |
| Random Forest Regressor | Forecasting | 100 alberi su finestre appiattite |
| MLP Regressor | Forecasting | Strati nascosti da 128 e 64 unità; dropout 0,2 |
| MLP Autoencoder (AE) | Ricostruzione | Encoder-decoder denso; spazio latente di 32 dimensioni |
| LSTM Regressor | Forecasting | Due strati LSTM da 80 unità; dropout 0,3 |
| LSTM Autoencoder | Ricostruzione | Hidden size 64; spazio latente di 32 dimensioni |

## Pipeline

```mermaid
flowchart LR
    A[Telemetria multivariata] --> B[Finestre temporali]
    B --> C[Forecasting]
    B --> D[Ricostruzione]
    C --> E[Anomaly score]
    D --> E
    E --> F[Smoothing EWMA]
    F --> G[Soglia dinamica e pruning]
    G --> H[Valutazione delle anomalie]
```

1. **Windowing:** finestre di 250 timesteps, con stride 1, su tutte le 25 feature.
2. **Modellazione:** i regressori prevedono i successivi 10 valori della feature `0`; i modelli di ricostruzione ricostruiscono l'intera finestra multivariata.
3. **Anomaly score:** errore assoluto sul segnale previsto, con aggregazione `first`, oppure errore quadratico medio di ricostruzione sulla feature `0` lungo la finestra.
4. **Post-processing:** smoothing EWMA, soglia dinamica ispirata a Telemanom, buffer temporale, pruning e rilevamento su errori invertiti.
5. **Valutazione:** confronto con gli intervalli annotati, analisi per canale e riepilogo globale.

La configurazione comune del post-processing utilizza batch da 70, finestra di 30 batch, `smoothing_perc=0.05`, `error_buffer=100`, `p=0.13` e `inverse=True`.

Le reti neurali utilizzano Adam, learning rate `1e-3`, MSE loss, batch size 64 e un massimo di 35 epoche. L'ultimo 20% delle finestre di training viene riservato alla validazione, con early stopping e ripristino dei pesi migliori (`patience=10`, `min_delta=0.0003`).

## Risultati

Valori ricavati dagli **output già salvati nel notebook V5**, arrotondati a tre decimali; non rappresentano una nuova esecuzione.

| Modello | Precision | Recall | F1 | F0.5 | AP macro | TP | FP | FN |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PCA | 0.641 | 0.368 | 0.467 | 0.558 | 0.322 | 25 | 14 | 43 |
| Random Forest | 0.758 | 0.691 | 0.723 | 0.744 | 0.456 | 47 | 15 | 21 |
| MLP | 0.714 | 0.809 | 0.759 | 0.731 | 0.466 | 55 | 22 | 13 |
| MLP Autoencoder | 0.679 | 0.559 | 0.613 | 0.651 | 0.348 | 38 | 18 | 30 |
| **LSTM** | **0.781** | **0.838** | **0.809** | **0.792** | **0.534** | **57** | **16** | **11** |
| LSTM Autoencoder | 0.811 | 0.441 | 0.571 | 0.694 | 0.347 | 30 | 7 | 38 |

La **LSTM Regressor** ottiene il miglior F1 e individua 57 delle 68 sequenze anomale valutate. La MLP è la seconda configurazione per F1. La LSTM Autoencoder raggiunge la precision più alta, ma ha una recall inferiore e perde 38 sequenze anomale.

Nel confronto svolto, i modelli di forecasting superano quelli di ricostruzione. La conclusione riguarda le configurazioni e il protocollo sperimentale utilizzati.

### Come leggere le metriche

- **Precision, recall, F1 e F0.5** sono calcolati sui conteggi globali di eventi, sommati tra i canali. Il rilevamento usa la sovrapposizione tra intervalli predetti e annotati. Nell'implementazione, ogni intervallo predetto viene associato al primo intervallo reale sovrapposto; i veri positivi contano gli intervalli reali distinti rilevati.
- **AP macro** è la media per canale dell'Average Precision point-wise, calcolata con `average_precision_score`. Nel notebook è denominata `pr_auc_macro`; non è un'area calcolata con integrazione trapezoidale.
- Gli score vengono riallineati alle coordinate del test con offset 250. Il confronto usa la lunghezza comune tra predizioni e label; i primi 250 score sono impostati a zero.

## Esecuzione

### 1. Preparare l'ambiente

Il notebook riporta **Python 3.11.15** nei metadati. Dalla cartella del progetto, creare un ambiente Python 3.11 e installare le dipendenze:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas matplotlib scikit-learn torch jupyterlab ipykernel
python -m ipykernel install --user --name smap-anomaly --display-name "SMAP Anomaly Detection"
```

Su Windows, attivare l'ambiente con `.venv\Scripts\activate` al posto di `source .venv/bin/activate`.

Le versioni delle librerie non sono fissate in un file di ambiente: questi comandi preparano le dipendenze necessarie, ma non ricostruiscono esattamente l'ambiente dell'esperimento originale.

### 2. Preparare i dati

Seguire le indicazioni di download del [repository Telemanom](https://github.com/khundman/telemanom#Getting-Started) e disporre i file nella struttura seguente:

```text
.
├── README.md
├── Anomaly_Detection_V5.ipynb
├── Data_path/
│   ├── labeled_anomalies.csv
│   └── data/
│       └── data/
│           ├── train/
│           │   └── <chan_id>.npy
│           └── test/
│               └── <chan_id>.npy
└── Results/                       # CSV generati dall'esecuzione
```

In alternativa, modificare le tre variabili iniziali del notebook:

```python
TRAIN_PATH = "Data_path/data/data/train"
TEST_PATH = "Data_path/data/data/test"
LABELS = "Data_path/labeled_anomalies.csv"
```

I percorsi sono relativi alla directory di lavoro del notebook. Per ogni canale SMAP presente nel CSV devono essere disponibili entrambi i file di training e test.

### 3. Avviare il notebook

```bash
jupyter lab Anomaly_Detection_V5.ipynb
```

Selezionare il kernel **SMAP Anomaly Detection** ed eseguire le celle in ordine. L'esecuzione completa ripete preprocessing, addestramento e valutazione dei sei modelli su tutti i canali.

Il notebook utilizza CPU o Apple MPS. La prima selezione del dispositivo include anche CUDA, ma le celle successive prima delle sezioni MLP, AE e LSTM sovrascrivono CUDA con CPU quando MPS non è disponibile. Per usare una GPU NVIDIA, uniformare queste celle alla selezione iniziale con `if / elif / else`.

Le finestre di tutti i canali vengono materializzate in memoria: l'esecuzione completa può richiedere molta RAM e tempi di addestramento significativi.

## Output

Il notebook produce grafici dei segnali, anomaly score, soglie, intervalli rilevati e confronti tra modelli. La funzione `save_results` scrive:

```text
Results/<metodo>/<metodo>_channel_results_.csv
Results/<metodo>/<metodo>_global_summary_.csv
```

I nomi dei metodi sono `Random_Forest`, `PCA`, `MLP`, `AE`, `LSTM` e `LSTM-AE`. I CSV vengono sovrascritti a ogni esecuzione delle rispettive celle di salvataggio.

I modelli addestrati e le predizioni restano nei dizionari in memoria. Il notebook V5 non salva né carica automaticamente checkpoint. I grafici del confronto finale vengono mostrati nel notebook; per esportarli, impostare `save_dir` nella chiamata a `plot_metric_comparisons`.

## Limiti e sviluppi futuri

Le reti neurali non hanno un seed globale fissato, quindi i risultati possono variare tra esecuzioni. Inoltre, la separazione training/validazione avviene dopo il windowing: finestre adiacenti ai due lati del confine possono condividere osservazioni.

Forecasting e ricostruzione condividono il post-processing, ma producono score con significato temporale diverso. Le metriche per evento misurano il rilevamento delle sequenze e non quantificano direttamente il ritardo di allarme.

Le estensioni proposte nel notebook includono la calibrazione delle soglie su un set di validazione annotato separato, soglie specifiche per canale e score che sfruttino l'intero orizzonte di previsione. Le label di test devono rimanere riservate alla valutazione finale.
