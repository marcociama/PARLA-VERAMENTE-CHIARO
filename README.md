# PARLA CHIARO — Dialect-Aware Warning System (DAWS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/LLM-Mistral%207B%20v0.3-orange.svg)](https://ollama.com/)
[![ASR](https://img.shields.io/badge/ASR-WhisperX%20Large%20V3-red.svg)](https://github.com/m-bain/whisperX)

**PARLA CHIARO** è un framework local-first di **Uncertainty Quantification (UQ)** applicato a sistemi in cascata (Speech-to-Text $\rightarrow$ LLM) in ambito sanitario (*Healthcare*). Il sistema è specificamente progettato per intercettare ed arrestare la propagazione degli errori diagnostici indotti dalle alterazioni fonetiche e semantiche del **dialetto napoletano**.

L'architettura implementa il metodo proprietario **1D Markov Spettrale**, in grado di operare in modalità totalmente online e single-pass a runtime (senza Ground Truth), superando lo Stato dell'Arte offline di *Inv-Entropy* (NeurIPS 2025) sul recupero del danno semantico top-beam ($+0.5688$ vs $+0.4947$).

---

## 🛠️ Architettura del Sistema

Il framework si articola in quattro macro-moduli sequenziali:

1. **ASR & Test-Time Data Augmentation (TTA):** WhisperX Large V3 effettua la trascrizione primaria. Un modulo di perturbazione acustica gaussiana ($\alpha=0.005$) genera due varianti audio live per quantificare l'incertezza aleatoria dello speaker.
2. **SVM Geometrica 1D:** Gli embedding a 768D di SBERT vengono proiettati su retti di drift separate per Input (acustico) e Output (clinico), abbattendo la complessità computazionale a $O(N^2)$ online.
3. **Core Markoviano Spettrale:** Modellazione causale asimmetrica tramite accoppiamento di matrici stocastiche generati da Kernel Laplaciani ($k=1$). L'entropia spettrale $H_{\text{spectral}}$ viene estratta dal modulo degli autovalori complessi del prodotto congiunto $P_y P_x$.
4. **Triage Dashboard:** Frontend Streamlit che mappa il rischio clinico tramite un *Severity Score* moltiplicativo e clippato contro gli outlier.

---

## 🚀 Quick Start (Installazione Locale)

Il progetto è configurato per girare interamente in locale su hardware Apple Silicon (accelerazione MPS) o GPU NVIDIA.

### 1. Prerequisiti

* **Python 3.10+**
* **MongoDB** (attivo in background sulla porta di default `27017`)
* **Ollama** con il modello Mistral caricato:
  ```bash
  ollama run mistral
  ```

### 2. Clonazione e Configurazione Ambiente

```bash
git clone https://github.com/tuo-username/parla_chiaro.git
cd parla_chiaro

# Creazione virtual environment
python -m venv .venv
source .venv/bin/activate  # Su Windows: .venv\Scripts\activate

# Installazione dipendenze core
pip install -r requirements.txt
```

### 3. Ripristino Database e Calibrazione Offline (N=50)

Prima di avviare la produzione, è necessario calibrare le soglie empiriche del potenziometro di triage sul corpus di 50 pazienti:

```bash
python scripts/benchmark_final.py
```

Questo script genererà il file `config/geometry_calibration.json` popolando i limiti min/max di entropia e salvando i vettori della retta di drift.

### 4. Avvio della Dashboard Live

```bash
streamlit run daws/ui/app.py
```

---

## 📊 Risultati del Benchmark Scientifico

La validazione sul corpus PARLA CHIARO ($N=50$, 29 speaker bilanciati, età 18-80+) mostra i seguenti coefficienti di correlazione di Pearson rispetto alle metriche di danno reale:

| Metodo | GT a Runtime | vs WER (Acustico) | vs E_sem_top (Clinico) | vs E_sem_cross (Globale) |
| :--- | :---: | :---: | :---: | :---: |
| **Inv-Entropy H_k1 (Offline UB)** | Sì (Oracle) | $+0.5226$ | $+0.4947$ | $+0.6664$ |
| **1D Markov Spettrale ONLINE** | No | $+0.5010$ | $+0.5688$ | $+0.5684$ |

---

## 📂 Struttura del Repository

```plaintext
parla_chiaro/
├── config/                  # Parametri e soglie di calibrazione JSON
├── daws/
│   ├── database/            # Connettori e schemi MongoDB
│   ├── pipeline/            # Core logico del Dialect-Aware Warning System
│   └── ui/                  # Dashboard Streamlit (app.py)
├── outputs_geometrici/      # Dataset topologico N=50 e record storici
├── scripts/                 # Pipeline di calibrazione e benchmark finali
├── .gitignore               # Esclusione cache, modelli pesanti e wav locali
└── requirements.txt         # Dipendenze congelate del progetto
```

---

## 🎓 Ringraziamenti e Contesto

Sviluppato come progetto d'esame per il corso di Big Data 2026 presso l'Università degli Studi di Napoli Federico II — PICUS Lab / Microsoft LINGUA.
