# PARLA CHIARO — Report tecnico e scientifico
**Big Data Engineering 2026 — Università Federico II di Napoli**
*Generato: 8 maggio 2026*

---

## 1. Il progetto reale: contesto e obiettivi

### Cos'è PARLA CHIARO (il vero progetto)
PARLA CHIARO è un progetto di ricerca dell'Università Federico II (PICUS Lab, DIETI), selezionato tra i 16 vincitori europei della **Microsoft LINGUA Open Call** — un bando dedicato alla tutela dei dialetti italiani nell'era dell'intelligenza artificiale.

Il problema di partenza è concreto: circa il **60% dei pazienti** parla in dialetto napoletano durante le visite mediche. I sistemi AI (ASR + LLM) sono addestrati principalmente su italiano standard. Risultato: trascrizioni distorte, hallucination del modello, errori clinici.

### I tre deliverable ufficiali del progetto reale
1. **Dataset dialettale in ambito sanitario** — registrazioni etichettate di napoletano, siciliano, romanesco in contesti clinici (il corpus da 50 clip che abbiamo è la fase pilota di questo)
2. **Dialect-Aware Warning System (DAWS)** — il sistema di alert automatico che stiamo costruendo: è **il deliverable centrale**
3. **LLM Benchmark** — confronto sistematico tra i principali LLM sulla comprensione dei dialetti, per identificare quale sia più affidabile in contesto clinico

Questa è un'ottima notizia: **il benchmark LLM è un deliverable esplicito del progetto reale**, non un'aggiunta da noi inventata.

---

## 2. Spiegazione non tecnica: cosa fa il sistema

### Il problema in parole semplici
Immagina un anziano napoletano che va dal medico. Il medico usa un sistema AI per trascrivere la visita. Il paziente dice "ho nu dulore cà dint'" (ho un dolore qui dentro). Il sistema AI:
1. Trascrive male (non conosce il napoletano): "ho un dolore caro dentro"
2. L'LLM risponde come se avesse capito tutto, con falsa sicurezza
3. Il medico si fida della risposta AI — errore clinico potenziale

### La soluzione: PARLA CHIARO DAWS
Il sistema fa tre cose:
1. **Trascrive** l'audio con WhisperX (Whisper Large V3 + allineamento forzato)
2. **Misura l'incertezza** della risposta LLM (quanto è davvero sicuro ciò che dice?)
3. **Lancia un alert** se la trascrizione è probabilmente dialettale E il modello risponde con troppa sicurezza su qualcosa che in realtà non ha capito bene

### Il trucco chiave: la semantic entropy
Invece di chiedere al modello "sei sicuro?" (risponderebbe sempre sì), gli facciamo la stessa domanda **10 volte** con piccole variazioni casuali. Se risponde sempre lo stesso modo → è davvero sicuro. Se dà risposte diverse ogni volta → non sa. Questa misura si chiama **semantic entropy** (entropia semantica) ed è basata su un paper ICLR 2023 (Kuhn et al.).

---

## 3. Architettura tecnica della pipeline

```
Audio WAV
   │
   ▼
[WhisperX Large V3]
   │  → Trascrizione testo
   │  → Confidence per parola ∈ [0,1]
   │
   ▼
[OOV Detector — mDeBERTa-v3-base tokenizer]
   │  → Flag: parola in vocabolario italiano standard?
   │  → Segnale dialettale = bassa_conf AND out-of-vocabulary
   │
   ▼
[LLM — Gemini 2.5 Flash Lite / Claude Haiku]
   │  → Risposta al claim (T=0.5, n=10 campionamenti)
   │
   ▼
[NLI Clusterer — cross-encoder/nli-deberta-v3-large]
   │  → Entailment bidirezionale: raggruppa risposte con stesso significato
   │  → Ogni cluster = un "significato distinto"
   │
   ▼
[Semantic Entropy Calculator]
   │  SE = -Σ p(cluster) log p(cluster)
   │  SE = 0 → certezza assoluta (sospetta: è overconfidence?)
   │  SE = log(N) → massima incertezza
   │
   ▼
[DAWS — Dialect-Aware Warning System]
   │  risk_score = 0.6 × SE_norm + 0.4 × asr_overlap
   │  Verde / Giallo / Rosso
   │  → Genera domanda di chiarimento se alert
```

---

## 4. Stack tecnologico aggiornato

| Componente | Tecnologia | Note |
|---|---|---|
| ASR | WhisperX Large V3 | ctranslate2, CPU/int8 su Apple Silicon |
| Allineamento | wav2vec2 (italiano) | MPS su Apple Silicon |
| OOV Detection | mDeBERTa-v3-base (Microsoft) | Sostituisce UmBERTo — multilingue, più robusto su testo distorto |
| NLI Clustering | cross-encoder/nli-deberta-v3-large | 3 classi (entail/neutral/contradict), granularità superiore |
| LLM (produzione) | Gemini 2.5 Flash Lite | API gratuita, 20 req/day, backoff retry |
| LLM (alternativa) | Claude Haiku 4.5 | Anthropic API, log_prob uniforme |
| LLM (offline) | LLaMA-3 via Ollama | Fallback senza connessione |
| Dashboard | Streamlit (app.py) | **Da completare** |
| Storage | MongoDB (core/daws.py) | Per logging alert e audit trail |
| Device mgmt | utils/device.py | MPS/CUDA/CPU automatico |

**Scelta chiave:** mDeBERTa-v3-base per OOV (cattura `intrasato`, `aggia`, `ttruvà`) ma DeBERTa-v3-large per NLI (3 cluster vs 2 sullo stesso testo, SE=1.05 vs 0.50).

---

## 5. Contributo novel: dual-level uncertainty

### L'idea
Il contributo originale di questo progetto rispetto alla letteratura è la combinazione di **due livelli di incertezza** distinti:

**Livello upstream (ASR)**
- Confidence WhisperX per parola ∈ [0,1]
- OOV flag via tokenizer mDeBERTa
- Segnale: `low_conf AND OOV` → probabile dialetto non capito

**Livello downstream (LLM)**
- Semantic entropy di Kuhn et al. applicata a claim estratti dalla risposta LLM
- Misura quanto l'LLM sia davvero certo di ciò che afferma

### Perché è novel
Kuhn et al. (ICLR 2023) misurano l'incertezza del LLM in isolamento. Noi la misuriamo **in funzione della qualità dell'input**: la stessa domanda con trascrizione corretta vs distorta dovrebbe produrre SE diverse. Se non le produce (SE_corretta ≈ SE_distorta basse entrambe) → **overconfidence sistematica**.

### L'analisi ΔSE
```
ΔSE = SE(whisper_output) - SE(ground_truth_text)
```
- ΔSE > 0 → il modello è più incerto sul testo distorto (sistema funziona)
- ΔSE ≤ 0 → il modello risponde con uguale o maggiore confidenza sul testo sbagliato → **overconfidence su input dialettale**

**Test statistico:** Wilcoxon signed-rank (paired, non-parametric, one-tailed). H0: ΔSE ≤ 0. Se p < 0.05 → dimostriamo che il dialetto induce overconfidence misurabile.

### Ablation study (per la presentazione)
| Configurazione | Precision | Recall |
|---|---|---|
| ASR-only (WhisperX conf + OOV) | ? | ? |
| SE-only (Kuhn) | ? | ? |
| Combined (DAWS completo) | ? | ? |

Il combined dovrebbe essere strettamente migliore perché i due errori sono complementari: WhisperX "italianizza" con alta confidence → ASR-only = 0 segnali. SE cattura ciò che ASR non vede.

---

## 6. Finding critico: Whisper "italianizza" con alta confidenza

Dall'analisi del corpus:
- WhisperX Large V3 su audio napoletano → trascrive in italiano standard con confidence **alta** (0.7-0.9)
- Esempio Cacioli 2026: "me so' intrasato" → Whisper → testo italianizzato con WER=87%
- Conseguenza: il segnale `low_conf AND OOV` è quasi sempre assente (0 token flaggati)
- **Questo è il contributo principale**: dimostrare che il segnale ASR da solo è insufficiente, serve la semantic entropy sul downstream LLM

Il corpus Cacioli 2026 (141 clip HuggingFace: `anonymous-nsc-author/Neapolitan-Spoken-Corpus`) fornisce una tassonomia ready-made degli errori Whisper:
1. **Phonetic Hallucination** — suono dialettale → parola italiana inventata
2. **Syntactic Overcorrection** — struttura grammaticale normalizzata
3. **Automatic Italianization** — lessema napoletano → lessema italiano simile per suono
4. **Stress Misplacement** — accento sbagliato cambia il significato

---

## 7. LLM Benchmark (deliverable esplicito del progetto reale)

### Struttura proposta
Confronto tra LLM su coppie (ground_truth, whisper_distorted):

**Metriche:**
- SE_corretto: entropia semantica con testo ground truth
- SE_distorto: entropia semantica con output Whisper
- ΔSE = SE_distorto - SE_corretto
- Overconfidence rate: % casi con SE_distorto < soglia (modello sicuro su input sbagliato)

**LLM da confrontare:**
1. Gemini 2.5 Flash Lite (già integrato)
2. Claude Haiku 4.5 (già integrato)
3. LLaMA-3-8B via Ollama (già integrato)
4. GPT-4o-mini (da aggiungere, OpenAI API)
5. Gemini 2.5 Flash (version più grande, se quota disponibile)

**Ipotesi:** modelli più grandi → ΔSE più alto (riconoscono meglio l'incertezza sull'input distorto). Ma tutti hanno overconfidence sistematica (ΔSE < soglia ottimale) → giustifica DAWS.

---

## 8. Stato attuale del codice

### File completati
- `utils/device.py` — device detection MPS/CUDA/CPU automatico
- `core/semantic_entropy.py` — pipeline Kuhn + GeminiLLM + AnthropicLLM + OllamaLLM
- `core/whisperx_integration.py` — WhisperX + mDeBERTa OOV detector
- `core/daws.py` — risk scoring + alert levels
- `main.py` — entry point CLI (5 modalità: smoke, corpus, transcribe, pipeline, overconfidence)
- `.env` — GEMINI_API_KEY

### File da completare (priorità per la commissione)
1. **`app.py`** — Streamlit dashboard (CRITICO: è ciò che vede la commissione)
   - Audio upload → trascrizione live → heatmap token → risk gauge
   - Confronto SE(ground_truth) vs SE(distorted) interattivo
2. **`analysis/overconfidence_analysis.py`** — loop su corpus + Wilcoxon test + scatter plot
3. **`utils/corpus_loader.py`** — già parzialmente definito, da completare per Cacioli

### Bug noti e fix applicati
- DeBERTa 807>512 token overflow → SE=0 artificiale → **FIXATO** (truncate a 200 chars)
- Gemini 429 quota → **FIXATO** (backoff exponential 5/10/20s + modello corretto gemini-2.5-flash-lite)
- UmBERTo miss su `guaglione` → **FIXATO** (sostituito con mDeBERTa-v3-base)

---

## 9. Allineamento con la traccia ufficiale

| Requisito traccia | Stato |
|---|---|
| ASR con WhisperX | Implementato |
| OOV detection | Implementato (mDeBERTa, upgrade da UmBERTo) |
| Semantic entropy (Kuhn et al.) | Implementato |
| NLI clustering (DeBERTa) | Implementato |
| DAWS con risk scoring | Implementato |
| Streamlit dashboard | **Mancante — priorità assoluta** |
| MongoDB storage | Struttura in daws.py, da connettere |
| Corpus PARLA CHIARO | 50 clip disponibili in data/ |
| Novel contribution | Dual-level uncertainty + ΔSE analysis |
| LLM Benchmark | Design pronto, da eseguire su corpus |

---

## 10. Piano per le prossime sessioni

### Fase 1 — App Streamlit (urgente)
```python
# app.py deve mostrare:
# - Upload audio WAV
# - Trascrizione WhisperX in tempo reale
# - Heatmap token colorata (rosso = segnale dialettale)
# - Risposta LLM con SE per ogni claim
# - Gauge del risk score DAWS (verde/giallo/rosso)
# - Domanda di chiarimento se alert rosso
```

### Fase 2 — Overconfidence analysis su corpus
```python
# analysis/overconfidence_analysis.py
# Per ogni clip in data/:
#   ground_truth = promptText (già nel JSON)
#   whisper_out = WhisperX.transcribe(wav)
#   SE_gt = pipeline.evaluate(ground_truth)
#   SE_wh = pipeline.evaluate(whisper_out)
#   delta_SE = SE_wh - SE_gt
# Wilcoxon signed-rank test su delta_SE
# Scatter plot SE_gt vs SE_wh
```

### Fase 3 — LLM Benchmark
Eseguire pipeline con tutti gli LLM sulla stessa coppia (ground_truth, distorted), tabulare risultati, produrre figura comparativa.

---

## Fonti
- [PARLA CHIARO — Microsoft Source EMEA](https://news.microsoft.com/source/emea/features/il-progetto-parla-chiaro-delluniversita-di-napoli-federico-ii-selezionato-nellambito-della-lingua-open-call-di-microsoft-per-la-tutela-dei-dialetti-italiani-nellera-dellintelligenza-art/?lang=it)
- [PARLA CHIARO — UNINA comunicato](https://www.unina.it/it/w/il-progetto-parla-chiaro-dell-universit%C3%A0-federico-ii-selezionato-nell-ambito-della-lingua-open-call-di-microsoft)
- [PARLA CHIARO — sito ufficiale](https://parla-chiaro.azurewebsites.net/)
- [Libero.it — analisi dialetti e AI](https://www.libero.it/tecnologia/dialetti-problema-serio-ai-medici-parla-chiaro-114574)
- Kuhn et al., "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in NLG", ICLR 2023
- Cacioli et al., "Neapolitan Spoken Corpus", HuggingFace 2026 (`anonymous-nsc-author/Neapolitan-Spoken-Corpus`)
