# PARLA CHIARO — DialectGuard
**Progetto Federico II / Big Data Engineering 2026**
**Vincitore Microsoft LINGUA Open Call (1 di 16 europei selezionati)**

## Obiettivo del progetto reale
PARLA CHIARO è un progetto UNINA PICUS Lab + Microsoft: costruire strumenti AI per proteggere i parlanti dialettali (napoletano, siciliano, romanesco) da errori clinici causati da sistemi ASR+LLM che non capiscono i dialetti.

**I 3 deliverable ufficiali del progetto reale (verificati via web):**
1. Dataset dialettale in ambito sanitario (corpus pilota da 50 clip già disponibile)
2. **Dialect-Aware Warning System (DAWS)** — deliverable centrale, quello che stiamo costruendo
3. **LLM Benchmark** — confronto LLM su comprensione dialettale — è un deliverable esplicito, non un'aggiunta nostra

**60%** dei pazienti parla napoletano durante le visite → AI hallucination → errori clinici.

## Struttura progetto
```
parla_chiaro/
  core/
    semantic_entropy.py      # Kuhn et al. ICLR 2023 — COMPLETO
    token_heatmap.py         # uncertainty attribution per token → dashboard
    whisperx_integration.py  # WhisperX + mDeBERTa OOV detector — COMPLETO
    daws.py                  # risk scoring + MongoDB + clarification — COMPLETO
  analysis/
    overconfidence_analysis.py  # DA COMPLETARE: SE(corretta) vs SE(distorta) + Wilcoxon
  utils/
    device.py                # device detection MPS/CUDA/CPU — COMPLETO
    corpus_loader.py         # loader corpus PARLA CHIARO — DA COMPLETARE
  app.py                     # Streamlit dashboard — MANCANTE (priorità assoluta)
  main.py                    # entry point CLI — COMPLETO (5 modalità)
  requirements.txt
  .env                       # GEMINI_API_KEY=... (non committare)
  REPORT.md                  # report completo tecnico+non-tecnico
```

## Stack aggiornato (sessione 2)
- ASR: WhisperX Large V3 — ctranslate2 CPU/int8 su Apple Silicon, wav2vec2 alignment su MPS
- LLM primario: **Gemini 2.5 Flash Lite** (`gemini-2.5-flash-lite`) — 20 req/day free tier, retry backoff
- LLM alternativo: Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) — ANTHROPIC_API_KEY
- LLM offline: LLaMA-3 via Ollama
- NLI clustering: **cross-encoder/nli-deberta-v3-large** (mantenuto: 3 classi vs 2, SE=1.05 vs 0.50)
- OOV detection: **microsoft/mdeberta-v3-base** (sostituisce UmBERTo — cattura intrasato, aggia, ttruvà)
- Storage: MongoDB
- Dashboard: Streamlit

## Device strategy (Apple M3 Pro)
- `utils/device.py` centralizza tutto
- ctranslate2/WhisperX → `cpu`, `int8` (MPS non supportato)
- wav2vec2 alignment → `mps`
- transformers (DeBERTa, mDeBERTa) → `mps`
- `get_transformers_device_id()` ritorna `"mps"` su Apple Silicon

## Dati disponibili
1. **Subset PARLA CHIARO** (50 registrazioni): `data/`
   - `recordings/{participantId}/{filename}.wav` + `.json`
   - `participants/{participantId}.json`
   - `promptText` nel JSON = **ground truth già disponibile** (non serve annotazione manuale)
   - `participantId` formato: `{data}_{ora}_{dialetto}_{sesso}_{fascia_età}`

2. **Corpus Cacioli 2026** (141 clip napoletane): HuggingFace `anonymous-nsc-author/Neapolitan-Spoken-Corpus`
   - Single speaker, read-aloud (limite dichiarato nel paper)
   - Tassonomia errori Whisper: Phonetic Hallucination, Syntactic Overcorrection, Automatic Italianization, Stress Misplacement
   - WER similarity=0.13 → 87% parole sbagliate su audio napoletano

## Finding critico
**WhisperX "italianizza" il napoletano con ALTA confidenza (0.7-0.9).**
Il segnale `low_conf AND OOV` produce quasi sempre 0 token flaggati.
→ Questo è il contributo principale: dimostrare che ASR-only è insufficiente, serve SE downstream.

## Novel contribution: dual-level uncertainty
**Upstream (ASR):** WhisperX confidence + mDeBERTa OOV flag → segnale dialettale AND logico
**Downstream (LLM):** Semantic entropy Kuhn et al. su claim estratti dalla risposta LLM

**ΔSE analysis:**
```
ΔSE = SE(whisper_output) - SE(ground_truth_text)
```
- ΔSE ≤ 0 → overconfidence: LLM risponde con uguale o maggiore sicurezza sull'input sbagliato
- Test: Wilcoxon signed-rank (paired, non-parametric, one-tailed), H0: ΔSE ≤ 0

**Ablation study:** ASR-only vs SE-only vs Combined(DAWS) → Combined strettamente migliore

## Bug risolti (non riaprire)
1. DeBERTa 807>512 token overflow → SE=0 artificiale → FIXATO: truncate a 200 chars in `_entails()`
2. Gemini 429 quota su gemini-2.0-flash → FIXATO: usare gemini-2.5-flash-lite (limite=20/day)
3. UmBERTo miss su `guaglione` (2 subword) → FIXATO: mDeBERTa-v3-base (threshold=3 subword)
4. Gemini 503 crash → FIXATO: exponential backoff retry (5,10,20s, max 4 tentativi)

## Priorità prossima sessione
1. **`app.py` Streamlit** — upload audio → trascrizione → heatmap → SE gauge → alert (CRITICO per commissione)
2. **`analysis/overconfidence_analysis.py`** — loop corpus + ΔSE + Wilcoxon + scatter plot
3. **LLM Benchmark** — confronto Gemini/Claude/LLaMA/GPT-4o-mini su stessa coppia (gt, distorted)
4. **`utils/corpus_loader.py`** — loader per batch processing del corpus

## Riferimenti
- Kuhn et al., "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in NLG", ICLR 2023
- Cacioli et al., Neapolitan Spoken Corpus, HuggingFace 2026
- [PARLA CHIARO Microsoft](https://news.microsoft.com/source/emea/features/il-progetto-parla-chiaro-delluniversita-di-napoli-federico-ii-selezionato-nellambito-della-lingua-open-call-di-microsoft-per-la-tutela-dei-dialetti-italiani-nellera-dellintelligenza-art/?lang=it)
- [PARLA CHIARO UNINA](https://www.unina.it/it/w/il-progetto-parla-chiaro-dell-universit%C3%A0-federico-ii-selezionato-nell-ambito-della-lingua-open-call-di-microsoft)
- [Sito ufficiale](https://parla-chiaro.azurewebsites.net/)
