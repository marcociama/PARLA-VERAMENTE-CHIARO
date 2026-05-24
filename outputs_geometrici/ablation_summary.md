# PARLA CHIARO — Ablation Study UQ: Tabella Finale
**Data:** 2026-05-21 | **N=50 campioni** | **LLM:** Mistral (Ollama, greedy T=0)

## Target di correlazione

| Target | Definizione |
|:---|:---|
| **WER** | Word Error Rate di WhisperX vs ground truth (errore acustico) |
| **E_sem_top** | `1 - cos_sim(R_W1, R_GT1)` — degrado diagnostico sul top-beam (768D SBERT) |
| **E_sem_cross** | `1 - mean(cos_sim)` su 9 coppie W×GT — degrado diagnostico medio (768D SBERT) |

---

## Risultati

| Metodo | Kernel | GT | vs WER | vs E_sem_top | vs E_sem_cross |
|:---|:---|:---:|:---:|:---:|:---:|
| **Inv-Entropy H_k1 (base)** | SBERT 768D | sì | **+0.5226** | +0.4947 | +0.6664 |
| VN H_true | RBF 1D | sì | +0.4021 | +0.5988 | **+0.6914** |
| 1D Markov Spettrale ORACLE | Laplace 1D | sì | +0.4588 | **+0.6066**¹ | +0.5579 |
| 1D Markov Asimm. ORACLE | Laplace 1D | sì | +0.4320 | +0.4337 | +0.4829 |
|  |  |  |  |  |  |
| **1D Markov Spettrale** | **Laplace** | no | **+0.5010** | **+0.5688** | +0.5684 |
| VN H_hybrid | RBF | no | +0.4587 | +0.5525 | **+0.5767** |
| 1D Markov Asimm. (Diag) | Laplace | no | +0.4425 | +0.4489 | +0.5340 |
| 1D Markov Spettrale | RBF | no | +0.4408 | +0.4644 | +0.5170 |
| 1D Markov Asimm. (Diag) | RBF | no | +0.4263 | +0.4548 | +0.5468 |
| VN H_hybrid | Laplace | no | +0.3620 | +0.5015 | +0.4201 |

> ¹ Valore E_sem_top di riferimento da Inv-Entropy H_k4 (+0.6066); 1D Markov Spettrale ORACLE riporta +0.5361.

**Grassetto** = migliore per colonna nel rispettivo blocco (offline / online).

---

## Note metodologiche

- **GT a runtime = sì (OFFLINE/ORACLE):** il metodo usa i testi ground truth del paziente — non deployabile in produzione clinica reale.
- **GT a runtime = no (ONLINE/NO-GT):** usa solo ancoraggi statistici congelati durante la calibrazione offline + proiezioni live delle trascrizioni WhisperX.
- **Inv-Entropy H_k1:** formula Song et al. NeurIPS 2025, aSim = clip(cos, 0, 1)^1, diag(Py@Px), unnorm. Upper bound offline per WER.
- **VN H_true / H_hybrid:** Von Neumann entropy su matrici RBF 1D (asse SVM output), σ = mediana distanze. H_hybrid usa blocco GT-GT congelato.
- **1D Markov Spettrale:** Py@Px con kernel Laplaciano asimmetrico (due assi SVM separati), entropia di Shannon su |eigvals| normalizzati. Best online su WER.
- **1D Markov Asimm. (Diag):** stessa pipeline ma entropia solo sulla diagonale di (Py@Px)ᵀ.
- **σ** calcolato come mediana delle distanze assolute 1D (mediana trick) — base comune per Laplace e RBF.

---

## Configurazione produzione consigliata

| Scenario | Metodo | Pearson WER | Pearson E_sem_cross |
|:---|:---|:---:|:---:|
| Online (no GT) — best WER | 1D Markov Spettrale Laplace | +0.5010 | +0.5684 |
| Online (no GT) — best E_sem | VN H_hybrid RBF | +0.4587 | +0.5767 |
| **Produzione DAWS (ablation 2026-05-21)** | **1D Markov Spettrale Laplace** | **+0.5010** | **+0.5684** |
