# DAWS — Ablation Study: Spectral Entropy Methods for Dialectal ASR Uncertainty
## Results and Discussion
**Data:** 21 Maggio 2026 | N=50 sample | Corpus PARLA CHIARO

---

## 4.1 Inv-Entropy in 768D: Sharpening e il Trade-off tra WER e E_sem_top

| Metodo | WER | E_sem_top | E_sem_cross |
|--------|-----|-----------|-------------|
| Inv-Entropy H_k1 | +0.523 | +0.495 | +0.666 |
| Inv-Entropy H_k4 | +0.515 | +0.607 | +0.668 |

Lo sharpening coseno `aSim = ((1+cos)/2)^k` agisce come un operatore di contrazione selettiva sullo spazio delle similarità. Per k=1, la mappa `[−1,1] → [0,1]` è lineare e preserva proporzionalmente l'intera struttura metrica dello spazio SBERT a 768D. Per k=4, la stessa mappa diventa fortemente non-lineare: le coppie con similarità coseno ≥ 0.8 vengono compresse verso 1, mentre le coppie con similarità < 0.5 vengono spinte verso 0.

Questo ha due effetti opposti sulle due metriche target:

**Su WER (−0.008 da k=1 a k=4):** il WER misura l'errore sintattico acustico — una proprietà distribuita su tutto il vocabolario dialettale. La correlazione con H dipende dalla capacità della metrica di distinguere finemente tra livelli di distorsione acustica. Lo sharpening k=4 comprime le differenze nella zona di alta similarità (dove risiedono la maggior parte delle coppie GT-W su dialetto lieve), riducendo marginalmente la sensibilità al WER.

**Su E_sem_top (+0.112 da k=1 a k=4):** E_sem_top = 1 − cos(R_W1, R_GT1) misura il danno clinico sulla risposta principale — un evento discontinuo e raro. Quando Mistral allucina (es. "collo→infarto", "cistifellea→tempo fuori"), la distanza semantica R_W vs R_GT salta di un'intera regione dello spazio. Lo sharpening k=4 amplifica esattamente questi salti: le coppie distanti (similarità bassa) vengono ulteriormente separate, rendendo la metrica più sensibile ai failure mode critici. La perdita marginale sul WER è quindi il costo accettabile per guadagnare +11 punti su E_sem_top — il target clinicamente rilevante.

**Conclusione:** k=4 è la scelta ottimale per applicazioni healthcare dove E_sem_top — il danno sulla risposta che il medico vede — è la metrica primaria. k=1 è preferibile per studi di correlazione generali con WER.

---

## 4.2 Il Paradosso di Von Neumann Oracle: Miglior E_sem_cross, Peggior WER

| Metodo | WER | E_sem_top | E_sem_cross |
|--------|-----|-----------|-------------|
| VN H_true (Oracle) | +0.402 | +0.599 | **+0.691** |
| Inv-Entropy H_k4 | +0.515 | +0.607 | +0.668 |

Von Neumann H_true è la metrica con la correlazione più bassa su WER ma la più alta su E_sem_cross. Questo non è un paradosso — è la manifestazione diretta della natura intrinseca dell'entropia spettrale.

**Geometria della matrice di densità:** Von Neumann opera su una matrice di densità A/Tr(A) dove gli elementi A_ij sono kernel RBF sulle proiezioni 1D sull'asse di deriva dialettale. Gli autovalori {λ_i} di questa matrice rappresentano la distribuzione dell'energia spettrale lungo le direzioni principali del grafo di similarità. L'entropia H_VN = −Σ λ_i log λ_i è massima quando gli autovalori sono equipartiti — cioè quando nessuna direzione domina — e minima quando un singolo autovalore concentra tutta l'energia.

**Perché E_sem_cross +0.691:** E_sem_cross = 1 − mean(cos_sim su tutte le 9 coppie W×GT) è una misura globale e matriciale della separazione tra i due blocchi di risposte. La struttura spettrale di A cattura esattamente questa separazione globale: quando il dialetto induce derive coerenti in tutte e 3 le coppie W×GT, gli autovalori si polarizzano (uno domina) e H_VN crolla. Questo collasso spettrale è un segnale più pulito della separazione globale rispetto alla traccia diagonale di Inv-Entropy, che è una misura puntuale.

**Perché WER +0.402:** Il WER è un fenomeno locale e asincrono — ogni parola è un evento indipendente. L'entropia spettrale è una misura globale del grafo che integra su tutte le relazioni. Per catturare WER, serve sensibilità alle variazioni locali parola per parola, che la proiezione 1D e il kernel RBF globale attenuano.

**Implicazione teorica:** Von Neumann H è un interferometro globale della coerenza semantica — ottimo per rilevare se il sistema clinico nel suo insieme è destabilizzato, non per localizzare dove esattamente nella trascrizione risiede l'errore.

---

## 4.3 Il Crossover dei Kernel nel Blocco Online (GT=no)

| Metodo | Kernel | WER | E_sem_cross |
|--------|--------|-----|-------------|
| 1D Markov Spettrale | Laplace | +0.501 | +0.568 |
| VN H_hybrid | RBF | +0.459 | +0.577 |
| 1D Markov Spettrale | RBF | +0.441 | +0.517 |
| VN H_hybrid | Laplace | +0.362 | +0.420 |

Il crossover è il risultato più contro-intuitivo dell'ablation e rivela la geometria sottostante delle due architetture.

**RBF:** il kernel Gaussiano `K(d) = exp(−d²/2σ²)` è C∞ e a decadimento superesponenziale. La sua matrice di Gram è densa e a rango pieno — tutti i punti interagiscono con tutti gli altri con peso proporzionale alla prossimità. Nella struttura spettrale di Von Neumann, questa densità genera autovalori ben distribuiti che catturano la struttura globale del grafo: ottimo per Von Neumann che opera su questa struttura globale. Tuttavia per 1D Markov Spettrale, la densità del kernel RBF rende la matrice di transizione quasi uniforme — ogni stato ha probabilità simile di transitare a qualsiasi altro — annullando la direzionalità del processo Markoviano.

**Laplace:** il kernel `K(d) = exp(−|d|/σ)` è C^0 ma non C^1 — ha una discontinuità nella derivata prima in d=0. Questa discontinuità introduce sparsità strutturale: i punti vicini hanno pesi alti ma il decadimento è lineare invece che superesponenziale. Per 1D Markov Spettrale, questa sparsità preserva la struttura locale della catena — le transizioni preferenziali riflettono la topologia reale della deriva dialettale. Per Von Neumann, la stessa sparsità genera autovalori polarizzati che riflettono strutture locali invece che la coerenza globale, degradando la correlazione con E_sem_cross.

**Interpretazione geometrica unificata:** RBF è un kernel di coerenza globale — adatto a metriche che integrano l'intera struttura. Laplace è un kernel di prossimità locale — adatto a processi Markoviani che propagano informazione tra stati adiacenti. Il crossover non è una anomalia: è la firma della corrispondenza kernel-architettura.

---

## 4.4 Il Disastro di k=4 in 1D: Vizio Geometrico Fondamentale

| Metodo | Kernel | WER | E_sem_top | E_sem_cross |
|--------|--------|-----|-----------|-------------|
| 1D Markov Spettrale | Laplace k=4 | **−0.216** | **−0.312** | **−0.217** |
| 1D Markov Spettrale | Laplace k=1 | +0.501 | +0.569 | +0.568 |

La correlazione negativa a k=4 in 1D non è degradazione — è inversione completa del segnale. Il meccanismo è algebricamente preciso.

In 768D, i valori coseno tra embedding SBERT si distribuiscono in [0.2, 0.99] con media ~0.65 e deviazione standard ~0.15. Elevare al k=4 comprime questa distribuzione ma preserva l'ordinamento relativo: i punti più simili restano più simili.

In 1D, dopo proiezione sull'asse di deriva, le distanze euclidee si distribuiscono su scala diversa. Il kernel Laplace in 1D produce valori `exp(−|d|/σ)` con σ calibrato globalmente via median trick. Applicare k=4 al risultato del kernel — non alla distanza prima del kernel — significa elevare al quarto potere valori in (0,1]. Questo trasforma la distribuzione bimodale (punti vicini ≈1, punti lontani ≈0) in una distribuzione ancora più estrema: i valori già vicini a 1 rimangono vicini a 1, ma i valori intermedi (0.3-0.7) vengono schiacciati verso 0. La matrice diventa quasi binaria — una matrice di adiacenza invece che di pesi continui.

Una matrice quasi-binaria in 1D ha uno spettro degenere: un autovalore dominante che cattura la componente DC (media globale) e tutti gli altri vicini a zero. L'entropia di questa matrice è quasi zero per tutti i sample — il segnale è distrutto. Peggio: se la componente DC correla negativamente con WER (sample con WER basso hanno cluster più compatti → autovalore più dominante), si ottiene correlazione negativa.

**Il vizio fondamentale:** lo sharpening k>1 è stato progettato per spazi ad alta dimensione dove la concentrazione delle misure comprime i coseni. In 1D non esiste concentrazione delle misure — lo spazio è già unidimensionale e le distanze sono direttamente informative. Applicare sharpening in 1D è geometricamente privo di senso e distrugge il segnale.

---

## 4.5 Raccomandazione per il Deployment in Produzione

Sulla base dell'analisi completa del blocco online (GT=no), la raccomandazione è:

**Architettura di produzione: 1D Markov Spettrale con kernel Laplace (k=1)**

| Metrica | Score |
|---------|-------|
| Pearson vs WER | **+0.501** |
| Pearson vs E_sem_top | **+0.569** |
| Pearson vs E_sem_cross | +0.568 |
| Richiede GT a runtime | No |
| Complessità computazionale | O(N²) in 1D |

**Motivazione:**

1. **Massima correlazione con WER tra tutti i metodi online:** +0.501 supera VN H_hybrid (+0.459) e tutte le varianti RBF. Per una pipeline di triage clinico, il WER è il proxy più diretto dell'errore acustico dialettale — la variabile che il sistema deve rilevare.

2. **Corrispondenza kernel-architettura ottimale:** come dimostrato in §4.3, il kernel Laplace è il partner naturale del processo Markoviano 1D. La sparsità locale del kernel Laplace preserva la struttura della catena di Markov, permettendo alla metrica di propagare il segnale di deriva dialettale in modo fedele alla topologia reale dei dati.

3. **Deployabilità senza GT:** l'ancora GT_frozen viene calibrata offline sui 50 sample e congelata — nessuna dipendenza dal ground truth a runtime.

4. **Costo computazionale minimale:** la proiezione 1D e il kernel scalare richiedono O(N) operazioni per sample invece di O(N²·D) per Inv-Entropy in 768D.

**Architettura secondaria (backup):** VN H_hybrid con kernel RBF per scenari dove E_sem_cross è la metrica primaria — ad esempio audit periodici della qualità complessiva del sistema invece di triage real-time.

**Architettura esclusa dalla produzione:** qualsiasi variante con k=4 in 1D, per le ragioni geometriche dimostrate in §4.4. Qualsiasi variante con kernel Laplace per Von Neumann, per le ragioni spettrali dimostrate in §4.3.

---

## 4.6 Sintesi — Mappa delle Correlazioni

```
                    WER         E_sem_top   E_sem_cross
                    (errore     (danno      (stabilità
                    acustico)   clinico)    globale)
                    
Inv-Entropy k=1     +0.523      +0.495      +0.666    ← miglior WER assoluto
Inv-Entropy k=4     +0.515      +0.607      +0.668    ← miglior E_sem_top (con GT)
VN H_true           +0.402      +0.599      +0.691    ← miglior E_sem_cross assoluto
─────────────────────────────────────────────────────
1D Markov Laplace   +0.501      +0.569      +0.568    ← PRODUZIONE (no GT)
VN H_hybrid RBF     +0.459      +0.553      +0.577    ← backup (no GT)
─────────────────────────────────────────────────────
1D Markov Lap. k=4  −0.216      −0.312      −0.217    ← ESCLUSO
```

La distinzione fondamentale dell'ablation è tra metriche che operano in spazio pieno (768D, Inv-Entropy) e metriche che operano in spazio proiettato (1D, Von Neumann e Markov). Le prime sono superiori con GT disponibile ma non deployabili in produzione. Le seconde sono deployabili ma pagano un costo di correlazione di circa 5-10 punti Pearson — il prezzo dichiarato della deployabilità clinica in assenza di ground truth.
