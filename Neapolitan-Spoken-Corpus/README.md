---
tags:
- audio
- speech-recognition
- neapolitan
- low-resource
license: cc-by-nc-4.0
---
# Neapolitan-Spoken-Corpus (NSC)

**Neapolitan-Spoken-Corpus (NSC)** is the first publicly available speech corpus designed specifically for benchmarking Automatic Speech Recognition (ASR) systems on Neapolitan, a low-resource Romance dialect of Southern Italy. It includes 141 sentence-level audio recordings along with gold-standard orthographic transcriptions.

The dataset was created to address the lack of computational resources for dialectological research and the development of equitable speech technologies.

## Dataset Description

- **Language:** Neapolitan (ISO 639-3: nap)  
- **Audio Format:** `.m4a`  
- **Number of Samples:** 141  
- **Domains Covered:** Traditional plays, regional poetry, community blogs  
- **Transcriptions:** Orthographic Neapolitan sentences provided by native speakers  
- **Ethical Considerations:** All participants provided informed consent; dataset contains no personal or sensitive information.

## Dataset Structure
```
Neapolitan-Spoken-Corpus/
├── audioData/
│   ├── 002.m4a
│   ├── 003.m4a
│   ├── ...
│   └── 142.m4a
├── code/
│   ├── generate_json.py
│   ├── transcribe_whisper.py
│   └── evaluate_metrics.py
├── .gitattributes
├── README.md
├── requirements.txt
└── transcripts.csv
```

## Intended Uses & Limitations

The dataset is primarily intended for evaluating and developing ASR systems that support dialectal languages, particularly those with minimal computational resources. It provides a benchmark for dialect-aware speech recognition and can also support linguistic research in computational dialectology and language preservation.

## How to Use

To use this dataset and its associated scripts:

```bash
# Clone repository
git clone https://huggingface.co/datasets/anonymous-nsc-author/neapolitan-spoken-corpus
cd neapolitan-spoken-corpus
# Install dependencies
pip install -r requirements.txt
# (Optional) Generate sentences.json
python code/generate_json.py
# Transcribe audio files with Whisper ASR (requires OPENAI_API_KEY)
export OPENAI_API_KEY=your-key-here
python code/transcribe_whisper.py
# Evaluate transcription accuracy metrics (WER, BLEU, etc.)
python code/evaluate_metrics.py
```

## Evaluation Results

The dataset was evaluated using OpenAI's Whisper model with the language set to Standard Italian. The results indicate significant performance degradation on Neapolitan dialect speech:

| Metric                  | Mean  | Std Dev | Min    | Max    |
|-------------------------|-------|---------|--------|--------|
| WER (1 - WER similarity)| 0.1306| 0.1654  | 0.0000 | 0.9091 |
| Levenshtein (normalized)| 0.6360| 0.1375  | 0.0870 | 0.9804 |
| BLEU                    | 0.0436| 0.0961  | 0.0000 | 0.8932 |
| Jaccard                 | 0.1078| 0.1294  | 0.0000 | 0.8333 |

## Ethical Considerations

All participants involved in creating this dataset provided explicit informed consent. Audio and transcription data include no sensitive, private, or personally identifiable information.
