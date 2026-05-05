# Soft Token Prediction for Language Modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains the implementation for **"Soft Token Prediction for Language Modeling"**. 

This project investigates the behaviors of language models trained as an **item-based** learner,
a **class-first** learner, or one in between by modifying the standard autoregressive Cross-Entropy loss. 
By distributing "partial credit" ($\epsilon$) across tokens sharing the same Part-of-Speech (POS) tag,
we mitigate lexical rigidity and improve syntactic generalization in data-constrained environments.

## Key Concept: POS-Aware Loss

Standard language modeling is binary: you either predict the exact token or you are penalized.
Our approach introduces a "soft" objective:

$$\text{Loss} = (1 - \epsilon) \cdot \text{CE}(y_{gold}) + \epsilon \cdot \text{CE}(S_{POS})$$

Where $S_{POS}$ represents the set of all tokens sharing the gold label's syntactic category. This allows us to ablate between a pure item-based learner ($\epsilon=0$) and a class-only learner ($\epsilon=1.0$).

## Quick Start

### Installation
```bash
git clone [https://github.com/Aatlantise/partial-credit-lm](https://github.com/Aatlantise/partial-credit-lm)
cd partial-credit-lm
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Training
To train a model with a specific epsilon (e.g., $\epsilon=0.1$):
```bash
python train.py --mode partial --epsilon 0.1 --output_dir ./models/eps_0.1
```

### Evaluation (BLiMP)
We use the EleutherAI LM Evaluation Harness for linguistic benchmarking.
After installing EleutherAI's `lm-evaluation-harness`, run `run-eval.sh`.
