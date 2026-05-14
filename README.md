# 🎙️ Deepfake Audio Detection

## Overview

This project detects AI-generated (deepfake) audio using the FoR (Fake-or-Real) Dataset and a pretrained wav2vec2 model. Instead of training from scratch, we load a fine-tuned model directly from HuggingFace and evaluate it on the FoR dataset with an optimized decision threshold.

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 82.60% |
| F1-Score | 0.8129 |
| ROC-AUC | 0.9286 |
| Best Threshold | 0.30 |

## Dataset

The Fake-or-Real (FoR) Dataset — for-norm split

| Split | Fake | Real | Total |
|-------|------|------|-------|
| Training | 26,927 | 26,941 | 53,868 |
| Validation | 5,398 | 5,400 | 10,798 |
| Testing | 2,370 | 2,264 | 4,634 |

- Real: Genuine human speech recordings
- Fake: AI-generated TTS (Text-to-Speech) audio

## Model

| Detail | Value |
|--------|-------|
| Model | Gustking/wav2vec2-large-xlsr-deepfake-audio-classification |
| Architecture | Wav2Vec2ForSequenceClassification |
| Base | facebook/wav2vec2-large-xlsr-53 |
| Task | Binary Classification (Real / Fake) |
| Source | HuggingFace |
