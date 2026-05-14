# 🎙️ Deepfake Audio Detection

> Binary classification of audio files as **Real** or **Fake** using a pretrained Wav2Vec2 model — no training required.

---

## 📌 Overview

This project detects AI-generated (deepfake) audio using the **FoR (Fake-or-Real) Dataset** and a pretrained `wav2vec2` model. Instead of training from scratch, we load a fine-tuned model directly from HuggingFace and evaluate it on the FoR dataset with an optimized decision threshold.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | 82.60% |
| F1-Score | 0.8129 |
| ROC-AUC | 0.9286 |
| Best Threshold | 0.30 |

---

## 🗂️ Dataset

**The Fake-or-Real (FoR) Dataset** — `for-norm` split

| Split | Fake | Real | Total |
|-------|------|------|-------|
| Training | 26,927 | 26,941 | 53,868 |
| Validation | 5,398 | 5,400 | 10,798 |
| Testing | 2,370 | 2,264 | 4,634 |

- **Real**: Genuine human speech recordings
- **Fake**: AI-generated TTS (Text-to-Speech) audio

---

## 🤖 Model

| Detail | Value |
|--------|-------|
| Model | `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification` |
| Architecture | `Wav2Vec2ForSequenceClassification` |
| Base | `facebook/wav2vec2-large-xlsr-53` |
| Task | Binary Classification (Real / Fake) |
| Source | HuggingFace 🤗 |



```
📦 deepfake-audio-detection
 ┣ 📓 finalspeechproject.ipynb   # Main notebook
 ┣ 📄 README.md                  # This file
 ┗ 📊 evaluation.png             # Evaluation plots (generated at runtime)
```

### Notebook Sections

| Section | Description |
|---------|-------------|
| 1. Install & Import | Install all required libraries |
| 2. Load Dataset | Load and count FoR dataset files |
| 3. EDA | Exploratory Data Analysis (waveforms, spectrograms, class distribution) |
| 4. Load Pretrained Model | Load model from HuggingFace |
| 5. Inference Function | `predict_audio()` function with threshold |
| 6. Evaluation | Evaluate on 500 test samples |
| 7. Threshold Sweep | Find the optimal decision threshold automatically |
| 8. Plots | Confusion Matrix, ROC Curve, Score Distribution |
| 9. Test Your Own Audio | Upload any audio file and classify it |
| 10. Gradio GUI | Interactive web interface with full audio analysis |

---

## ⚙️ Installation

```bash
pip install transformers librosa soundfile torch scikit-learn \
            matplotlib seaborn tqdm ipywidgets gradio jupyter
```

---

## 🚀 Quick Start

### Run on Kaggle
1. Open `finalspeechproject.ipynb` on Kaggle
2. Add the FoR dataset from Kaggle Datasets
3. Run all cells — the model loads automatically from HuggingFace
4. Use the Gradio GUI cell to launch the web interface

### Run Locally
```bash
# Install dependencies
pip install transformers librosa soundfile torch scikit-learn matplotlib seaborn tqdm gradio jupyter

# Launch Jupyter
jupyter notebook finalspeechproject.ipynb
```

> ⚠️ **Note:** The model (~1GB) requires internet access on the **first run only**. After that it is cached locally and works offline.

---

## 🖥️ Gradio GUI

The notebook includes a full interactive GUI built with **Gradio** that provides:

- 📁 Audio file upload (`.wav`, `.mp3`, `.ogg`)
- 🌊 Waveform visualization
- 🎨 Mel Spectrogram
- 📊 MFCC (40 coefficients)
- 📈 Spectral Centroid
- ⚡ Zero Crossing Rate
- 🎯 Real / Fake prediction with confidence scores

```python
demo.launch(share=True)  # generates a public link
```

---

## 🔧 Key Design Decisions

### Pretrained Model (No Training)
Training `wav2vec2-large` from scratch takes **3-4 hours** on a T4 GPU. By using a pretrained model, the notebook runs in **~17 minutes** with comparable accuracy.

### Threshold Optimization
Default threshold (0.5) gave 77% accuracy. After running a threshold sweep from 0.1 to 0.9, the optimal threshold of **0.30** improved accuracy to **82.6%**.

```python
# Threshold sweep
thresholds = np.arange(0.1, 0.9, 0.05)
# Best threshold found: 0.30
pred_id = 1 if probs[1] > 0.30 else 0
```

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `transformers` | Load Wav2Vec2 model |
| `librosa` | Audio loading and feature extraction |
| `torch` | Model inference |
| `gradio` | Interactive GUI |
| `scikit-learn` | Evaluation metrics |
| `matplotlib` / `seaborn` | Visualization |

---

## 📈 Evaluation Plots

The notebook generates three evaluation plots:
- **Confusion Matrix** — True vs Predicted labels
- **ROC Curve** — AUC = 0.9286
- **Score Distribution** — P(Fake) distribution for Real vs Fake files

---

## ⚠️ Limitations

- Model is trained on **English speech only** — performance may drop on other languages
- Maximum audio duration analyzed: **5 seconds**
- Optimized for TTS-generated fakes; may not detect all types of voice cloning

---

## 🔮 Future Work

- Fine-tune on the full FoR dataset for higher accuracy
- Add support for multilingual audio
- Extend to detect voice cloning and real-time audio streams
- Deploy as a standalone web application

---

## 📄 References

- [The Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
- [Wav2Vec2 — HuggingFace](https://huggingface.co/facebook/wav2vec2-large)
- [Gustking Model](https://huggingface.co/Gustking/wav2vec2-large-xlsr-deepfake-audio-classification)
