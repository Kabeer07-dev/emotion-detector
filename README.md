# Emotion Detector

A text-based emotion classification project that predicts the emotion behind a sentence — **sadness, anger, love, surprise, fear,** or **joy** — using classic NLP preprocessing and Logistic Regression.

## Overview

Given a piece of text, the model predicts which of six emotions it expresses. The full pipeline — data cleaning, feature extraction, model training, and evaluation — is implemented in a single Jupyter notebook, `emotion_detector.ipynb`.

## Dataset

- **File:** `train.txt`
- **Format:** `text;emotion` (semicolon-separated, no header), ~16,000 rows
- **Emotions:** `sadness`, `anger`, `love`, `surprise`, `fear`, `joy`

Example rows:
```
i didnt feel humiliated;sadness
im grabbing a minute to post i feel greedy wrong;anger
i am ever feeling nostalgic about the fireplace i will know that it is still on the property;love
```

Label distribution:

| Emotion  | Count |
|----------|-------|
| joy      | 5,362 |
| sadness  | 4,666 |
| anger    | 2,159 |
| fear     | 1,937 |
| love     | 1,304 |
| surprise | 572   |

## Pipeline

1. **Load data** with `pandas`, using `;` as the separator.
2. **Encode labels** — map each emotion string to an integer id.
3. **Clean text**:
   - Lowercase everything
   - Strip digits
   - Strip non-ASCII characters (emojis, etc.)
   - Remove punctuation
   - Remove English stopwords (via `nltk`)
4. **Split data** — 80% train / 20% test (`train_test_split`, `random_state=42`).
5. **Vectorize text** two ways for comparison:
   - `CountVectorizer` (Bag of Words)
   - `TfidfVectorizer` (TF-IDF)
6. **Train** a `LogisticRegression` classifier on each feature set.
7. **Evaluate** with `accuracy_score` and run sample predictions on new sentences.

## Results

| Feature extraction | Test accuracy |
|---------------------|---------------|
| Bag of Words (CountVectorizer) | **88.97%** |
| TF-IDF | 86.28% |

Bag of Words slightly outperformed TF-IDF on this dataset.

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn (`CountVectorizer`, `TfidfVectorizer`, `LogisticRegression`, `train_test_split`, `accuracy_score`)
- nltk (stopword removal)
- Jupyter Notebook

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn nltk jupyter
```

### Run

1. Clone the repo:
   ```bash
   git clone https://github.com/Kabeer07-dev/emotion-detector.git
   cd emotion-detector
   ```
2. Launch the notebook:
   ```bash
   jupyter notebook emotion_detector.ipynb
   ```
3. Run all cells. The first NLTK-related cell downloads the stopwords corpus automatically:
   ```python
   import nltk
   nltk.download("stopwords")
   ```

### Example Prediction

```python
texts = [
    "I am so excited!",
    "I hate this.",
    "I feel lonely.",
    "Everything is amazing."
]

vectors = bow.transform(texts)
predictions = model.predict(vectors)

for text, emotion in zip(texts, predictions):
    print(f"{text} --> {emotion}")
```

## Project Structure

```
emotion-detector/
├── emotion_detector.ipynb   # Full pipeline: preprocessing, training, evaluation
├── train.txt                # Labeled dataset (text;emotion)
└── README.md
```

## Possible Improvements

- Map predicted integer labels back to emotion names for readable output
- Try additional models (SVM, Naive Bayes, or a neural network / transformer-based classifier)
- Use n-grams or word embeddings (Word2Vec, GloVe) instead of BoW/TF-IDF
- Add a proper validation set / cross-validation and a confusion matrix for per-class performance
- Wrap the trained model in a simple script or API for real-time predictions

## 👤 Developer

**Muhammad Kabeer Jawed**
Feel free to connect or reach out with questions/suggestions.


## License

No license specified yet — consider adding one (e.g., MIT) if you plan to share or accept contributions.