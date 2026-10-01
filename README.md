# Tokopedia Review Sentiment Analysis

## Overview

This project implements an end-to-end Natural Language Processing (NLP) pipeline to collect, preprocess, analyze, and classify customer reviews of the Tokopedia mobile application from the Google Play Store into three sentiment categories: Positive, Neutral, and Negative. The classification is performed using Deep Learning / Multi-Layer Perceptron (MLP) neural network architectures combined with TF-IDF vectorization and SMOTE oversampling.

---

## Dataset Description

The dataset was scraped directly from Google Play Store using the `google-play-scraper` library.

- Source Application: Tokopedia (`com.tokopedia.tkpd`)
- Total Cleaned Samples: 8,447 reviews
- Date Range: October 9, 2025 to February 27, 2026
- Average Review Length: 101 characters
- Labeling Mapping:
  - Positive: Rating 4 and 5
  - Neutral: Rating 3
  - Negative: Rating 1 and 2

### Class Distribution

| Sentiment | Rating Stars | Sample Count | Percentage |
| :--- | :--- | :--- | :--- |
| Negative | 1 - 2 (3,832 / 563) | 4,395 | 52.03% |
| Positive | 4 - 5 (448 / 3,050) | 3,498 | 41.41% |
| Neutral | 3 (554) | 554 | 6.56% |
| Total | - | 8,447 | 100.00% |

---

## Project Structure

```text
submission/
|-- dataset_tokopedia.csv         # Scraped and labeled review dataset
|-- pelatihan_model .ipynb        # Jupyter notebook for preprocessing, training, and evaluation
|-- requirements.txt              # Python library dependencies
|-- scraping.py                   # Script for scraping reviews from Google Play Store
|-- README.md                     # Project documentation
```

---

## How to Run

### 1. Prerequisites and Environment Setup

Ensure Python 3.10+ is installed on your system. It is recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Scraping Data (Optional)

The dataset is already provided as `dataset_tokopedia.csv`. If you wish to scrape new or additional data, execute:

```bash
python scraping.py
```

This script scrapes the latest reviews, cleans duplicates, drops missing entries, assigns sentiment labels based on ratings, and outputs `dataset_tokopedia.csv`.

### 3. Running the Training and Evaluation Notebook

You can run the notebook locally using Jupyter Notebook, JupyterLab, VS Code, or upload it to Google Colab.

```bash
# Launch Jupyter Notebook
jupyter notebook "pelatihan_model .ipynb"
```

In Google Colab:
1. Open Google Colab (https://colab.research.google.com).
2. Upload `pelatihan_model .ipynb` and `dataset_tokopedia.csv`.
3. Execute the cells sequentially from top to bottom.

### 4. Running Inference

The notebook includes an inference function that loads the saved model bundle (`model_tokopedia_sentimen.pkl`):

```python
import pickle
import re

# Load model bundle
with open('model_tokopedia_sentimen.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
vectorizer = bundle['vectorizer']
label_encoder = bundle['label_encoder']

def predict_sentiment(text):
    # Preprocessing
    cleaned = text.lower()
    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    vec = vectorizer.transform([cleaned]).toarray()
    pred = model.predict(vec)
    prob = model.predict_proba(vec)
    
    sentiment = label_encoder.inverse_transform(pred)[0]
    confidence = max(prob[0]) * 100
    return sentiment, confidence

# Example
text = "Aplikasi Tokopedia sangat membantu, belanja jadi mudah dan cepat!"
sentiment, confidence = predict_sentiment(text)
print(f"Sentiment: {sentiment} ({confidence:.2f}%)")
```

---

## Experimental Evaluation Results

Three distinct training schemes were designed, trained, and evaluated. Each scheme combined a different neural network configuration, text feature representation, and train/test split. SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training set across all schemes to address class imbalance.

### Scheme 1: MLP with Word n-grams (80:20 Split)

- Model Architecture: Multi-Layer Perceptron (512, 256, 128 hidden units), ReLU activation, Adam optimizer, alpha = 0.001, adaptive learning rate
- Feature Extraction: TF-IDF Word-level (1-3 gram), max_features = 20,000, min_df = 2, max_df = 0.95, sublinear_tf = True (13,005 extracted features)
- Split: 80% Train (6,757 samples, 10,548 after SMOTE), 20% Test (1,690 samples)
- Training Time: 223.3 seconds (18 iterations)
- Train Accuracy: 97.15%
- Test Accuracy: 79.59%
- Weighted F1-Score: 78.46%

#### Classification Report (Scheme 1)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Negatif | 0.80 | 0.89 | 0.84 | 879 |
| Netral | 0.17 | 0.09 | 0.12 | 111 |
| Positif | 0.85 | 0.79 | 0.82 | 700 |
| **Accuracy** | | | **0.80** | **1690** |
| **Macro Avg** | **0.60** | **0.59** | **0.59** | **1690** |
| **Weighted Avg** | **0.78** | **0.80** | **0.78** | **1690** |

---

### Scheme 2: Deep MLP with Character n-grams (80:20 Split)

- Model Architecture: Deep Multi-Layer Perceptron (512, 256, 128, 64 hidden units), ReLU activation, Adam optimizer, alpha = 0.0005, adaptive learning rate, batch_size = 64
- Feature Extraction: TF-IDF Character-level within word boundaries `char_wb` (3-5 gram), max_features = 20,000, sublinear_tf = True (20,000 extracted features)
- Split: 80% Train (6,757 samples, 10,548 after SMOTE), 20% Test (1,690 samples)
- Training Time: 1002.8 seconds (20 iterations)
- Train Accuracy: 98.32%
- Test Accuracy: 79.47%
- Weighted F1-Score: 78.83%

#### Classification Report (Scheme 2)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Negatif | 0.81 | 0.86 | 0.83 | 879 |
| Netral | 0.22 | 0.15 | 0.18 | 111 |
| Positif | 0.83 | 0.82 | 0.83 | 700 |
| **Accuracy** | | | **0.79** | **1690** |
| **Macro Avg** | **0.62** | **0.61** | **0.61** | **1690** |
| **Weighted Avg** | **0.78** | **0.79** | **0.79** | **1690** |

---

### Scheme 3: MLP with Tanh Activation (70:30 Split)

- Model Architecture: Multi-Layer Perceptron (384, 192, 96 hidden units), Tanh activation, Adam optimizer, alpha = 0.001, adaptive learning rate
- Feature Extraction: TF-IDF Word-level (1-3 gram), max_features = 20,000, min_df = 2, sublinear_tf = True (11,372 extracted features)
- Split: 70% Train (5,912 samples, 9,228 after SMOTE), 30% Test (2,535 samples)
- Training Time: 124.6 seconds (17 iterations)
- Train Accuracy: 92.54%
- Test Accuracy: 81.10%
- Weighted F1-Score: 80.24%

#### Classification Report (Scheme 3)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Negatif | 0.82 | 0.89 | 0.85 | 1319 |
| Netral | 0.15 | 0.10 | 0.12 | 166 |
| Positif | 0.86 | 0.83 | 0.85 | 1050 |
| **Accuracy** | | | **0.81** | **2535** |
| **Macro Avg** | **0.61** | **0.60** | **0.61** | **2535** |
| **Weighted Avg** | **0.80** | **0.81** | **0.80** | **2535** |

---

### Summary Comparison of All Schemes

| Scheme | Architecture | Feature Extractor | Split | Train Accuracy | Test Accuracy | Weighted F1-Score | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Scheme 1 | MLP (512-256-128) ReLU | TF-IDF Word (1-3 gram) | 80:20 | 97.15% | 79.59% | 78.46% | 223.3s |
| Scheme 2 | Deep MLP (512-256-128-64) ReLU | TF-IDF Char (3-5 gram) | 80:20 | 98.32% | 79.47% | 78.83% | 1002.8s |
| Scheme 3 | MLP (384-192-96) Tanh | TF-IDF Word (1-3 gram) | 70:30 | 92.54% | 81.10% | 80.24% | 124.6s |

---

## Conclusion

1. **Best Performing Model**: Scheme 3 (MLP 384-192-96 with Tanh activation and word-level TF-IDF on a 70:30 split) achieved the best generalization, with a Test Accuracy of 81.10% and a Weighted F1-Score of 80.24%. It also proved to be the most computationally efficient model, completing training in only 124.6 seconds.
2. **Impact of Activation Function and Model Complexity**:
   - Scheme 1 (512-256-128 ReLU) and Scheme 2 (512-256-128-64 ReLU) exhibited higher training accuracy (97.15% and 98.32%), but lower test accuracy (~79.5%), demonstrating noticeable overfitting.
   - Scheme 3, utilizing Tanh activation and a more compact architecture, regulated weight updates more smoothly, reducing the generalization gap between train (92.54%) and test (81.10%) sets.
3. **Feature Extraction Efficiency**:
   - Character n-grams (Scheme 2) dramatically increased training duration to 1002.8 seconds due to high feature dimensionality and dense representation, yet produced test performance comparable to Scheme 1 (79.47% vs 79.59%). Word-level n-grams proved to be far more resource-efficient for this task.
4. **Challenge with the Neutral Class**:
   - Across all schemes, classification performance for the neutral class remained low (F1-score 0.12 - 0.18). This occurs because neutral reviews represent only 6.56% of the dataset (554 samples) and often contain mixed, subjective, or ambiguous expressions that overlap with positive and negative sentiments. Even with SMOTE oversampling in training, real test data retains this imbalance.
5. **Recommendations for Future Improvements**:
   - Implementing contextualized language models such as IndoBERT or RoBERTa to capture semantic context and colloquial Indonesian slang more accurately.
   - Refactoring the rating-to-sentiment mapping or applying cost-sensitive learning / focal loss to better distinguish neutral nuances.
   - Expanding the dataset with more verified neutral reviews to mitigate extreme class distribution disparity.
