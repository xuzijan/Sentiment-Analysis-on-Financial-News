# Financial News Sentiment Analysis Project

## Project Overview

This project employs natural language processing (NLP) and machine-learning techniques to classify the sentiment expressed in financial news headlines, determining whether they convey positive, neutral, or negative sentiment toward a company or the market. The project has completed data acquisition, preprocessing, EDA, and implemented three parallel modeling tracks (Naïve Bayes + TF-IDF, LSTM, and fine-tuned Transformer/DistilBERT), comprehensive multi-metric evaluation, and application to real financial news with interpretable results.

## Data Sources

### 1. **CFSC-ABSA Dataset** (Chinese)
- 15,446 Chinese financial news sentences
- 41,496 aspect terms
- Sentiment labels: Positive, Neutral, Negative
- Categories: Companies, Stocks, Markets, Economics

### 2. **Twitter Financial News Sentiment Dataset** (English)
- 11,932 tweets
- Labels: Bearish, Bullish, Neutral
- Training set: 9,940 samples
- Validation set: Additional samples

### 3. **Financial News Market Events Dataset** (English)
- 3,026 financial event news articles
- Contains market indicators, industries, companies, and other metadata

## Project Structure

```
project/
├── notebooks/
│   ├── 01_data_loading_eda.ipynb           # Data loading and EDA
│   ├── 02_data_cleaning_preprocessing.ipynb # Data cleaning and preprocessing
│   ├── 03_model_naive_bayes.ipynb          # Naive Bayes model
│   ├── 04_model_lstm.ipynb                 # LSTM deep learning model
│   ├── 05_model_transformer.ipynb          # Transformer fine-tuning
│   ├── 06_model_comparison.ipynb           # Model comparison and analysis
│   └── 07_application_new_headlines.ipynb  # Application to new headlines
├── data/
│   ├── raw/                                # Raw data
│   ├── processed/                          # Processed data
│   └── train_test/                         # Train/test datasets
├── models/
│   ├── naive_bayes_model.pkl               # Trained Naive Bayes model
│   ├── tfidf_vectorizer.pkl                # TF-IDF vectorizer
│   ├── lstm_model_best.pt                  # Best LSTM model
│   ├── word2vec_model.pkl                  # Word2Vec embeddings
│   └── transformer_model_final/            # Fine-tuned Transformer model
├── results/
│   ├── predictions/                        # Model predictions (CSV/JSON)
│   ├── visualizations/                     # Analysis charts and plots
│   ├── new_headlines_predictions.csv       # New headlines predictions
│   └── new_headlines_predictions.json      # New headlines results (JSON)
├── requirements.txt
├── PROJECT_SUMMARY.md                      # Project completion summary
└── README.md
```

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Jupyter Notebooks
```bash
jupyter notebook
```
Execute the notebooks in the `notebooks/` directory in the following order:

1. `01_data_loading_eda.ipynb` - Data loading and exploration
2. `02_data_cleaning_preprocessing.ipynb` - Text preprocessing
3. `03_model_naive_bayes.ipynb` - Naive Bayes + TF-IDF (67.32% accuracy)
4. `04_model_lstm.ipynb` - LSTM + Word2Vec (60.25% accuracy)
5. `05_model_transformer.ipynb` - Transformer/DistilBERT (73.28% accuracy) ⭐
6. `06_model_comparison.ipynb` - Comprehensive model comparison
7. `07_application_new_headlines.ipynb` - Apply best model to new headlines
