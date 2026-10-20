# Airline Sentiment Analysis — ML vs BERT vs LLM

An end-to-end NLP project comparing traditional machine learning, fine-tuned BERT, and OpenAI LLM-based sentiment classification, with an interactive Tableau dashboard.

## Project Overview

Analyzed 14,640 airline tweets from the Twitter US Airline Sentiment dataset to classify customer feedback into three categories: **negative, neutral, and positive**.

The objective was to compare three NLP approaches and evaluate how effectively they identify customer sentiment.

## Technologies Used

- **Programming:** Python, Pandas, NumPy
- **Machine Learning:** Scikit-learn, TF-IDF, Logistic Regression
- **Deep Learning:** Hugging Face Transformers, PyTorch, BERT
- **Generative AI:** OpenAI API, prompt-based sentiment classification
- **Evaluation:** Accuracy, Precision, Recall, Macro F1, Confusion Matrix
- **Visualization:** Matplotlib, Tableau Public
- **Development:** Google Colab, Jupyter Notebook

## Dataset

**Source:** [Twitter US Airline Sentiment — Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

The dataset contains 14,640 tweets with the following sentiment distribution:

| Sentiment | Percentage |
|---|---:|
| Negative | 62.69% |
| Neutral | 21.17% |
| Positive | 16.14% |

Because the dataset is imbalanced, Macro F1 was used alongside accuracy to evaluate performance fairly across all three classes.

## Methodology

### 1. Traditional Machine Learning

- Converted tweet text into TF-IDF numerical features.
- Trained a Logistic Regression classifier.
- Evaluated accuracy, Macro F1, and classification errors.

### 2. BERT Fine-Tuning

- Used the pretrained `bert-base-uncased` model.
- Tokenized tweets with padding, truncation, and attention masks.
- Fine-tuned BERT for three-class sentiment classification.
- Evaluated performance on held-out test data.

### 3. OpenAI LLM Classification

- Used the OpenAI API to classify tweets as negative, neutral, or positive.
- Developed a prompt requesting a single sentiment label.
- Evaluated the LLM on 300 sampled test tweets.

## Model Performance Comparison

All three models were evaluated on the same **300-tweet test subset**.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 78.33% | 0.6953 |
| Fine-Tuned BERT | 84.00% | 0.7888 |
| OpenAI LLM | **85.00%** | **0.8032** |

The LLM achieved the highest observed accuracy and Macro F1 on this evaluation subset
