# Customer Review Sentiment Analysis

## Project Overview
An NLP and machine learning project that classifies customer reviews into three sentiment categories: positive, neutral, and negative.

## Problem Statement
Manually analyzing large volumes of customer feedback is time-consuming. This project explores automated sentiment classification to help understand customer opinions.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Hugging Face Datasets
- Joblib
- Google Colab

## Project Workflow
1. Load the customer review dataset.
2. Inspect data quality and sentiment labels.
3. Clean and normalize text.
4. Convert text into numerical features using TF-IDF.
5. Split data into training and testing sets.
6. Train a Logistic Regression classifier.
7. Evaluate model performance.
8. Test predictions on new reviews.
9. Save and reload the trained model.

## Dataset
Hugging Face dataset: `jbeno/sentiment_merged`

Sentiment classes:
- Negative
- Neutral
- Positive

## Results

### Original Logistic Regression
- Accuracy: 67.07%
- Macro F1-score: 0.62

### Class-weighted Logistic Regression
- Accuracy: 65.64%
- Macro F1-score: 0.64
- Negative sentiment recall improved from 41% to 61%.

The class-weighted model improved detection of negative reviews, while overall accuracy decreased slightly.

## Future Improvements
- Experiment with word and phrase features (n-grams).
- Compare additional machine learning algorithms.
- Explore pretrained transformer models.
- Build a simple web application for predictions.

## How to Run
Install the dependencies listed in `requirements.txt`, open the notebook in Google Colab or Jupyter, and run its cells in order.

## Project Status
Baseline NLP sentiment classification and evaluation completed.
