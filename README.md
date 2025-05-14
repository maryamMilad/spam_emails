# Spam/Ham Email Classification Project

## Overview
This project implements a machine learning model to classify emails as either "spam" or "ham" (non-spam). The model uses Natural Language Processing (NLP) techniques to process email text and a Random Forest classifier to make predictions.

## Dataset
The dataset used is `spam_ham_dataset.csv` which contains:
- 5,171 email messages
- Each message is labeled as "spam" or "ham"
- The dataset includes:
  - Original text of the email
  - Label (spam/ham)
  - Numerical representation of the label (0 for ham, 1 for spam)

## Features
- Text preprocessing including:
  - Lowercasing
  - Punctuation removal
  - Stopword removal (except "not")
  - Porter stemming
- Feature extraction using CountVectorizer (with 42,500 max features)
- Random Forest classification model

## Model Performance
The model achieved excellent performance metrics:
- Accuracy: 97.2%
- Precision: 95.1%
- Recall: 95.4%
- F1 Score: 95.2%

## Requirements
To run this project, you'll need:
- Python 3.x
- Required libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - nltk

## Installation
1. Clone this repository
2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

## Usage
1. Place your dataset (`spam_ham_dataset.csv`) in the project directory
2. Run the Jupyter notebook or Python script
3. The script will:
   - Load and preprocess the data
   - Train the Random Forest classifier
   - Evaluate model performance
   - Save predictions to `submission.csv`

## File Structure
- `spam_ham_dataset.csv`: Input dataset
- `submission.csv`: Output predictions
- Jupyter notebook/Python script: Main implementation file

## Future Improvements
- Experiment with other classifiers (SVM, Naive Bayes)
- Try different text vectorization methods (TF-IDF, word embeddings)
- Implement more sophisticated text preprocessing
- Add hyperparameter tuning

## License
This project is open source and available under the MIT License.
