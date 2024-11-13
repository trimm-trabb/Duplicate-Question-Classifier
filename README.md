# Duplicate-Question-Classifier

This project aims to classify pairs of questions as duplicates or not, using a variety of natural language processing techniques and machine learning models.  

## Project Overview

### Data Preprocessing

The dataset consists of pairs of questions, each labeled as either duplicate or non-duplicate. To prepare this data for model input, two techniques are used:

TF-IDF (Term Frequency-Inverse Document Frequency): This method converts questions into a numerical format by quantifying the importance of each word relative to the entire dataset. TF-IDF features are then fed into traditional machine learning classifiers, such as Logistic Regression, Naive Bayes, and XGBoost.

BERT Embeddings: BERT (Bidirectional Encoder Representations from Transformers) captures deeper, contextual meaning by converting each question into dense embeddings. This approach allows the model to capture nuanced language patterns and semantic relationships between questions.

Also similarity metrics between question pairs (Cosine Similarity, Manhattan Distance, Euclidean Distance) were calculated, providing additional insights and aiding in model performance.

### Modeling Approaches
This project experiments with various models to find the best classifier for this task:

Traditional Machine Learning Models: These models work with TF-IDF features and are lightweight compared to deep learning approaches.

* Logistic Regression: A linear model that predicts the probability of a pair being duplicate, offering a baseline for comparison.

* Naive Bayes: A probabilistic classifier that assumes word independence, often effective for text-based tasks with TF-IDF features.

* XGBoost: A boosting algorithm that builds a series of weak learners to improve classification accuracy.

BERT-Based Neural Network Model: A custom neural network model built on top of BERT model was implemented. This model utilizes BERT embeddings from each question in a pair. The embeddings are fed into a fully connected classification layer, predicting whether the questions are duplicates. 

### Evaluation Metrics:
The main metric was Log Loss to understand how close the prediction probability is to the true value.

Precision, Recall, F1-Score and Confusion Matrix were also used.

### Technologies 
* Python: Core programming language for data processing and model training.
* Pandas: Data manipulation and handling, particularly for dataset management.
* NumPy: Array operations and mathematical computations.
* NLTK: Natural Language Toolkit, used for text preprocessing tasks 
* scikit-learn: Used for traditional machine learning models.
* XGBoost: Gradient boosting library.
* PyTorch: Deep learning framework used for implementing and fine-tuning the BERT-based model.
* Hugging Face Transformers: Provides pre-trained BERT models and tokenizers for embedding generation.
