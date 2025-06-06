📌 Overview
Sarcasm detection in text is a challenging task due to its context-sensitive and nuanced nature. This project aims to build a robust machine learning model to accurately classify text as sarcastic or non-sarcastic using traditional ML algorithms and ensemble learning techniques.

👨‍🎓 Presented By
Name: Chirunomula Vamshi Krishna Babu
College: Mahindra University
Department: Artificial Intelligence
Email: chirunomulavamshikrishnababu@gmail.com
AICTE ID: STU67c7e49f922511741153439

📍 Problem Statement
Traditional NLP systems often misclassify sarcastic statements, which negatively affects applications like sentiment analysis, chatbots, and social media monitoring. This project addresses this issue using various classification models.

💡 Proposed Solution
Dataset: Kaggle dataset with 26,709 labeled headlines (Sarcasm or Not Sarcasm)

Preprocessing:

Label transformation (binary to textual)

Text vectorization using CountVectorizer and TF-IDF

Models Used:

Bernoulli Naive Bayes (Baseline)

Logistic Regression

Random Forest

Support Vector Machine (SVM) with Linear Kernel

Ensemble Voting Classifier (Best Performance)

Deployment: Interactive input to test real-time predictions (e.g., "Cows lose their jobs as milk prices drop" → Sarcasm)

🔧 System Approach
Libraries Used
python
Copy
Edit
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
Workflow
Text Vectorization

Train-Test Split

Model Training

Model Evaluation

🤖 Algorithms & Accuracy
Algorithm	Accuracy
Bernoulli Naive Bayes	84.48%
Logistic Regression	84.54%
Voting Classifier	85.38%

Classification Report:

Class	Precision	Recall	F1-Score
Not Sarcasm	0.85	0.89	0.87
Sarcasm	0.85	0.81	0.83

📊 Sample Output
Input: "Cows lose their jobs as milk prices drop"
Prediction: Sarcasm ✅

✅ Conclusion
The ensemble model gave the best results (85.38% accuracy).

Key challenge: Identifying sarcasm without deep contextual understanding.

🔭 Future Scope
Integrate contextual embeddings like BERT/GPT for better performance.

Extend dataset to include multilingual sarcasm.

Deploy the model as a web application or API.

