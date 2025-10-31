# Customer_Support_Ticket_Classification_System
Automate your customer support workflow with machine learning-powered ticket classification. This system intelligently categorizes support tickets into bugs, feature requests, and queries, reducing manual routing time by up to 70%.
# 🎯 Overview
An end-to-end text mining and machine learning solution that automatically classifies customer support tickets using advanced NLP techniques and Random Forest algorithms. The system achieves 79.3% overall accuracy with outstanding performance in bug detection (82.9% sensitivity, 0.935 AUC).
# ✨ Key Features

- Real-Time Web Scraping: Collects training data from Reddit, StackOverflow, and Hacker News
- Advanced Text Mining: Complete preprocessing pipeline with stemming, lemmatization, and n-gram analysis
- Dual Classification Models: Naive Bayes and Random Forest with comprehensive evaluation
- Pattern Extraction: Automatic detection of error codes, product names, and version numbers
- ROC Analysis: One-vs-Rest multi-class evaluation with AUC scoring
- Feature Importance: Interpretable insights into classification decisions

# 🚀 Quick Start
- Prerequisites
- rR >= 4.0.0
- RStudio (recommended)
- Installation

Install dependencies:

- rsource("install_packages.R")

Run the complete pipeline:

rsource("Text_analytics_ca4.R")
```

## 📁 Project Structure

ticket-classifier/
├── Text_analytics_ca4.R          # Main classification pipeline
├── README.md                      # Documentation
├── requirements.txt               # R package dependencies
├── ticket_data.rds               # Saved processed data
└── ticket_classifier_model.rds   # Trained Random Forest model
```

## 🔧 Pipeline Architecture
```
Web Scraping → Text Preprocessing → Feature Extraction → 
Model Training → Evaluation → Deployment
1. Data Collection (293 tickets)

Reddit r/techsupport (100 tickets)
StackOverflow Python tag (100 tickets)
Hacker News (93 tickets)

2. Text Preprocessing

HTML tag removal
URL and email filtering
Stopword removal (53.7% noise reduction)
Stemming and lemmatization
N-gram generation (bigrams, trigrams)

3. Feature Engineering

Bag-of-Words model: 970 features
96.93% sparsity (typical for text data)
TF-IDF weighting
Pattern extraction (error codes, products, versions)

4. Classification

Training/Test Split: 70/30 stratified sampling
Random Forest: 100 trees, Gini importance
Naive Bayes: Multinomial distribution
Categories: Bug, Feature Request, Query

📈 Key Findings
Top Predictive Features

issu (issue) - Bug indicator
problem - Bug indicator
tri (try) - Troubleshooting keyword
error - Bug indicator
like - Feature request indicator

Token Reduction

Initial tokens: 38,057
After preprocessing: 17,602 (53.7% reduction)
Meaningful features: 970

Temporal Patterns

74.7% of tickets submitted on weekdays
Peak day: Tuesday (17.4%)
Lowest activity: Saturday (11.9%)

🎓 Technical Implementation
Core Technologies

Language: R 4.0+
Text Mining: tm, tidytext, SnowballC, textstem
Machine Learning: randomForest, e1071, caret
Web Scraping: httr, rvest, jsonlite
Visualization: ggplot2, pROC

Algorithm Details
Random Forest Classifier
rntree = 100
importance = TRUE
method = Gini impurity
Text Preprocessing
r# Pipeline using magrittr pipes
text %>%
  str_replace_all("<[^>]+>", " ") %>%
  str_to_lower() %>%
  str_replace_all("[^a-z0-9\\s]", " ") %>%
  str_trim()
📊 Evaluation Metrics
Confusion Matrix (Random Forest)

True Positives: Strong diagonal performance
Bug recall: 82.9%
Query recall: 90.9%
Feature Request: 0% (insufficient training data)

ROC Curve Analysis

One-vs-Rest multi-class approach
Mean AUC: 0.880 (Excellent)
All classes > 0.80 threshold

💡 Use Cases

Customer Support Automation: Reduce manual ticket routing by 60-70%
Priority Queue Management: Fast-track critical bugs
Resource Allocation: Route tickets to specialized teams
Response Time Optimization: Improve customer satisfaction
Support Analytics: Identify common issue patterns

🚧 Limitations

Class Imbalance: Feature requests underrepresented (9.6%)
Language: English only
Context: Limited to technical support domains
Short Text: Minimal tickets lack sufficient context

🔮 Future Enhancements

 BERT/Transformer integration for semantic understanding
 Multi-label classification support
 SMOTE for class balancing
 Real-time streaming with Apache Kafka
 Multilingual support
 Priority prediction beyond categorization
 Sentiment analysis integration
 Auto-summarization for long tickets

📚 Dependencies
rreadtext, rvest, httr, jsonlite, dplyr, magrittr, 
stringr, tidytext, tm, SnowballC, textstem, tidyr, 
caret, randomForest, e1071, pROC, ggplot2
