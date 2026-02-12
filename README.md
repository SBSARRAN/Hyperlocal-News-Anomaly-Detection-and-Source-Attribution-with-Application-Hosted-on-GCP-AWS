# Hyperlocal-News-Anomaly-Detection-and-Source-Attribution-with-Application-Hosted-on-GCP-AWS

📰 Hyperlocal News Anomaly Detection & Source Discrepancy System

# An end-to-end NLP system that detects:
 Linguistic anomalies in hyperlocal news
 Source location discrepancies
 Topic patterns
 Sentiment-based irregularities
 Combined anomaly scoring
 Interactive Streamlit Dashboard
 Deployed on AWS EC2
 
# Project Overview
Hyperlocal news platforms may sometimes publish articles that:
Do not match their declared source location
Contain unusual linguistic patterns
Show abnormal sentiment or thematic structure
This project builds a multi-module anomaly detection pipeline combining:
Named Entity Recognition (NER)
Sentiment Analysis
Topic Modeling (LDA)
Isolation Forest Anomaly Detection
Logistic Regression for Source Discrepancy Detection
Streamlit Interactive Dashboard
AWS Deployment

# System Architecture
Articles.xlsx
  
   
Text Cleaning & Header Location Extraction
      
      
NER Location Extraction (spaCy)
      
      
Sentiment Analysis (VADER)
      
      
Topic Modeling (LDA)
      
      
Isolation Forest (Linguistic Anomaly Detection)
      
      
TF-IDF + Logistic Regression
(Source Discrepancy Detection)
      
      
Final_Anomaly = Linguistic OR Source Mismatch
      
      
Streamlit Dashboard
      
      
AWS EC2 Hosting

# Technologies Used
| Category            | Tools               |
| ------------------- | ------------------- |
| Programming         | Python              |
| NLP                 | spaCy, NLTK         |
| Sentiment           | VADER               |
| Topic Modeling      | LDA (Sklearn)       |
| Anomaly Detection   | Isolation Forest    |
| Source Detection    | Logistic Regression |
| Feature Engineering | TF-IDF              |
| ML Tools            | Scikit-learn        |
| Dashboard           | Streamlit           |
| Deployment          | AWS EC2             |

#  Header Location Extraction
 Extracts city name from article header using regex:
 def extract_city(article):
    match = re.match(r"(?:STRONG>)?([A-Z/\s]+):", article)
Example:
  "HONG KONG: Asian markets..."
→ Header_Location = hong kong

#  Named Entity Recognition (NER)

Using spaCy to extract location entities:
doc = nlp(text)
ent.label_ in ['GPE','LOC']

Extracted entities:
Countries
Cities
Regions

# Sentiment Analysis
Using VADER:
scores = sia.polarity_scores(text)
Sentiment Labels:
Positive
Neutral
Negative

#  Linguistic Anomaly Detection
Using Isolation Forest on numeric features:
Features used:
sentiment score
topic
encoded news type
encoded sentiment label

# Scorce Discrepancy Detection (Core Module)
Goal:
Predict publication location using only article content.
Process:
Convert text → TF-IDF vectors
Train Logistic Regression model
Predict location from article
Compare with actual Header_Location

# Final Anomaly Decision
df['Final_Anomaly'] = ((df['anomaly_label'] == 'Anomaly') |(df['Source_Mismatch'] == True))
An article is flagged if:Linguistically abnormal OR Source location mismatch detected

# Streamlit Dashboard Features
Filter by:
Anomaly Type
Source Mismatch
View full article details
Explore anomaly scores
Interactive table visualization
Run App: streamlit run app.py

# AWS Deployment (EC2)
Instance Details:
Instance Type: t3.micro
OS: Amazon Linux 2023
Port: 8501 (Streamlit)
Run App: streamlit run app.py --server.address 0.0.0.0 --server.port 8501
Access Via: http://<EC2_PUBLIC_IP>:8501

# Project Structure
  Hyperlocal-News-Anomaly
 ┣  Articles.xlsx
 ┣  app.py
 ┣  final_news_output.csv
 ┣  location_model.pkl
 ┣  tfidf_vectorizer.pkl
 ┣  label_encoder.pkl
 ┣  requirements.txt
 ┗  README.md

# Sample Output 
| Header_Location | Predicted_Location | Source_Mismatch | Discrepancy_Score | Final_Anomaly |
| --------------- | ------------------ | --------------- | ----------------- | ------------- |
| karachi         | karachi            | False           | 0.37              | False         |
| new york        | singapore          | True            | 0.79              | True          |

# Key Achievements
 Built complete NLP pipeline
 Multi-model anomaly detection
 Source misattribution detection
 End-to-end ML system
 Cloud deployment on AWS
 Real-time interactive dashboard
