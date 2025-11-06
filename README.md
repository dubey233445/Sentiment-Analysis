# 💬 Sentiment Analysis using AI & NLP – Internship Project

**Internship Organization:** Growfinix Technology  
**Internship Domain:** Artificial Intelligence / Natural Language Processing (NLP)  
**Project Title:** Sentiment Analysis – Understanding Emotions in Text  
**Developer:** Ashish Dubey  

---

## 🧠 Project Overview
During my **Artificial Intelligence internship at Growfinix Technology**, I worked on an exciting project focused on **Sentiment Analysis** — using **AI and NLP** to understand emotions hidden in text data such as product reviews, customer feedback, or social media comments.  

The idea was simple yet powerful:  
> “Use AI to make machines understand human opinions — whether they are positive, negative, or neutral.”  

While working on this project, I realized how impactful this technology can be in transforming how businesses understand their customers and improve products based on real feedback.  

---

## 👨‍💻 What I Did
- 🧹 Cleaned and prepared raw Amazon review data (removed stopwords, punctuation, and special characters).  
- 🧠 Trained multiple ML models (Logistic Regression, Naive Bayes, SVM, Random Forest) to classify text as **positive**, **negative**, or **neutral**.  
- 📊 Compared and evaluated model performance using accuracy and F1-score.  
- 🧩 Used NLP techniques such as tokenization, stemming, lemmatization, and TF-IDF vectorization.  
- ⚙️ Evaluated and visualized model performance through confusion matrices and word clouds.  

---

## 🧰 Tech Stack & Tools

| Category | Tools / Libraries |
|-----------|------------------|
| Programming Language | Python |
| NLP Libraries | NLTK, spaCy, TextBlob |
| Machine Learning | Scikit-learn |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Environment | Jupyter Notebook / Google Colab |

---

## 📂 Dataset

The project uses the **Amazon Customer Reviews Dataset**, which contains millions of product reviews and star ratings submitted by real users on Amazon.  

- 📦 **Dataset Source:** [Amazon Product Reviews Dataset (Kaggle)](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)  
- 🧾 The dataset includes:
  - Review text
  - Star rating (1–5)
  - Product category
  - Review summary  

### Label Mapping:
To simplify training:
- ⭐ 1–2 → Negative 😡  
- ⭐ 3 → Neutral 😐  
- ⭐ 4–5 → Positive 😃  

---

## ⚙️ Project Workflow

### 1. Data Preprocessing
- Removed punctuation, numbers, and special characters.  
- Lowercased all text.  
- Removed stopwords using NLTK.  
- Applied tokenization, stemming, and lemmatization.  
- Labeled reviews based on star ratings.

### 2. Feature Extraction
Converted text data into numerical form using:
- **TF-IDF Vectorizer** (Term Frequency–Inverse Document Frequency)  
- Experimented with **Bag of Words (BoW)** model  

### 3. Model Training
Trained and compared the following models:
- Logistic Regression  
- Naive Bayes  
- Support Vector Machine (SVM)  
- Random Forest  

### 4. Model Evaluation
Evaluated model performance using:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

### 5. Visualization
- Displayed model performance using confusion matrix heatmaps.  
- Generated word clouds for positive and negative reviews.  

---

## 📈 Results & Insights
✅ **Best Performing Model:** Logistic Regression (~91% Accuracy)  
✅ **TF-IDF** outperformed Bag-of-Words in feature representation.  
✅ **Observation:** The model was highly effective at distinguishing strong positive and negative reviews, with slightly more confusion on neutral ones.  

### Key Learnings
- Hands-on experience with NLP preprocessing and text vectorization.  
- Understanding of how ML algorithms interpret textual data.  
- Insights into sentiment trends and their potential business impact.  

---

## 🌍 Real-World Applications

| Sector | Use Case |
|--------|-----------|
| 🛍️ E-Commerce | Analyze customer reviews and detect satisfaction trends. |
| 💼 Business Intelligence | Understand customer opinions to improve products and services. |
| 📰 Media Monitoring | Track public sentiment toward brands, events, or topics. |
| 💬 Social Media | Analyze user engagement and emotional tone in comments. |
| 🎬 Entertainment | Gauge audience reactions to movies, shows, or music. |

Example:  
> Imagine Amazon receiving thousands of product reviews every day. Instead of manually reading them, AI can automatically classify reviews as positive, neutral, or negative — helping businesses act faster and make data-driven decisions.

---

## 💻 How to Run the Project

### 1. Clone the Repository
```bash
git https://github.com/dubey233445/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
jupyter notebook sentiment.ipynb
```

Or run the script:
```bash
python App2.py
```

---

## 📊 Example Output

| Sample Input | Predicted Sentiment |
|---------------|--------------------|
| “Absolutely love this product!” | Positive 😃 |
| “Terrible quality, waste of money.” | Negative 😡 |
| “It’s okay, does the job.” | Neutral 😐 |

---

## 🚀 Future Enhancements
- Integrate **Deep Learning** models (LSTM, BERT) for contextual understanding.  
- Build a **Streamlit or Flask Web App** for live sentiment prediction.  
- Add **multilingual sentiment support** for global datasets.  
- Deploy model via **REST API** or **AWS Lambda** for real-time use.  

---

## 🙌 Acknowledgement
Big thanks to **Growfinix Technology** for the guidance and mentorship throughout this internship.  
This project gave me hands-on exposure to AI, ML, and NLP — and more importantly, helped me see how these technologies create real-world impact. 🙌  

> “AI isn’t just about automation — it’s about understanding human emotion at scale.”  

---

## 🧩 Author
**Ashish Dubey**  
AI Intern @ Growfinix Technology  
📧 [dubeyashish8957@gmail.com]  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/ashish-dubey-8904a52b3/)  
🐙 [GitHub Profile](https://github.com/dubey233445)

---

⭐ *If you found this project helpful, please star the repository on GitHub!* ⭐
