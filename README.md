# 📧 Email Spam Classifier

A machine learning project to classify emails as **spam** or **not spam** using natural language processing (NLP) techniques and a Naive Bayes classifier.

# 📝 Introduction

Email remains one of the most widely used communication tools. 
However, it's also a popular target for spammers to send unsolicited, irrelevant, or malicious messages. 
This not only clutters inboxes but also poses security risks. 
The goal of this project is to build a machine learning model that can automatically classify emails as **spam** or **ham (not spam)** by learning patterns in their textual content.

# 🚀 Problem Statement

Develop a machine learning pipeline that:

- Cleans and processes raw email text.
- Converts text into numerical features using **Bag-of-Words**.
- Trains a classification model to detect spam messages.
- Evaluates model performance on test data.
- Can be deployed or integrated into an email system.

# 📂 Dataset

The dataset used is the **SMS Spam Collection Dataset**, available from:

📥 [Download from Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

- Total entries: ~5,572 SMS messages.
- Columns:
  - `v1`: Label (`ham` or `spam`)
  - `v2`: Message text

# 🧠 Algorithms Used

- **Text Vectorization**: `CountVectorizer` (Bag-of-Words)
- **Classifier**: `Multinomial Naive Bayes`

# 🛠️ Tech Stack

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib (for visualizations)
- Jupyter Notebook (for EDA)

# 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

# 🧪 How to Run

1. Clone the Repository

```bash
git clone https://github.com/yourusername/email-spam-classifier.git
cd email-spam-classifier

2. Install Requirements
pip install -r requirements.txt

3. Run the Classifier Notebook
jupyter notebook spam_classifier.ipynb
Or run Python script:
python main.py
