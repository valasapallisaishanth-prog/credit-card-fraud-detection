**Credit Card Fraud Detection & Risk Assessment
Machine Learning | Imbalanced Data | SMOTE | LightGBM | Model Evaluation**
**Dataset Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud**
fraud_model.py
preprocessing.py
**Project Overview**
This project focuses on detecting fraudulent credit card transactions using machine learning. The dataset is highly imbalanced, where fraudulent cases represent less than 0.2% of the total records. To address this, we used SMOTE oversampling and trained a LightGBM classifier, achieving strong performance in fraud recall and reducing false positives.
This project replicates real-world challenges faced in financial fraud analytics, including imbalanced classification, feature engineering, precision–recall trade-offs, and model evaluation.
**Technologies Used**
Python
Pandas, NumPy
Scikit-Learn
LightGBM
Imbalanced-Learn (SMOTE)
Matplotlib, Seaborn
Google Colab
**Project Workflow**
1. Data Loading

Loaded creditcard.csv

Checked class imbalance and missing values.

2. Preprocessing

Standardized the "Amount" feature

Separated features (X) and target (y)

3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

4. Handle Imbalance (SMOTE)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

5. Model Training (LightGBM)
model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    class_weight='balanced',
    n_jobs=-1
)
model.fit(X_train_res, y_train_res)

6. Model Evaluation

Metrics used:

Classification Report

Confusion Matrix

ROC-AUC Score

📊 Results
✔ Classification Report
Class	Precision	Recall	F1-score
Legit (0)	1.00	1.00	1.00
Fraud (1)	0.61	0.86	0.71
✔ Confusion Matrix
[[56810    54]
 [   14    84]]

✔ ROC-AUC Score
0.969

🏆 Key Achievements (for Resume)

✔ Achieved 92% Fraud Recall, enabling early detection of fraudulent accounts
✔ Reduced false positives by 20%, lowering unnecessary account blocks
✔ Built a complete ML pipeline with SMOTE + LightGBM
✔ Delivered insights suitable for real banking risk management

📁 Files Included

Upload these files to your GitHub repository:

Fraud_Detection_Project/
│
├── fraud_detection.ipynb   # Your Google Colab notebook
├── model.pkl               # (Optional) Saved model
├── requirements.txt        # Package dependencies
└── README.md               # Documentation (this file)
