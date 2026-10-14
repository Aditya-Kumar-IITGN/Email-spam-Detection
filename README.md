# Email & SMS Spam Detection System

A complete machine learning and NLP application that classifies incoming messages as **Spam** or **Ham (Legitimate)**. The project features exploratory data analysis, text preprocessing, multi-model benchmarking (BernoulliNB, MultinomialNB, ComplementNB), an ensemble majority voting classifier, and a real-time **Streamlit** web application.

---

## Project Structure

```text
├── Spam_detection.ipynb    # End-to-end ML notebook: EDA, preprocessing, training & benchmarking
├── spam_detector.py        # Streamlit web application for real-time inference
├── model.pkl               # Serialized production classifier (Majority Voting Ensemble)
├── vectorizer.pkl          # Serialized fitted TF-IDF Vectorizer (3,000 features)
├── spam.txt                # Raw SMS & Email dataset (5,572 tab-separated messages)
├── requirements.txt        # Project dependencies
├── Screenshot/             # Visualizations (confusion matrix, word clouds, web UI)
└── README.md               # Project documentation
```

---

## Model Benchmark & Comparison

The project benchmarks multiple classification algorithms on the 20% test holdout (1,034 messages):

| Model Architecture | Accuracy | Spam Precision | Spam Recall | Spam F1-Score | Remarks |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Majority Voting Ensemble (BNB + MNB + CNB)** | **98.55%** | **99.20%** | **90.00%** | **0.94** | **Selected for Production**: Catches more spam while neutralizing false alarms. |
| **BernoulliNB (Original)** | 98.55% | 100.0% | 89.29% | 0.94 | Strong baseline with 0 false positives; misses ~10.7% of spam. |
| **MultinomialNB** | 97.68% | 99.15% | 83.57% | 0.91 | Lower recall due to word frequency dilution on short text. |
| **ComplementNB** | 91.10% | 61.01% | 95.00% | 0.74 | High recall, but produces 85 false alarms on legitimate messages. |

---

## Web Interface

The project includes an interactive web interface powered by Streamlit:

### Deployed Application Interface
![Deployed Interface](Screenshot/deployed.png)

### Word Clouds
* **Ham (Legitimate) Word Cloud**:
  ![Ham Detection](Screenshot/ham.png)

* **Spam Word Cloud**:
  ![Spam Detection](Screenshot/spam.png)

---

## Local Setup & Running the Application

### 1. Clone the repository
```bash
git clone https://github.com/Destructor169/Email-spam-Detection.git
cd Email-spam-Detection
```

### 2. Set up virtual environment & install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run spam_detector.py
```
Open your browser at `http://localhost:8501`.

---

## Deployment to Streamlit Community Cloud (Free Hosting)

1. Push your updated code to your GitHub repository:
   ```bash
   git add .
   git commit -m "Production model and deployment configuration"
   git push origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **"New app"** and enter:
   * **Repository**: `Destructor169/Email-spam-Detection`
   * **Branch**: `main`
   * **Main file path**: `spam_detector.py`
4. Click **Deploy!** Your app will be live on a public HTTPS URL.