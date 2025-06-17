# 🧠 Sentiment Analyzer App

A multi-model sentiment analysis web app that lets you analyze text using **LLM-based**, **Machine Learning-based**, and **Rule-Based** techniques. Built with Gradio, the app allows real-time, interpretable predictions using a trained Logistic Regression model, rule-based NLP tools, or a powerful LLaMA-3 model via Groq API.

## 🚀 Features

- **Three Sentiment Models**:
  - **LLM (LLaMA-3 via Groq)** – Zero-shot sentiment with explanation and confidence.
  - **Machine Learning (Logistic Regression)** – Trained on 25K+ reviews using TF-IDF.
  - **Rule-Based (VADER + TextBlob)** – Fast and interpretable polarity detection.

- **Advanced Preprocessing**:
  - Stopword removal, negation handling (`not good` → `not_good`).
  - Keyword enhancement for boosting ML accuracy.

- **Interactive UI with Gradio**:
  - Clean, user-friendly interface with selectable models and example prompts.
  - Real-time analysis with styled result formatting and emoji indicators.

## 📂 Project Structure

sentiment-analyzer/
├── Reviews.csv # Dataset of customer reviews with ratings
├── app.py # Main Python script (Gradio UI + model logic)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

markdown
Copy
Edit

## 🧠 Models Used

- **LLM (LLaMA-3-70B)** via [Groq API](https://console.groq.com)
- **ML Model**: Logistic Regression with TF-IDF (5000 features)
- **Rule-Based**: VADER Sentiment + TextBlob Polarity

## 📊 Dataset

- **Source**: `Reviews.csv`  
- **Fields Used**: `Review-Body`, `rating`  
- **Size**: ~25,000 reviews  
- Ratings are mapped to `positive`, `neutral`, `negative` labels based on thresholds.

## ⚙️ Installation & Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sentiment-analyzer.git
cd sentiment-analyzer
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Set Environment Variable
Create a .env file or export your Groq API key:

bash
Copy
Edit
export GROQ_API_KEY=your_groq_api_key_here
4. Run the App
bash
Copy
Edit
python app.py
The app will launch in your browser via Gradio.

🔍 Examples
I absolutely love this product! It exceeded all my expectations. → Positive

The food was cold and the service was terrible. → Negative

Not bad, but could be better. → Neutral

📦 Requirements
Python 3.8+

Gradio

NLTK

TextBlob

scikit-learn

pandas

vaderSentiment

groq

Install all dependencies with:

bash
Copy
Edit
pip install -r requirements.txt
🧪 Future Improvements
Add support for multiple languages

Visual analytics for sentiment trends

Model explainability using SHAP or LIME
