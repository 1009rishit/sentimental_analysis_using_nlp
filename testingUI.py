import re
import json
import logging
import os
import pandas as pd
from typing import List
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import gradio as gr
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from groq import Groq

nltk.download('punkt')
nltk.download('stopwords')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SentimentApp")

# ----------------------------
# Preprocessing & Rule-Based
# ----------------------------
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def clean(self, text: str) -> List[str]:
        text = text.lower()
        words = word_tokenize(text)
        processed = []
        skip_next = False

        for i, word in enumerate(words):
            if skip_next:
                skip_next = False
                continue
            if word == 'not' and i + 1 < len(words):
                combined = f"not_{words[i + 1]}"
                processed.append(combined)
                skip_next = True
            elif word.isalnum() and word not in self.stop_words:
                processed.append(word)
        return processed

    def split_sentences(self, text: str) -> List[str]:
        return sent_tokenize(text)


class RuleBasedSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.pre = TextPreprocessor()

    def analyze(self, text: str) -> str:
        words = self.pre.clean(text)
        vader_scores = self.vader.polarity_scores(text)
        polarity = TextBlob(text).sentiment.polarity

        compound = vader_scores['compound']
        if compound >= 0.05:
            sentiment = 'Positive'
        elif compound <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return f"[Rule-Based]\nSentiment: {sentiment}\nVADER Compound: {compound:.2f}\nTextBlob Polarity: {polarity:.2f}"


# ----------------------------
# Trained NLP Model
# ----------------------------
class TrainedSentimentModel:
    def __init__(self, csv_path: str = "Reviews.csv"):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000)
        self.preprocessor = TextPreprocessor()
        self._train(csv_path)

    def _parse_rating(self, rating: str) -> float:
        match = re.match(r"(\d\.\d)", str(rating))
        return float(match.group(1)) if match else 3.0

    def _map_rating_to_label(self, rating: float) -> str:
        if rating >= 4:
            return "positive"
        elif rating <= 2:
            return "negative"
        return "neutral"

    def _extract_keywords(self, text_tokens: List[str], top_n: int = 5) -> List[str]:
        freq = Counter(text_tokens)
        most_common = freq.most_common(top_n)
        return [word for word, count in most_common]

    def _train(self, path: str):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df = df[['Review-Body', 'rating']].dropna()
        df['rating'] = df['rating'].apply(self._parse_rating)
        df['label'] = df['rating'].apply(self._map_rating_to_label)

        cleaned_texts = []
        for text in df['Review-Body'].astype(str):
            tokens = self.preprocessor.clean(text)
            keywords = self._extract_keywords(tokens, top_n=5)
            combined_text = " ".join(tokens + keywords)
            cleaned_texts.append(combined_text)

        df['clean'] = cleaned_texts

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(df['clean'], df['label'], test_size=0.2, random_state=42)

        # Vectorization and training
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.model.fit(X_train_vec, y_train)

        # Evaluation
        y_pred = self.model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        logger.info("Trained NLP model with stopword removal and keyword enhancement.")
        logger.info(f"Model Accuracy: {acc:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Classification Report:\n{cr}")

    def predict(self, text: str) -> str:
        tokens = self.preprocessor.clean(text)
        keywords = self._extract_keywords(tokens, top_n=5)
        combined_text = " ".join(tokens + keywords)

        X = self.vectorizer.transform([combined_text])
        pred_proba = self.model.predict_proba(X)[0]
        pred_label = self.model.predict(X)[0]
        label_index = list(self.model.classes_).index(pred_label)
        confidence = pred_proba[label_index]

        feature_names = self.vectorizer.get_feature_names_out()
        coefs = self.model.coef_[label_index]
        tfidf_scores = X.toarray()[0]

        feature_weights = [(feature_names[i], tfidf_scores[i] * coefs[i])
                           for i in range(len(feature_names)) if tfidf_scores[i] > 0]

        top_features = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)[:5]
        top_features_str = ", ".join(f"{word} ({weight:.2f})" for word, weight in top_features)

        return (
            f"[ML Model]\n"
            f"Predicted Sentiment: {pred_label.capitalize()}\n"
            f"Confidence: {confidence*100:.1f}%\n"
            f"Top Contributing Words: {top_features_str}"
        )


# ----------------------------
# LLM (Groq + LLaMA)
# ----------------------------
class LLMAnalyzer:
    def __init__(self):
        key = os.getenv
        self.client = Groq(api_key="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")

    def analyze(self, text: str) -> str:
        messages = [
            {"role": "system", "content": "Provide sentiment (Positive/Neutral/Negative), confidence %, and explanation."},
            {"role": "user", "content": text}
        ]
        try:
            res = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                messages=messages
            )
            content = res.choices[0].message.content
            try:
                data = json.loads(content)
                return f"[LLM]\nSentiment: {data.get('sentiment')}\nConfidence: {data.get('confidence', 0)*100:.0f}%\nExplanation: {data.get('explanation')}"
            except:
                return f"[LLM Raw Response]\n{content}"
        except Exception as e:
            return f"LLM error: {str(e)}"


# ----------------------------
# Gradio App
# ----------------------------
llm_analyzer = LLMAnalyzer()
rule_analyzer = RuleBasedSentimentAnalyzer()
ml_analyzer = TrainedSentimentModel("Reviews.csv")

def analyze(text: str, method: str) -> str:
    if not text.strip():
        return "Please enter valid text."
    if method == "LLM":
        return llm_analyzer.analyze(text)
    elif method == "Rule-Based":
        return rule_analyzer.analyze(text)
    else:
        return ml_analyzer.predict(text)


with gr.Blocks(title="Sentiment Analyzer", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        <div style='text-align:center; margin-bottom: 20px;'>
            <h1>🧠 Sentiment Analyzer App</h1>
            <p style='font-size:16px; color:#555;'>Analyze your reviews using <strong>LLM</strong>, <strong>Machine Learning</strong>, or <strong>Rule-Based</strong> models</p>
        </div>
        <hr>
        """, elem_id="header"
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            input_text = gr.Textbox(
                label="📝 Enter Review Text",
                placeholder="Type your review here...",
                lines=6,
                show_copy_button=True,
                elem_id="input-textbox"
            )

            analysis_mode = gr.Radio(
                ["LLM", "ML Model", "Rule-Based"],
                label="🧪 Choose Analysis Method",
                info="Switch between LLM (LLaMA-3), ML Model (Logistic Regression), or Rule-Based (VADER + TextBlob)",
                value="LLM",
                elem_id="method-radio"
            )

            analyze_btn = gr.Button("🚀 Analyze Sentiment", variant="primary", elem_id="analyze-btn")

            examples = gr.Examples(
                examples=[
                    ["I absolutely love this product! It exceeded all my expectations.", "LLM"],
                    ["The food was cold and the service was terrible.", "Rule-Based"],
                    ["Not satisfied with the purchase, but delivery was fast.", "ML Model"],
                    ["Great value for the price. Would buy again!", "ML Model"],
                    ["Not bad, but could be better.", "Rule-Based"],
                ],
                inputs=[input_text, analysis_mode],
                label="Try Examples",
                cache_examples=False,
                elem_id="examples"
            )

        with gr.Column(scale=6):
            result_card = gr.Accordion("📊 Analysis Result", open=True, elem_id="result-accordion")
            with result_card:
                result_box = gr.HTML("<p style='color: gray;'>Results will appear here after analysis.</p>", elem_id="result-box")

    def analyze_and_format(text, method):
        result = analyze(text, method)
        if result.startswith("[LLM Raw Response]") or "error" in result.lower():
            return f"<span style='color:red; font-weight:bold;'>❗ {result}</span>"

        formatted = ""
        lines = result.split("\n")
        emoji_map = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}
        for line in lines:
            if "Sentiment:" in line:
                sentiment = line.split(":")[-1].strip()
                color = {"Positive": "#27ae60", "Neutral": "#f39c12", "Negative": "#e74c3c"}.get(sentiment, "#000")
                emoji = emoji_map.get(sentiment, "")
                formatted += f"<h3 style='color:{color}'>Sentiment: {sentiment} {emoji}</h3>"
            elif "Confidence" in line:
                formatted += f"<p><strong>Confidence:</strong> {line.split(':')[-1].strip()}</p>"
            elif "Explanation" in line or "Top Contributing Words" in line:
                label = line.split(':')[0]
                details = ':'.join(line.split(':')[1:]).strip()
                formatted += f"<p><strong>{label}:</strong></p><blockquote>{details}</blockquote>"
            else:
                formatted += f"<p>{line}</p>"

        return formatted

    analyze_btn.click(fn=analyze_and_format, inputs=[input_text, analysis_mode], outputs=result_box)


if __name__ == "__main__":
    app.launch()
