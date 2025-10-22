from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import numpy as np

class IntentModel:
    def __init__(self, model_path='model.joblib'):
        self.model_path = model_path
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=20000)),
            ('clf', LogisticRegression(max_iter=1000))
        ])

    def train(self, texts, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        self.pipeline.fit(X_train, y_train)
        preds = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        return acc, report

    def predict(self, text):
        """
        text: str or list of str
        returns: (pred_label, confidence)
        """
        single_input = False
        if isinstance(text, str):
            text = [text]
            single_input = True

        preds = self.pipeline.predict(text)

        # Get confidence if possible
        if hasattr(self.pipeline.named_steps['clf'], 'predict_proba'):
            proba = self.pipeline.predict_proba(text)
            confidence = proba.max(axis=1)
        else:
            confidence = np.array([1.0]*len(preds))

        if single_input:
            return preds[0], float(confidence[0])
        return list(zip(preds, confidence))

    def save(self):
        joblib.dump(self.pipeline, self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
            return True
        return False
