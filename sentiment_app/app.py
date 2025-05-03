# app.py
import joblib
from utils.text_preprocessor import preprocess_text

# Load saved model and vectorizer
model = joblib.load("model/logistic_regression_model.pkl")  # or the best one saved
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# CLI Input
print("🎬 IMDB Review Sentiment Predictor")
user_input = input("\nEnter a movie review: ")

# Preprocess the input review
cleaned_text = preprocess_text(user_input)

# Vectorize the input
vectorized_input = vectorizer.transform([cleaned_text])

# Predict sentiment
prediction = model.predict(vectorized_input)[0]

# Output the result
print("\n🧠 Predicted Sentiment:", "✅ Positive" if prediction == 1 else "❌ Negative")
