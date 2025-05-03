# train_model.py

import pandas as pd
from utils.text_preprocessor import preprocess_text
from utils.tfidf_vectorizer import prepare_tfidf_features
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# 1. Load data
df = pd.read_csv("data/IMDB Dataset.csv")
df['review'] = df['review'].astype(str)

# 2. Preprocess review column
print("🧹 Preprocessing reviews...")
df['cleaned_review'] = df['review'].apply(preprocess_text)

# 3. TF-IDF vectorization
X_train_tfidf, X_test_tfidf, y_train, y_test = prepare_tfidf_features(
    df,
    text_column='cleaned_review',
    label_column='sentiment'
)

# 4. Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

# 5. Train and evaluate all models
results = {}
best_model = None
best_model_name = ""
best_accuracy = 0.0

for name, model in models.items():
    print(f"\n📌 Training {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("🧱 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    results[name] = acc

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# 6. Save the best model
if best_model:
    save_path = f'model/{best_model_name.lower().replace(" ", "_")}_model.pkl'
    joblib.dump(best_model, save_path)
    print(f"\n✅ Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    print(f"📦 Saved to: {save_path}")
