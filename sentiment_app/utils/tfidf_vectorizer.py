# utils/tfidf_vectorizer.py

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def prepare_tfidf_features(df, text_column, label_column, max_features=5000, test_size=0.2, random_state=42):
    """
    Splits the data, fits TF-IDF on training data, transforms both sets,
    and saves the vectorizer for production use.
    """
    X = df[text_column]
    y = df[label_column]

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"\n📊 Splitting complete:")
    print(f"Training size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # Initialize and fit vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    print("\n🔧 Fitting TF-IDF on training data...")
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    print("✅ Transforming test data...")
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Save vectorizer
    joblib.dump(tfidf_vectorizer, 'model/tfidf_vectorizer.pkl')
    print("📦 Saved TF-IDF vectorizer to model/tfidf_vectorizer.pkl")

    return X_train_tfidf, X_test_tfidf, y_train, y_test
