# utils/text_preprocessor.py

import re
import nltk
from nltk.corpus import stopwords

# Download if not already available (optional in prod)
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Clean and preprocess input text: remove HTML, punctuation, lowercase, remove stopwords.
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and non-alphabet characters
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Join words back into string
    return ' '.join(words)
