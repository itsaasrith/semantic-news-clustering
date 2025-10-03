"""
Step 2: Preprocess Text
Cleans and preprocesses text data for better embedding quality
"""
import re
import string
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """Handles text preprocessing and cleaning"""
    
    def __init__(self, remove_stopwords: bool = True, 
                 lowercase: bool = True,
                 min_token_length: int = 2):
        """
        Initialize the preprocessor
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lowercase: Whether to convert to lowercase
            min_token_length: Minimum token length to keep
        """
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.min_token_length = min_token_length
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text document
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers (optional - keeping some context)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation but keep sentence structure
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        
        # Tokenize and filter
        tokens = word_tokenize(text)
        
        # Filter tokens
        tokens = [
            token for token in tokens 
            if len(token) >= self.min_token_length 
            and token not in self.stop_words
        ]
        
        # Rejoin tokens
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text.strip()
    
    def preprocess_batch(self, texts: List[str], verbose: bool = True) -> List[str]:
        """
        Preprocess a batch of texts
        
        Args:
            texts: List of raw texts
            verbose: Whether to print progress
            
        Returns:
            List of cleaned texts
        """
        if verbose:
            print(f"Preprocessing {len(texts)} documents...")
        
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        if verbose:
            avg_len_before = sum(len(t) for t in texts) / len(texts)
            avg_len_after = sum(len(t) for t in cleaned_texts) / len(cleaned_texts)
            print(f"Average length: {avg_len_before:.0f} -> {avg_len_after:.0f} chars")
        
        return cleaned_texts


if __name__ == "__main__":
    # Example usage
    from step1_data_loader import NewsDataLoader
    
    # Load data
    loader = NewsDataLoader(subset='train')
    texts, labels, categories = loader.load_data(num_samples=100)
    
    # Preprocess
    preprocessor = TextPreprocessor()
    cleaned_texts = preprocessor.preprocess_batch(texts[:5])
    
    print("\nOriginal text (first 200 chars):")
    print(texts[0][:200])
    print("\nCleaned text (first 200 chars):")
    print(cleaned_texts[0][:200])
