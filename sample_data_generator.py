"""
Sample data generator for testing when internet access is unavailable
"""
import numpy as np
from typing import Tuple, List


def generate_sample_news_data(n_samples: int = 100, 
                              n_categories: int = 5) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Generate sample news articles for testing
    
    Args:
        n_samples: Number of sample documents to generate
        n_categories: Number of categories
        
    Returns:
        texts: List of sample news texts
        labels: Array of category labels
        categories: List of category names
    """
    # Sample categories and their associated keywords
    category_data = {
        'Technology': [
            'software development artificial intelligence machine learning',
            'computer science programming database cloud computing',
            'internet technology digital innovation cybersecurity',
            'mobile apps smartphone tablet device electronics',
            'data science analytics big data algorithm optimization'
        ],
        'Sports': [
            'football soccer basketball game match tournament',
            'player team coach championship league competition',
            'athlete training fitness exercise health wellness',
            'olympic games medal winner runner athlete performance',
            'baseball tennis golf swimming racing victory defeat'
        ],
        'Politics': [
            'government election vote democracy republican democrat',
            'president senate congress legislation policy reform',
            'international relations diplomacy treaty alliance conflict',
            'campaign political party candidate debate speech',
            'economy budget taxation fiscal monetary policy regulation'
        ],
        'Science': [
            'research laboratory experiment hypothesis discovery theory',
            'physics chemistry biology medicine health disease',
            'space astronomy planet star galaxy universe cosmos',
            'climate environment ecology conservation sustainability',
            'genetics DNA evolution species molecular cell protein'
        ],
        'Business': [
            'company corporation market stock investment profit',
            'economy growth inflation interest rate financial',
            'entrepreneur startup business plan strategy marketing',
            'trade export import commerce industry manufacturing',
            'banking finance credit loan insurance investment fund'
        ]
    }
    
    # Select categories
    all_categories = list(category_data.keys())[:n_categories]
    
    # Generate texts
    texts = []
    labels = []
    
    np.random.seed(42)
    
    for i in range(n_samples):
        # Select category
        category_idx = i % len(all_categories)
        category = all_categories[category_idx]
        
        # Generate text by combining keywords
        keyword_phrases = category_data[category]
        n_sentences = np.random.randint(3, 8)
        
        sentences = []
        for _ in range(n_sentences):
            # Pick random keyword phrases
            phrase1 = np.random.choice(keyword_phrases)
            phrase2 = np.random.choice(keyword_phrases)
            phrase3 = np.random.choice(keyword_phrases)
            
            # Extract individual words
            words1 = phrase1.split()
            words2 = phrase2.split()
            words3 = phrase3.split()
            
            word1 = np.random.choice(words1)
            word2 = np.random.choice(words2)
            word3 = np.random.choice(words3)
            
            # Create sentence
            templates = [
                f"The {word1} is important for {word2} and {word3}.",
                f"Recent developments in {word1} show progress in {word2}.",
                f"Experts discuss {word1} in relation to {word2} and {word3}.",
                f"New research on {word1} reveals insights about {word2}.",
                f"The impact of {word1} on {word2} continues to grow."
            ]
            sentence = np.random.choice(templates)
            sentences.append(sentence)
        
        text = " ".join(sentences)
        texts.append(text)
        labels.append(category_idx)
    
    return texts, np.array(labels), all_categories


if __name__ == "__main__":
    texts, labels, categories = generate_sample_news_data(n_samples=20, n_categories=3)
    
    print(f"Generated {len(texts)} sample documents")
    print(f"Categories: {categories}")
    print(f"\nSample text (category: {categories[labels[0]]}):")
    print(texts[0])
