"""
Text Summarizer Project
======================

A comprehensive text summarization tool that implements both extractive and abstractive
summarization techniques using Python NLP libraries.

Features:
- Extractive summarization using TextRank algorithm
- Abstractive summarization using transformer models
- Evaluation metrics (ROUGE scores)
- Text preprocessing and cleaning
- Example usage and demonstrations

Author: NLP Project Generator
Date: 2024
"""

import re
import math
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Tuple, Dict

# NLP Libraries
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Transformer Libraries
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Evaluation
from rouge_score import rouge_scorer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')

class TextPreprocessor:
    """
    Handles all text preprocessing tasks including cleaning, tokenization,
    and sentence segmentation.
    """
    
    def __init__(self, language='english'):
        """
        Initialize the preprocessor with language-specific settings.
        
        Args:
            language (str): Language for stopwords and processing
        """
        self.language = language
        self.stop_words = set(nltk.corpus.stopwords.words(language))
        
        # Try to load spaCy model, fallback to basic processing if not available
        try:
            self.nlp = spacy.load('en_core_web_sm')
            self.use_spacy = True
            print("✓ Using spaCy for advanced preprocessing")
        except OSError:
            print("⚠ spaCy model not found. Using basic preprocessing.")
            print("  Install with: python -m spacy download en_core_web_sm")
            self.use_spacy = False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize the input text.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep sentence-ending punctuation
        text = re.sub(r'[^\w\s\.\!\?\;\:]', '', text)
        
        return text
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Split text into individual sentences.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        if self.use_spacy:
            # Use spaCy for better sentence segmentation
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            # Fallback to NLTK
            sentences = nltk.sent_tokenize(text)
        
        # Filter out very short sentences (less than 3 words)
        sentences = [s for s in sentences if len(s.split()) >= 3]
        
        return sentences
    
    def preprocess_for_vectorization(self, text: str) -> str:
        """
        Preprocess text specifically for TF-IDF vectorization.
        
        Args:
            text (str): Input sentence
            
        Returns:
            str: Preprocessed text ready for vectorization
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize and remove stopwords
        if self.use_spacy:
            doc = self.nlp(text)
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop and not token.is_punct and token.text.strip()]
        else:
            # Basic tokenization without lemmatization
            tokens = nltk.word_tokenize(text)
            tokens = [word for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)


class ExtractiveSummarizer:
    """
    Implements extractive summarization using the TextRank algorithm.
    This method selects the most important sentences from the original text.
    """
    
    def __init__(self, preprocessor: TextPreprocessor):
        """
        Initialize the extractive summarizer.
        
        Args:
            preprocessor (TextPreprocessor): Text preprocessing instance
        """
        self.preprocessor = preprocessor
    
    def _compute_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Compute similarity matrix between sentences using TF-IDF and cosine similarity.
        
        Args:
            sentences (List[str]): List of sentences
            
        Returns:
            np.ndarray: Similarity matrix
        """
        # Preprocess sentences for vectorization
        processed_sentences = [
            self.preprocessor.preprocess_for_vectorization(sent) 
            for sent in sentences
        ]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=1
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(processed_sentences)
            
            # Compute cosine similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix
            
        except ValueError:
            # Handle edge case where no features can be extracted
            print("⚠ Warning: Could not create TF-IDF vectors. Using uniform similarity.")
            return np.ones((len(sentences), len(sentences))) * 0.1
    
    def _rank_sentences(self, similarity_matrix: np.ndarray) -> List[float]:
        """
        Rank sentences using PageRank algorithm.
        
        Args:
            similarity_matrix (np.ndarray): Sentence similarity matrix
            
        Returns:
            List[float]: Sentence scores
        """
        # Create graph from similarity matrix
        nx_graph = nx.from_numpy_array(similarity_matrix)
        
        # Apply PageRank algorithm
        try:
            scores = nx.pagerank(nx_graph, max_iter=100, tol=1e-4)
            return [scores[i] for i in range(len(scores))]
        except:
            # Fallback: return uniform scores
            return [1.0] * similarity_matrix.shape[0]
    
    def summarize(self, text: str, summary_ratio: float = 0.3) -> str:
        """
        Generate extractive summary of the input text.
        
        Args:
            text (str): Input text to summarize
            summary_ratio (float): Fraction of sentences to include in summary
            
        Returns:
            str: Extractive summary
        """
        # Clean and segment text
        clean_text = self.preprocessor.clean_text(text)
        sentences = self.preprocessor.segment_sentences(clean_text)
        
        if len(sentences) <= 3:
            return text  # Return original if too short
        
        # Compute similarity matrix and rank sentences
        similarity_matrix = self._compute_similarity_matrix(sentences)
        sentence_scores = self._rank_sentences(similarity_matrix)
        
        # Select top sentences
        num_sentences = max(1, int(len(sentences) * summary_ratio))
        
        # Get indices of top-ranked sentences
        ranked_indices = sorted(
            range(len(sentence_scores)), 
            key=lambda i: sentence_scores[i], 
            reverse=True
        )[:num_sentences]
        
        # Sort selected indices to maintain original order
        selected_indices = sorted(ranked_indices)
        
        # Create summary
        summary_sentences = [sentences[i] for i in selected_indices]
        summary = ' '.join(summary_sentences)
        
        return summary


class AbstractiveSummarizer:
    """
    Implements abstractive summarization using pre-trained transformer models.
    This method generates new sentences that capture the essence of the original text.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the abstractive summarizer.
        
        Args:
            model_name (str): Hugging Face model name for summarization
        """
        self.model_name = model_name
        self.summarizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the summarization model and tokenizer."""
        try:
            print(f"🔄 Loading model: {self.model_name}")
            
            # Check if CUDA is available
            device = 0 if torch.cuda.is_available() else -1
            if device == 0:
                print("✓ Using GPU acceleration")
            else:
                print("✓ Using CPU (consider GPU for faster processing)")
            
            # Load the summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                device=device
            )
            
            print("✓ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Falling back to a smaller model...")
            
            try:
                # Fallback to a smaller, more reliable model
                self.model_name = "sshleifer/distilbart-cnn-12-6"
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model_name,
                    device=-1  # Use CPU for fallback
                )
                print("✓ Fallback model loaded successfully")
            except Exception as e2:
                print(f"❌ Critical error: {e2}")
                print("Please install transformers and torch properly")
                self.summarizer = None
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """
        Generate abstractive summary of the input text.
        
        Args:
            text (str): Input text to summarize
            max_length (int): Maximum length of summary
            min_length (int): Minimum length of summary
            
        Returns:
            str: Abstractive summary
        """
        if not self.summarizer:
            return "❌ Abstractive summarization unavailable. Please check model installation."
        
        try:
            # Truncate text if it's too long for the model
            max_input_length = 1024  # Most models have input limits
            if len(text.split()) > max_input_length:
                text = ' '.join(text.split()[:max_input_length])
                print(f"⚠ Text truncated to {max_input_length} words")
            
            # Generate summary
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,  # Use deterministic generation
                num_beams=4,      # Beam search for better quality
                length_penalty=2.0,
                early_stopping=True
            )
            
            return summary[0]['summary_text']
            
        except Exception as e:
            print(f"❌ Error during summarization: {e}")
            return f"Error generating summary: {str(e)}"


class SummarizerEvaluator:
    """
    Evaluates the quality of generated summaries using various metrics.
    """
    
    def __init__(self):
        """Initialize the evaluator with ROUGE scorer."""
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate_summary(self, reference: str, summary: str) -> Dict[str, float]:
        """
        Evaluate summary quality using ROUGE metrics.
        
        Args:
            reference (str): Reference (ground truth) summary
            summary (str): Generated summary
            
        Returns:
            Dict[str, float]: ROUGE scores
        """
        scores = self.scorer.score(reference, summary)
        
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rouge2_p': scores['rouge2'].precision,
            'rouge2_r': scores['rouge2'].recall,
            'rougeL_f': scores['rougeL'].fmeasure,
            'rougeL_p': scores['rougeL'].precision,
            'rougeL_r': scores['rougeL'].recall,
        }
    
    def print_evaluation(self, scores: Dict[str, float]):
        """
        Print evaluation results in a formatted way.
        
        Args:
            scores (Dict[str, float]): ROUGE scores dictionary
        """
        print("\n📊 ROUGE Evaluation Results:")
        print("=" * 40)
        print(f"ROUGE-1 F-Score: {scores['rouge1_f']:.4f}")
        print(f"ROUGE-1 Precision: {scores['rouge1_p']:.4f}")
        print(f"ROUGE-1 Recall: {scores['rouge1_r']:.4f}")
        print(f"ROUGE-2 F-Score: {scores['rouge2_f']:.4f}")
        print(f"ROUGE-L F-Score: {scores['rougeL_f']:.4f}")


class TextSummarizer:
    """
    Main class that combines extractive and abstractive summarization methods.
    """
    
    def __init__(self):
        """Initialize the complete text summarization system."""
        print("🚀 Initializing Text Summarizer...")
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.extractive_summarizer = ExtractiveSummarizer(self.preprocessor)
        self.abstractive_summarizer = AbstractiveSummarizer()
        self.evaluator = SummarizerEvaluator()
        
        print("✓ Text Summarizer ready!")
    
    def summarize(self, text: str, method: str = "both", **kwargs) -> Dict[str, str]:
        """
        Generate summaries using specified method(s).
        
        Args:
            text (str): Input text to summarize
            method (str): "extractive", "abstractive", or "both"
            **kwargs: Additional parameters for summarization methods
            
        Returns:
            Dict[str, str]: Dictionary containing summaries
        """
        results = {}
        
        if method in ["extractive", "both"]:
            print("🔄 Generating extractive summary...")
            extractive_summary = self.extractive_summarizer.summarize(
                text, 
                summary_ratio=kwargs.get('summary_ratio', 0.3)
            )
            results['extractive'] = extractive_summary
        
        if method in ["abstractive", "both"]:
            print("🔄 Generating abstractive summary...")
            abstractive_summary = self.abstractive_summarizer.summarize(
                text,
                max_length=kwargs.get('max_length', 150),
                min_length=kwargs.get('min_length', 50)
            )
            results['abstractive'] = abstractive_summary
        
        return results
    
    def compare_methods(self, text: str, reference_summary: str = None):
        """
        Compare extractive and abstractive summarization methods.
        
        Args:
            text (str): Input text
            reference_summary (str, optional): Reference summary for evaluation
        """
        print("\n🔍 Comparing Summarization Methods")
        print("=" * 50)
        
        # Generate summaries
        summaries = self.summarize(text, method="both")
        
        # Display results
        print(f"\n📄 Original Text ({len(text.split())} words):")
        print("-" * 30)
        print(text[:200] + "..." if len(text) > 200 else text)
        
        if 'extractive' in summaries:
            print(f"\n📝 Extractive Summary ({len(summaries['extractive'].split())} words):")
            print("-" * 30)
            print(summaries['extractive'])
        
        if 'abstractive' in summaries:
            print(f"\n🧠 Abstractive Summary ({len(summaries['abstractive'].split())} words):")
            print("-" * 30)
            print(summaries['abstractive'])
        
        # Evaluate if reference is provided
        if reference_summary:
            print("\n📊 Evaluation Results:")
            print("=" * 30)
            
            if 'extractive' in summaries:
                print("\nExtractive Summary Evaluation:")
                ext_scores = self.evaluator.evaluate_summary(reference_summary, summaries['extractive'])
                self.evaluator.print_evaluation(ext_scores)
            
            if 'abstractive' in summaries:
                print("\nAbstractive Summary Evaluation:")
                abs_scores = self.evaluator.evaluate_summary(reference_summary, summaries['abstractive'])
                self.evaluator.print_evaluation(abs_scores)


def main():
    """
    Main function demonstrating the text summarization system.
    """
    print("🎯 Text Summarization Project Demo")
    print("=" * 50)
    
    # Example text - AI and technology article
    example_text = """
    Artificial intelligence (AI) is rapidly transforming industries across the globe by automating complex tasks and providing unprecedented data-driven insights. Machine learning algorithms can now process vast amounts of information in seconds, identifying patterns that would take humans years to discover. However, the rise of AI also brings significant ethical concerns that society must address carefully.
    
    Privacy remains one of the most pressing challenges in the AI era. As AI systems require enormous datasets to function effectively, there are growing concerns about how personal information is collected, stored, and used. Many AI applications collect data from users without their explicit consent, creating potential vulnerabilities for privacy breaches.
    
    Bias in AI systems represents another critical issue that needs immediate attention. AI algorithms learn from historical data, which often contains inherent biases reflecting societal inequalities. When these biased datasets are used to train AI models, the resulting systems can perpetuate and even amplify existing discrimination in areas such as hiring, lending, and law enforcement.
    
    The job displacement caused by AI automation is also a significant concern for many workers and policymakers. While AI can increase productivity and create new types of jobs, it may also eliminate traditional roles faster than new opportunities can be created. This technological unemployment could lead to increased inequality if not managed properly through retraining programs and social support systems.
    
    Despite these challenges, AI continues to offer tremendous benefits for society. In healthcare, AI-powered diagnostic tools can detect diseases earlier and more accurately than human doctors in many cases. In transportation, autonomous vehicles promise to reduce traffic accidents and improve mobility for disabled individuals. In education, personalized learning systems can adapt to individual student needs, potentially revolutionizing how we learn.
    
    The key to harnessing AI's potential while mitigating its risks lies in developing robust governance frameworks, ensuring transparent and explainable AI systems, and fostering inclusive dialogue between technologists, policymakers, and society at large. Only through collaborative effort can we ensure that AI serves humanity's best interests while respecting fundamental human rights and values.
    """
    
    # Initialize summarizer
    summarizer = TextSummarizer()
    
    # Demonstrate comparison
    print("\n🎭 Demonstration: Comparing Summarization Methods")
    summarizer.compare_methods(example_text)
    
    # Interactive example
    print("\n" + "=" * 50)
    print("🎮 Try it yourself!")
    print("Paste your own text or press Enter to skip:")
    
    user_text = input().strip()
    if user_text:
        print(f"\nProcessing your text ({len(user_text.split())} words)...")
        summaries = summarizer.summarize(user_text, method="both")
        
        print(f"\n📝 Extractive Summary:")
        print(summaries.get('extractive', 'Not available'))
        
        print(f"\n🧠 Abstractive Summary:")
        print(summaries.get('abstractive', 'Not available'))
    
    print("\n✨ Demo completed! Check out the Streamlit app for a better interface.")
    print("Run: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
