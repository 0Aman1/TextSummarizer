# 🎯 AI Text Summarizer

A comprehensive text summarization tool that implements both **extractive** and **abstractive** summarization techniques using advanced NLP methods. Perfect for beginners to intermediate learners looking to understand and implement modern text summarization approaches.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Technical Implementation](#-technical-implementation)
- [Web Interface](#-web-interface)
- [Evaluation Metrics](#-evaluation-metrics)
- [Project Structure](#-project-structure)
- [Extensions & Improvements](#-extensions--improvements)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Project Overview

This project demonstrates two main approaches to automatic text summarization:

### 📝 Extractive Summarization
- **Method**: TextRank algorithm with TF-IDF vectorization
- **Approach**: Selects the most important sentences from the original text
- **Advantages**: Preserves original phrasing, factually accurate
- **Use Cases**: News articles, research papers, reports

### 🧠 Abstractive Summarization  
- **Method**: Pre-trained transformer models (BART, T5)
- **Approach**: Generates new sentences that capture the essence
- **Advantages**: More concise, human-like summaries
- **Use Cases**: Content curation, executive summaries

### 🌍 Real-World Applications
- **News aggregation** - Summarize multiple news articles
- **Academic research** - Create abstracts from papers
- **Content curation** - Generate social media posts
- **Business reports** - Executive summaries
- **Legal documents** - Case summaries
- **Educational content** - Study guides

## ✨ Features

- **Dual Summarization Methods**: Both extractive and abstractive approaches
- **Smart Preprocessing**: Advanced text cleaning and sentence segmentation
- **Flexible Configuration**: Adjustable summary length and quality parameters
- **Evaluation Metrics**: Built-in ROUGE scoring for quality assessment
- **Web Interface**: User-friendly Streamlit application
- **GPU Support**: Accelerated processing with CUDA (optional)
- **Extensible Design**: Modular architecture for easy customization
- **Comprehensive Examples**: Multiple demo texts included

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### Step 1: Clone or Download the Project
```bash
# Create project directory
mkdir text_summarizer_project
cd text_summarizer_project

# Download project files (text_summarizer.py, streamlit_app.py, requirements.txt)
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\\Scripts\\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_sm
```

### Step 4: Verify Installation
```bash
python text_summarizer.py
```

## 🚀 Quick Start

### Basic Usage - Command Line
```python
from text_summarizer import TextSummarizer

# Initialize the summarizer
summarizer = TextSummarizer()

# Your text to summarize
text = """
Your long text here...
"""

# Generate both types of summaries
summaries = summarizer.summarize(text, method="both")

# Display results
print("Extractive Summary:")
print(summaries['extractive'])

print("\\nAbstractive Summary:")
print(summaries['abstractive'])
```

### Web Interface - Streamlit App
```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

## 📚 Usage Examples

### Example 1: Extractive Summarization Only
```python
from text_summarizer import TextSummarizer

summarizer = TextSummarizer()

# Generate only extractive summary with custom length
summary = summarizer.summarize(
    text="Your text here...",
    method="extractive",
    summary_ratio=0.4  # Include 40% of original sentences
)

print(summary['extractive'])
```

### Example 2: Abstractive Summarization with Custom Parameters
```python
summary = summarizer.summarize(
    text="Your text here...",
    method="abstractive",
    max_length=200,  # Maximum tokens in summary
    min_length=75    # Minimum tokens in summary
)

print(summary['abstractive'])
```

### Example 3: Compare Both Methods
```python
# Compare both methods with evaluation
text = "Your article text..."
reference_summary = "Your reference summary..."

summarizer.compare_methods(text, reference_summary)
```

## 🔬 Technical Implementation

### 🧹 Text Preprocessing
1. **Text Cleaning**: Remove extra whitespace, special characters
2. **Sentence Segmentation**: Split text using spaCy/NLTK
3. **Tokenization**: Word-level tokenization with stopword removal
4. **Normalization**: Lowercasing and lemmatization

### 📊 Extractive Summarization Process
```
Text → Sentence Segmentation → TF-IDF Vectorization → 
Similarity Matrix → PageRank Algorithm → Top Sentences → Summary
```

**Key Components:**
- **TF-IDF Vectorization**: Convert sentences to numerical vectors
- **Cosine Similarity**: Measure sentence relationships
- **TextRank Algorithm**: Graph-based ranking using PageRank
- **Sentence Selection**: Choose top-ranked sentences maintaining order

### 🧠 Abstractive Summarization Process
```
Text → Tokenization → Transformer Model → 
Beam Search → Post-processing → Generated Summary
```

**Key Components:**
- **Pre-trained Models**: BART-large-CNN or DistilBART
- **Attention Mechanism**: Focus on important text portions
- **Beam Search**: Generate multiple candidate summaries
- **Length Control**: Enforce minimum/maximum summary lengths

### 📈 Evaluation Metrics

The project uses **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** metrics:

- **ROUGE-1**: Unigram overlap between generated and reference summaries
- **ROUGE-2**: Bigram overlap for fluency assessment
- **ROUGE-L**: Longest Common Subsequence for structural similarity

```python
from text_summarizer import SummarizerEvaluator

evaluator = SummarizerEvaluator()
scores = evaluator.evaluate_summary(reference_summary, generated_summary)
evaluator.print_evaluation(scores)
```

## 🌐 Web Interface

The Streamlit app provides an intuitive interface with:

### Features:
- **📄 Text Input**: Large text area with example texts
- **⚙️ Configuration Panel**: Method selection and parameter tuning
- **📊 Statistics**: Word counts and compression ratios
- **📥 Download Results**: Export summaries as text files
- **🎨 Modern UI**: Responsive design with custom styling

### Configuration Options:
- **Summarization Method**: Extractive, Abstractive, or Both
- **Summary Length**: Adjustable ratio for extractive method
- **Token Limits**: Min/max length for abstractive method
- **Example Texts**: Pre-loaded content for testing

### Running the Web App:
```bash
streamlit run streamlit_app.py
```

Access at: `http://localhost:8501`

## 📁 Project Structure

```
text_summarizer_project/
│
├── text_summarizer.py      # Main summarization classes and logic
├── streamlit_app.py        # Web interface application
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
│
├── examples/              # Example texts and demos
│   ├── ai_technology.txt
│   ├── climate_change.txt
│   └── space_exploration.txt
│
└── docs/                  # Additional documentation
    ├── technical_details.md
    └── evaluation_guide.md
```

### Core Classes:

- **`TextPreprocessor`**: Handles text cleaning and preparation
- **`ExtractiveSummarizer`**: Implements TextRank-based summarization
- **`AbstractiveSummarizer`**: Manages transformer model integration
- **`SummarizerEvaluator`**: Provides ROUGE-based evaluation
- **`TextSummarizer`**: Main orchestrator class

## 🔍 Example Input/Output

### Input Text:
```
Artificial intelligence (AI) is rapidly transforming industries across the globe by 
automating complex tasks and providing unprecedented data-driven insights. Machine 
learning algorithms can now process vast amounts of information in seconds, identifying 
patterns that would take humans years to discover. However, the rise of AI also brings 
significant ethical concerns that society must address carefully...
```

### Extractive Summary:
```
Artificial intelligence (AI) is rapidly transforming industries across the globe by 
automating complex tasks and providing unprecedented data-driven insights. However, 
the rise of AI also brings significant ethical concerns that society must address 
carefully. Despite these challenges, AI continues to offer tremendous benefits for society.
```

### Abstractive Summary:
```
AI is revolutionizing industries through automation and data insights, but ethical 
concerns about privacy and bias must be addressed to ensure responsible development 
and deployment of these powerful technologies.
```

## 🚀 Extensions & Improvements

### Beginner Extensions:
1. **Multi-language Support**: Extend to other languages
2. **Summary Length Options**: Predefined short/medium/long summaries
3. **File Upload**: Support for PDF, DOC, TXT files
4. **Batch Processing**: Summarize multiple documents at once

### Intermediate Extensions:
1. **Custom Models**: Train domain-specific summarization models
2. **API Development**: Create REST API for integration
3. **Database Integration**: Store and retrieve summaries
4. **Keyword Highlighting**: Identify and highlight important terms

### Advanced Extensions:
1. **Multi-document Summarization**: Combine multiple sources
2. **Query-focused Summarization**: Generate summaries based on specific questions
3. **Real-time Summarization**: Process streaming text data
4. **Cross-modal Summarization**: Include images and videos

### Deployment Options:
- **Streamlit Cloud**: Free cloud hosting for the web app
- **Hugging Face Spaces**: Share models and applications
- **Docker Container**: Containerized deployment
- **AWS/GCP/Azure**: Cloud platform deployment

## 🔧 Troubleshooting

### Common Issues:

**1. Model Download Errors**
```bash
# Clear transformers cache
rm -rf ~/.cache/huggingface/transformers/

# Reinstall transformers
pip uninstall transformers
pip install transformers
```

**2. spaCy Model Missing**
```bash
python -m spacy download en_core_web_sm
```

**3. CUDA Out of Memory**
```python
# Use CPU-only mode
device = -1  # Force CPU in AbstractiveSummarizer
```

**4. Slow Performance**
- Use smaller models (distilbart instead of bart-large)
- Enable GPU acceleration if available
- Reduce input text length

### Performance Optimization:
- **GPU Usage**: Install CUDA-compatible PyTorch
- **Model Selection**: Use smaller models for faster inference
- **Batch Processing**: Process multiple texts together
- **Caching**: Cache model outputs for repeated texts

## 📊 Performance Benchmarks

### Processing Speed (approximate):
- **Extractive**: ~1000 words/second
- **Abstractive (CPU)**: ~50 words/second  
- **Abstractive (GPU)**: ~200 words/second

### Memory Usage:
- **Extractive**: ~100MB RAM
- **Abstractive**: ~2-4GB RAM (model dependent)

### Quality Metrics (on news articles):
- **ROUGE-1 F-Score**: 0.35-0.45
- **ROUGE-2 F-Score**: 0.15-0.25
- **ROUGE-L F-Score**: 0.30-0.40

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** and test thoroughly
4. **Submit a pull request** with a clear description

### Contribution Ideas:
- Add support for new languages
- Implement additional evaluation metrics
- Create new example datasets
- Improve documentation
- Optimize performance
- Add unit tests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For providing pre-trained transformer models
- **spaCy**: For advanced natural language processing capabilities
- **NetworkX**: For graph-based algorithms
- **Streamlit**: For the web interface framework
- **NLTK**: For fundamental NLP tools

## 📞 Support

Having issues? Here's how to get help:

1. **Check the documentation** in this README
2. **Review common issues** in the Troubleshooting section
3. **Search existing issues** on the project repository
4. **Create a new issue** with detailed information about your problem

---

**Happy Summarizing! 🎉**

*Built with ❤️ for the NLP community*
