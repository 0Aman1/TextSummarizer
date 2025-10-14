"""
Streamlit Web Interface for Text Summarizer
==========================================

A user-friendly web interface for the text summarization project using Streamlit.
Provides an intuitive way to test both extractive and abstractive summarization methods.

Run this app with: streamlit run streamlit_app.py
"""

import streamlit as st
import time
from text_summarizer import TextSummarizer
import plotly.express as px
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="🎯 AI Text Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
        color: #000000;
    }
    .method-header {
        color: #2e7d32;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .stats-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = None
    st.session_state.summaries = {}

def initialize_summarizer():
    """Initialize the text summarizer with caching."""
    if st.session_state.summarizer is None:
        with st.spinner("🔄 Initializing AI models... This may take a moment."):
            st.session_state.summarizer = TextSummarizer()
    return st.session_state.summarizer

def display_summary_stats(original_text, summaries):
    """Display statistics about the summaries."""
    original_words = len(original_text.split())
    
    stats_data = []
    if 'extractive' in summaries:
        ext_words = len(summaries['extractive'].split())
        stats_data.append({
            'Method': 'Extractive',
            'Words': ext_words,
            'Compression Ratio': f"{ext_words/original_words:.2%}"
        })
    
    if 'abstractive' in summaries:
        abs_words = len(summaries['abstractive'].split())
        stats_data.append({
            'Method': 'Abstractive', 
            'Words': abs_words,
            'Compression Ratio': f"{abs_words/original_words:.2%}"
        })
    
    if stats_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="stats-box">
                <h3>📊 Original</h3>
                <p><strong>{original_words}</strong> words</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if 'extractive' in summaries:
                ext_words = len(summaries['extractive'].split())
                compression = (1 - ext_words/original_words) * 100
                st.markdown(f"""
                <div class="stats-box">
                    <h3>📝 Extractive</h3>
                    <p><strong>{ext_words}</strong> words</p>
                    <p>{compression:.1f}% compression</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if 'abstractive' in summaries:
                abs_words = len(summaries['abstractive'].split())
                compression = (1 - abs_words/original_words) * 100
                st.markdown(f"""
                <div class="stats-box">
                    <h3>🧠 Abstractive</h3>
                    <p><strong>{abs_words}</strong> words</p>
                    <p>{compression:.1f}% compression</p>
                </div>
                """, unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 AI Text Summarizer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the AI Text Summarizer! This tool uses advanced NLP techniques to create 
    both **extractive** (selecting key sentences) and **abstractive** (generating new text) summaries.
    """)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Method selection
        method = st.selectbox(
            "Choose Summarization Method",
            ["Both Methods", "Extractive Only", "Abstractive Only"],
            help="Select which summarization technique to use"
        )
        
        # Method mapping
        method_map = {
            "Both Methods": "both",
            "Extractive Only": "extractive", 
            "Abstractive Only": "abstractive"
        }
        selected_method = method_map[method]
        
        st.markdown("---")
        
        # Extractive settings
        if selected_method in ["extractive", "both"]:
            st.subheader("📝 Extractive Settings")
            summary_ratio = st.slider(
                "Summary Length Ratio",
                min_value=0.1,
                max_value=0.8,
                value=0.3,
                step=0.1,
                help="Fraction of original sentences to include"
            )
        else:
            summary_ratio = 0.3
        
        # Abstractive settings
        if selected_method in ["abstractive", "both"]:
            st.subheader("🧠 Abstractive Settings")
            max_length = st.slider(
                "Maximum Summary Length",
                min_value=50,
                max_value=300,
                value=150,
                step=25,
                help="Maximum number of tokens in summary"
            )
            min_length = st.slider(
                "Minimum Summary Length", 
                min_value=20,
                max_value=100,
                value=50,
                step=10,
                help="Minimum number of tokens in summary"
            )
        else:
            max_length = 150
            min_length = 50
        
        st.markdown("---")
        
        # Example texts
        st.subheader("📚 Example Texts")
        example_choice = st.selectbox(
            "Load an example:",
            ["None", "AI & Technology", "Climate Change", "Space Exploration"]
        )
    
    # Example texts
    examples = {
        "AI & Technology": """
        Artificial intelligence (AI) is rapidly transforming industries across the globe by automating complex tasks and providing unprecedented data-driven insights. Machine learning algorithms can now process vast amounts of information in seconds, identifying patterns that would take humans years to discover. However, the rise of AI also brings significant ethical concerns that society must address carefully.
        
        Privacy remains one of the most pressing challenges in the AI era. As AI systems require enormous datasets to function effectively, there are growing concerns about how personal information is collected, stored, and used. Many AI applications collect data from users without their explicit consent, creating potential vulnerabilities for privacy breaches.
        
        Bias in AI systems represents another critical issue that needs immediate attention. AI algorithms learn from historical data, which often contains inherent biases reflecting societal inequalities. When these biased datasets are used to train AI models, the resulting systems can perpetuate and even amplify existing discrimination in areas such as hiring, lending, and law enforcement.
        
        Despite these challenges, AI continues to offer tremendous benefits for society. In healthcare, AI-powered diagnostic tools can detect diseases earlier and more accurately than human doctors in many cases. In transportation, autonomous vehicles promise to reduce traffic accidents and improve mobility for disabled individuals. The key to harnessing AI's potential lies in developing robust governance frameworks and ensuring transparent, explainable AI systems.
        """,
        
        "Climate Change": """
        Climate change represents one of the most pressing challenges facing humanity in the 21st century. The overwhelming scientific consensus confirms that human activities, particularly the burning of fossil fuels, are the primary drivers of current climate change. Global temperatures have risen by approximately 1.1 degrees Celsius since pre-industrial times, leading to widespread environmental consequences.
        
        The effects of climate change are already visible across the globe. Rising sea levels threaten coastal communities and island nations. More frequent and severe weather events, including hurricanes, droughts, and heat waves, are causing billions of dollars in damage and displacing millions of people. Arctic ice is melting at unprecedented rates, contributing to further sea level rise and disrupting global weather patterns.
        
        Addressing climate change requires immediate and coordinated global action. The transition to renewable energy sources such as solar, wind, and hydroelectric power is essential for reducing greenhouse gas emissions. Energy efficiency improvements in buildings and transportation systems can also play a significant role in mitigation efforts. Additionally, protecting and restoring forests and other natural carbon sinks is crucial for absorbing atmospheric carbon dioxide.
        
        International cooperation through agreements like the Paris Climate Accord provides a framework for collective action. However, individual countries must implement ambitious policies to meet their emission reduction targets. This includes carbon pricing mechanisms, investments in clean technology, and support for communities affected by the transition away from fossil fuels.
        """,
        
        "Space Exploration": """
        Space exploration has entered a new era of unprecedented innovation and commercial involvement. Private companies like SpaceX, Blue Origin, and Virgin Galactic have revolutionized space travel by developing reusable rockets and reducing launch costs dramatically. This commercialization of space has opened new possibilities for scientific research, telecommunications, and even space tourism.
        
        NASA's Artemis program aims to return humans to the Moon by the mid-2020s, establishing a sustainable lunar presence that will serve as a stepping stone for future Mars missions. The program represents international collaboration, with partners from Europe, Japan, and Canada contributing essential components and expertise. The Moon will serve as a testing ground for technologies needed for deep space exploration.
        
        Mars exploration continues to yield fascinating discoveries about the Red Planet's history and potential for past or present life. The Perseverance rover is currently collecting samples that will be returned to Earth in future missions, potentially providing definitive answers about Martian biology. Additionally, the successful deployment of the Ingenuity helicopter demonstrated powered flight on another planet for the first time.
        
        Beyond our solar system, the James Webb Space Telescope is revolutionizing our understanding of the universe. Its unprecedented infrared capabilities allow scientists to observe the most distant galaxies and study the formation of stars and planetary systems. These observations are helping answer fundamental questions about the origin and evolution of the cosmos, while also identifying potentially habitable exoplanets.
        """
    }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        if example_choice != "None" and example_choice in examples:
            default_text = examples[example_choice]
        else:
            default_text = ""
        
        input_text = st.text_area(
            "📄 Enter your text to summarize:",
            value=default_text,
            height=300,
            placeholder="Paste your text here... (minimum 100 words recommended)"
        )
        
        # Summarize button
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if not input_text.strip():
                st.error("Please enter some text to summarize!")
            elif len(input_text.split()) < 20:
                st.warning("Text is quite short. Consider adding more content for better summarization.")
            else:
                # Initialize summarizer
                summarizer = initialize_summarizer()
                
                # Generate summaries
                with st.spinner("🔄 Generating summaries... Please wait."):
                    try:
                        summaries = summarizer.summarize(
                            input_text,
                            method=selected_method,
                            summary_ratio=summary_ratio,
                            max_length=max_length,
                            min_length=min_length
                        )
                        st.session_state.summaries = summaries
                        
                        # Display success message
                        st.success("✅ Summaries generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating summaries: {str(e)}")
                        st.info("💡 Try reducing the text length or check your internet connection.")
    
    with col2:
        # Information panel
        st.markdown("""
        ### 📖 How it Works
        
        **🔹 Extractive Summarization:**
        - Selects the most important sentences from the original text
        - Uses TextRank algorithm with TF-IDF vectors
        - Preserves original phrasing and terminology
        
        **🔹 Abstractive Summarization:**
        - Generates new sentences that capture the essence
        - Uses advanced transformer models (BART/T5)
        - May paraphrase and restructure content
        
        ### ⚙️ Tips for Best Results:
        - Use texts with at least 100-200 words
        - Ensure text is well-structured with clear sentences
        - Try both methods to compare results
        - Adjust settings based on your needs
        """)
    
    # Display results
    if st.session_state.summaries:
        st.markdown("---")
        st.header("📋 Summary Results")
        
        # Display statistics
        display_summary_stats(input_text, st.session_state.summaries)
        
        # Display summaries
        if 'extractive' in st.session_state.summaries:
            st.markdown("### 📝 Extractive Summary")
            st.markdown(f"""
            <div class="summary-box">
                {st.session_state.summaries['extractive']}
            </div>
            """, unsafe_allow_html=True)
        
        if 'abstractive' in st.session_state.summaries:
            st.markdown("### 🧠 Abstractive Summary")
            st.markdown(f"""
            <div class="summary-box">
                {st.session_state.summaries['abstractive']}
            </div>
            """, unsafe_allow_html=True)
        
        # Download results
        if st.button("📥 Download Results", use_container_width=True):
            results_text = f"""
Text Summarization Results
==========================

Original Text ({len(input_text.split())} words):
{input_text}

{'='*50}

"""
            if 'extractive' in st.session_state.summaries:
                results_text += f"""
Extractive Summary ({len(st.session_state.summaries['extractive'].split())} words):
{st.session_state.summaries['extractive']}

{'='*50}

"""
            if 'abstractive' in st.session_state.summaries:
                results_text += f"""
Abstractive Summary ({len(st.session_state.summaries['abstractive'].split())} words):
{st.session_state.summaries['abstractive']}
"""
            
            st.download_button(
                label="📁 Download as Text File",
                data=results_text,
                file_name="summary_results.txt",
                mime="text/plain"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🎯 AI Text Summarizer | Built with Streamlit, transformers, and scikit-learn</p>
        <p>💡 For technical details, check out the <code>text_summarizer.py</code> file</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
