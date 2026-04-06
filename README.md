# Text Summarization NLP Project

A comprehensive NLP tool built with [Streamlit](https://streamlit.io/) that performs advanced text summarization using **Extractive**, **Abstractive**, and **LLM-based** techniques. 

The application provides an interactive interface to summarize text, visualize sentence relationships using Knowledge Graphs, evaluate multiple summarization models, and conduct statistical analysis against human-generated highlights from the CNN/DailyMail dataset.

## 🚀 Features

### Summarization Methods
- **Extractive Summarization (TextRank)**
  - **Sentence Embeddings**: Uses `all-MiniLM-L6-v2` via `sentence-transformers` with Cosine Similarity to identify key sentences
  - **Word Overlap**: Calculates sentence importance based on common vocabulary overlap using custom similarity metrics
  
- **Abstractive Summarization**
  - Multiple pre-trained models: `facebook/bart-large-cnn`, `google/flan-t5-base`, `t5-base`
  - Generates human-like concise summaries from scratch
  - Configurable min/max length parameters
  
- **LLM Summarization**
  - OpenRouter API integration with Gemma and other open-source LLMs
  - System prompts for controlled output
  - Free-tier friendly with optimized token usage

### Evaluation & Analysis
- **Interactive Visualizations**
  - Knowledge Graph visualization (nodes = sentences, edges = similarity scores)
  - ROUGE metrics comparison with Plotly charts
  - Compression ratio analysis
  - Execution time benchmarking
  
- **Statistical Analysis**
  - Single sample evaluation with detailed metrics
  - Batch evaluation on CNN/DailyMail samples (Future Update)
  - Mean and standard deviation calculations across metrics (Future Update)
  - Win-loss comparison tables showing model performance distribution (Future Update)
  
- **ROUGE Metrics**
  - Supports ROUGE-1, ROUGE-2, ROUGE-L
  - Selectable metric type: F1 Score, Precision, or Recall
  - Detailed comparison tables with compression ratios

- **Dataset Integration**: Real-world evaluation against the CNN/DailyMail dataset with 13,000+ test samples

## 🛠️ Tech Stack & Libraries

- **Frontend/App Framework**: Streamlit
- **Machine Learning & NLP**: NLTK, Sentence-Transformers, Hugging Face `transformers`, Scikit-learn
- **Data & Metrics**: Pandas, NumPy, Hugging Face `datasets`, `rouge_score`
- **Visualization**: NetworkX, Plotly
- **API Integration**: OpenAI SDK (for OpenRouter API access)
- **Environment Management**: python-dotenv

## 📦 Installation

1. **Clone the repository** (or download the project files):
   ```bash
   git clone <your-repo-url>
   cd "nlp project"
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables**:
   Create a `.env` file in the project root for LLM integration:
   ```bash
   OPENROUTER_API_KEY=your_api_key_here
   ```
   
   Get a free API key from [OpenRouter](https://openrouter.ai)

5. **Run Necessary NLTK Downloads**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## 💻 Usage

### Running the Application

```bash
streamlit run main.py
```

Navigate to `http://localhost:8501` in your browser.

### Available Modes

1. **Embeddings Mode** (Extractive)
   - TextRank using sentence embeddings
   - Displays similarity matrix and knowledge graph
   - Real-time summarization of custom input

2. **Word Overlap Mode** (Extractive)
   - TextRank using word overlap similarity
   - Shows sentence relationships based on vocabulary
   - Fast execution on any text

3. **Abstractive Mode**
   - Choose from BART, T5, or Flan-T5 models
   - Configurable min/max length
   - Generates entirely new summary text

4. **LLM Mode**
   - OpenRouter API integration
   - Professional LLM-based summarization
   - Requires valid API key in `.env`

5. **Compare Mode** (Main Evaluation)
   - **Single Sample**: Load random CNN/DailyMail articles and compare all methods side-by-side
   - **Batch Results**: Pre-computed statistics on 100+ samples with:
     - Mean ± standard deviation for all metrics
     - Win-loss comparison showing which model performs best
     - Execution time benchmarks
     - ROUGE score distributions across samples

### Batch Evaluation (Recommended)
For comprehensive statistical analysis:
```bash
python precompute_stats.py  # Precompute results for 100 samples
streamlit run main.py       # View results in web interface
```

## 📁 Repository Structure

- `main.py`: The entry point and main Streamlit UI for the application.
- `functions.py`: Contains core NLP backend logic and graph generation functions.
- `lib/`: Contains external JavaScript/CSS dependencies for rendering interactive graphs (e.g., vis-network, tom-select).
- `requirements.txt`: Python package dependencies.
- `README.md`: This project documentation.

## 📝 License

This project is open-source and available for educational and developmental purposes.