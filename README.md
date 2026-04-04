# Text Summarization NLP Project

A comprehensive NLP tool built with [Streamlit](https://streamlit.io/) that performs advanced text summarization using both **Extractive** and **Abstractive** techniques. 

The application provides an interactive interface to summarize text, visualize sentence relationships using a Knowledge Graph, and compare different summarization models against human-generated highlights from the CNN/DailyMail dataset.

## 🚀 Features

- **Extractive Summarization (TextRank)**: 
  - **Sentence Embeddings**: Utilizes `all-MiniLM-L6-v2` via `sentence-transformers` and Cosine Similarity to identify key sentences.
  - **Word Overlap**: Calculates sentence importance based on common vocabulary overlap.
- **Abstractive Summarization**: 
  - Leverages Hugging Face Transformers (`facebook/bart-large-cnn`, `google/flan-t5-base`, `t5-base`) to generate human-like concise summaries from scratch.
- **Interactive Visualizations**: 
  - Visualizes text structure as a Knowledge Graph (nodes as sentences, edges as similarity scores) using `vis-network`.
  - Compare model performances using ROUGE (1, 2, L) scores visualized with Plotly.
- **Dataset Integration**: Automatically loads and compares results against real-world news articles from the `cnn_dailymail` dataset.

## 🛠️ Tech Stack & Libraries

- **Frontend/App Framework**: Streamlit
- **Machine Learning & NLP**: NLTK, Sentence-Transformers, Hugging Face `transformers`, Scikit-learn
- **Data & Metrics**: Pandas, Hugging Face `datasets`, `rouge_score`
- **Graphs & Visualization**: NetworkX, Plotly, custom PyVis / HTML-JS components

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

4. **Run Necessary NLTK Downloads**:
   Ensure you have downloaded the required NLTK data before running the application. You can do this in a python shell:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## 💻 Usage

Run the Streamlit application from your terminal:

```bash
streamlit run main.py
```

- Navigate to the provided local URL (typically `http://localhost:8501`).
- Use the sidebar to select your preferred summarization method:
  - **Embeddings**: For semantic extraction.
  - **Word Overlap**: For traditional word-count-based extraction.
  - **Abstractive**: To generate natural-sounding summaries with LLMs.
  - **Compare**: To evaluate all methods against the CNN/DailyMail dataset using ROUGE metrics.

## 📁 Repository Structure

- `main.py`: The entry point and main Streamlit UI for the application.
- `functions.py`: Contains core NLP backend logic and graph generation functions.
- `lib/`: Contains external JavaScript/CSS dependencies for rendering interactive graphs (e.g., vis-network, tom-select).
- `requirements.txt`: Python package dependencies.
- `README.md`: This project documentation.

## 📝 License

This project is open-source and available for educational and developmental purposes.