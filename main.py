import math
from typing import List
import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
import re
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objects as go
from transformers import pipeline
from datasets import load_dataset
from rouge_score import rouge_scorer
from functions import draw_graph,get_abs_summary,get_textrank_embed_summary,get_textrank_word_summary
import plotly.express as px

nltk.download('punkt')
nltk.download('stopwords')
stopwords = set(stopwords.words("english"))

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_cnndaily_samples(n:int):
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)
    return list(ds.take(n))

def calculate_metrics(target, prediction,metric:str):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(target, prediction)
    
    if metric == "F1 Score":
        return {
            "R-1": round(scores['rouge1'].fmeasure, 3),
            "R-2": round(scores['rouge2'].fmeasure, 3),
            "R-L": round(scores['rougeL'].fmeasure, 3)
        }
    elif metric == "Precision":
        return {
            "R-1": round(scores['rouge1'].precision, 3),
            "R-2": round(scores['rouge2'].precision, 3),
            "R-L": round(scores['rougeL'].precision, 3)
        }
    else:
        return {
            "R-1": round(scores['rouge1'].recall, 3),
            "R-2": round(scores['rouge2'].recall, 3),
            "R-L": round(scores['rougeL'].recall, 3)
        }


def calculate_compression_ratio(original,summary):
    return len(summary) / len(original)


with st.sidebar:
    method = st.radio(label="Summarization Method",options=["Embeddings","Word Overlap","Abstractive","Compare"])

if method == "Embeddings":
    st.title(f"Text Summarization : {method} TextRank")
    input_text = st.text_area(label="Enter your text here",height="content")
    if input_text:
        st.header("Split into sentences")
        sentences = nltk.sent_tokenize(input_text)
        for i, s in enumerate(sentences):
            st.write(f"**S{i}:** {s}")
                
        cleaned = []
        st.header("\n\nRemove stopwords and punctuation")
        for i,sentence in enumerate(sentences):
            tokens = [re.sub(r'[^\w\s]', '', w.lower()) 
                      for w in sentence.split() 
                      if w.lower() not in stopwords]
            cleaned_sentence = " ".join(t for t in tokens if t)
            cleaned.append(cleaned_sentence)
            st.write(f"**S{i}:** {cleaned_sentence}")

        st.header("\n\nVectorise sentences")
        model = load_model()
        x = model.encode(cleaned)
        df_vectors = pd.DataFrame(x)
        st.dataframe(df_vectors)

        st.header("\n\nCompute similarity matrix (Cosine Similarity)")
        sim_matrix = cosine_similarity(x)
        df_sim = pd.DataFrame(sim_matrix)
        st.dataframe(df_sim)

        st.header("\n\nCreate graph with sentences as nodes and similarity as edges")
        G,fig = draw_graph(sim_matrix,sentences)
        st.plotly_chart(fig, width='stretch')       
        st.header("\n\nRun pagerank algorithm on the graph")
        if len(G.edges()) > 0:
            damping = st.slider(label="Select damping factor",min_value=0.0,max_value=1.0)
            scores = nx.pagerank(G, weight='weight',alpha=damping)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            st.json(scores)
            N = min(3, len(sentences))
            top_idx = [i for i, _ in ranked[:N]]
            ext_summary = " ".join([sentences[i] for i in sorted(top_idx)])
            st.header("Final summary")
            st.success(ext_summary)
        else:
            st.warning("No similarity found between sentences to create a summary.")

    
elif method == "Word Overlap":
    st.title(f"Text Summarization : {method} TextRank")
    input_text = st.text_area(label="Enter your text here",height="content")
    if input_text:
        st.header("1. Split into sentences")
        sentences = nltk.sent_tokenize(input_text)
        for i, s in enumerate(sentences):
            st.write(f"**S{i}:** {s}")

        st.header("2. Remove stopwords and punctuation")
        cleaned_sentences = []
        for i,sentence in enumerate(sentences):
            words = re.sub(r'[^\w\s]', '', sentence.lower()).split()
            filtered_words = " ".join([w for w in words if w not in stopwords])
            cleaned_sentences.append(filtered_words)
            st.write(f"**S{i}:** {filtered_words}")

        def calculate_similarity(sent1_words, sent2_words):
            common_words = set(sent1_words).intersection(set(sent2_words))
            if len(common_words) == 0:
                return 0
            denominator = math.log(len(sent1_words)) + math.log(len(sent2_words))
            if denominator <= 0: 
                return 0
            return len(common_words) / denominator

        st.header("3. Compute similarity matrix (Overlapping Words)")
        n = len(sentences)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    sim_matrix[i][j] = calculate_similarity(cleaned_sentences[i], cleaned_sentences[j])
        df_sim = pd.DataFrame(sim_matrix)
        st.dataframe(df_sim)

        st.header("4. Knowledge Graph")
        G,fig = draw_graph(sim_matrix,sentences)
        st.plotly_chart(fig, width='stretch')

        st.header("5. Sentence Scores & Final Summary")            
        if len(G.edges()) > 0:
            damping = st.slider(label="Select damping factor",value=0.85,min_value=0.0,max_value=1.0)
            scores = nx.pagerank(G, weight='weight')
            ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
            st.json(scores)
            st.subheader("Summary Result")
            N = min(3, len(sentences))
            top_sentences = sorted(ranked_sentences[:N], key=lambda x: sentences.index(x[1]))
            ext_summary = " ".join([s[1] for s in top_sentences])
            st.success(ext_summary)
        else:
            st.warning("No overlapping words found between sentences to create a summary.")


elif method == "Abstractive":
    st.title("Text Summarization : Abstractive Method")
    model = st.radio(label="Choose model",options=["facebook/bart-large-cnn","google/flan-t5-base","t5-base"])
    summarizer = pipeline(task="summarization",model=model)
    input_text = st.text_area(label="Enter your text here",height="content")
    if input_text:
        min_len = st.number_input(label="Enter minimum length here",min_value=5,max_value=len(input_text))
        max_len = st.number_input(label="Enter maximum length here",min_value=5,max_value=len(input_text))
        if min_len and max_len:
            if summarizer.tokenizer is None:
                st.error("Tokenizer is not available for the selected model.")
                st.stop()
            tokens = len(summarizer.tokenizer.encode(input_text))
            abs_summary = summarizer(input_text, max_length=max_len, min_length=min_len, do_sample=True,temperature=0.7,no_repeat_ngram_size=3,repetition_penalty=2.0)
            if isinstance(abs_summary, list) and len(abs_summary) > 0 and isinstance(abs_summary[0], dict):
                st.success(abs_summary[0]['summary_text'])
            else:
                st.error("Unexpected output format from the summarization model.")
else:
    samples = get_cnndaily_samples(10)
    sample_idx = st.sidebar.selectbox("Select CNN Sample", range(len(samples)))
    rouge_metric = st.sidebar.selectbox("Select ROUGE Metric", options=["Precision","Recall","F1 Score"])
    abstractive_model = st.sidebar.selectbox("Select Abstractive Model", options=["facebook/bart-large-cnn","google/flan-t5-base","t5-base"])

    current_sample = samples[sample_idx]
    current_input = str(current_sample['article'])
    gold_summary = str(current_sample['highlights'])

    min_len = st.sidebar.number_input("Select Min Length",max_value=len(current_input))
    max_len = st.sidebar.number_input("Select Max Length",max_value=len(current_input))
    
    st.header("News article from CNN/DailyMail dataset")
    st.text(current_input,width='content')

    st.header("Highlights of the article - Ideal summary")
    st.text(gold_summary,width='content')

    col1,col2,col3 = st.columns(3,border=True)
    with col1:
        st.subheader("TextRank using Embeddings")
        embed_summary = get_textrank_embed_summary(input_text=current_input)
        st.write(embed_summary)

    with col2:
        st.subheader("TextRank using Word Overlap")
        word_summary = get_textrank_word_summary(input_text=current_input)
        st.write(word_summary)

    with col3:
        st.subheader(f"Abstractive summarization {abstractive_model}")
        abs_summary = get_abs_summary(input_text=current_input,model_name=abstractive_model,mini=min_len,maxi=max_len)
        st.write(abs_summary)

    compression_word = calculate_compression_ratio(current_input,word_summary)
    compression_embed = calculate_compression_ratio(current_input,embed_summary)
    compression_abs = calculate_compression_ratio(current_input,abs_summary)
    compression_ideal = calculate_compression_ratio(current_input,gold_summary)

    scores_word = calculate_metrics(gold_summary, embed_summary,rouge_metric)
    scores_embed = calculate_metrics(gold_summary, word_summary,rouge_metric)
    scores_abs = calculate_metrics(gold_summary, abs_summary,rouge_metric)

    comp_data = {
        "Metric" : ["Compression Ratio"] * 4,
        "Score" : [compression_word,compression_embed,compression_abs,compression_ideal],
        "Model" : ["Word Overlap","Embeddings","Abstractive","Ideal"]
    }

    df_plot = pd.DataFrame(comp_data)
    fig = px.bar(df_plot, x='Metric', y='Score', color='Model', barmode='group', title="Summary Length Comparison")
    st.plotly_chart(fig)

    score_data = {
        "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"] * 3,
        "Score": [
            scores_word['R-1'], scores_word['R-2'], scores_word['R-L'],
            scores_embed['R-1'], scores_embed['R-2'], scores_embed['R-L'],
            scores_abs['R-1'], scores_abs['R-2'], scores_abs['R-L']
        ],
        "Model": ["Word Overlap"]*3 + ["Embeddings"]*3 + ["Abstractive"]*3
    }

    df_plot = pd.DataFrame(score_data)
    fig = px.bar(df_plot, x='Metric', y='Score', color='Model', barmode='group', title="Relevance Comparison")
    st.plotly_chart(fig)

    





    
