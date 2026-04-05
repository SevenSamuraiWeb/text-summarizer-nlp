import math
import re
from string import capwords
import networkx as nx
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from typing import List
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import streamlit as st
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

stop_words = set(stopwords.words("english"))

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def draw_graph(sim_matrix:np.ndarray,sentences:List[str]):
        G = nx.Graph()
        for i in range(len(sentences)-1):
            G.add_node(i, label=sentences[i])

        for i in range(len(sentences)):
            for j in range(i+1, len(sim_matrix)):
                if sim_matrix[i][j] >= 0.2:
                    G.add_edge(i, j, weight=float(sim_matrix[i][j]))

        pos = nx.spring_layout(G, seed=42)

        edge_x, edge_y = [], []
        for i, j in G.edges():
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, 
            line=dict(width=0.5, color='#888'),
            hoverinfo='none', 
            mode='lines'
        )

        node_x, node_y, text = [], [], []
        for i in G.nodes():
            x, y = pos[i]
            node_x.append(x)
            node_y.append(y)
            text.append(sentences[i])

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[str(i) for i in G.nodes()],
            hovertext=text,
            hoverinfo='text',
            textposition="top center"
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        return G,fig


def get_abs_summary(input_text : str,model_name:str,mini:int=0,maxi:int=0):
    models = ["facebook/bart-large-cnn","google/flan-t5-base","t5-base"]
    if model_name in models:
        summarizer = pipeline(task="summarization",model=model_name)
        if input_text:
            if summarizer.tokenizer is None:
                return "Error"
            tokens = len(summarizer.tokenizer.encode(input_text))
            if mini <= 0:
                mini = max(30, int(tokens * 0.15))
            if maxi <= 0:    
                maxi = min(300, int(tokens * 0.5))
            abs_summary = summarizer(
                input_text, 
                max_length=maxi, 
                min_length=mini, 
                do_sample=True,
                temperature=0.7,
                no_repeat_ngram_size=3,
                repetition_penalty=2.0
            )
            if isinstance(abs_summary, list) and len(abs_summary) > 0 and isinstance(abs_summary[0], dict):
                return abs_summary[0]['summary_text']
            else:
                return "Error"
        else:
            return "Input text not provided"
    else:
        return "Model not available. Choose from 'facebook/bart-large-cnn','google/flan-t5-base' or 't5-base'"
    

def get_textrank_embed_summary(input_text : str):
    sentences = nltk.sent_tokenize(input_text)            
    cleaned = []
    for sentence in sentences:
        tokens = [re.sub(r'[^\w\s]', '', w.lower()) 
                  for w in sentence.split() 
                  if w.lower() not in stop_words]
        cleaned_sentence = " ".join(t for t in tokens if t)
        cleaned.append(cleaned_sentence)
    model = load_model()
    x = model.encode(cleaned)
    sim_matrix = cosine_similarity(x)
    G,_ = draw_graph(sim_matrix,sentences)
    if len(G.edges()) > 0:
        scores = nx.pagerank(G, weight='weight')
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        N = min(3, len(sentences))
        top_idx = [i for i, _ in ranked[:N]]
        ext_summary = " ".join([sentences[i] for i in sorted(top_idx)])
        return ext_summary
    else:
        return "Error"


def get_textrank_word_summary(input_text : str):
    if input_text:
        sentences = nltk.sent_tokenize(input_text)
        cleaned_sentences = []
        for sentence in sentences:
            words = re.sub(r'[^\w\s]', '', sentence.lower()).split()
            filtered_words = " ".join([w for w in words if w not in stop_words])
            cleaned_sentences.append(filtered_words)

        def calculate_similarity(sent1_words, sent2_words):
            common_words = set(sent1_words).intersection(set(sent2_words))
            if len(common_words) == 0:
                return 0
            denominator = math.log(len(sent1_words)) + math.log(len(sent2_words))
            if denominator <= 0: 
                return 0
            return len(common_words) / denominator

        n = len(sentences)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    sim_matrix[i][j] = calculate_similarity(cleaned_sentences[i], cleaned_sentences[j])
        
        G,_ = draw_graph(sim_matrix,sentences)           
        if len(G.edges()) > 0:
            scores = nx.pagerank(G, weight='weight')
            ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
            N = min(3, len(sentences))
            top_sentences = sorted(ranked_sentences[:N], key=lambda x: sentences.index(x[1]))
            ext_summary = " ".join([s[1] for s in top_sentences])
            return ext_summary
        else:
            return "No overlapping words found"