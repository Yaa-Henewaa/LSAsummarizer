import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
import os
import numpy as np
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import json
import string
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
  


class TextData(BaseModel):
    data: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load('LoRmodel.pkl')
vectorizer = joblib.load('tfidfLoR_vectorizer.pkl') 

@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the Text Categorization and Summarization API!",
        "endpoints": {
            "/": "GET - This message",
            "/predict": "POST - Submit text data for categorization and summarization",
        },
        "description": "This API allows you to categorize text into predefined categories and summarize it using LSA.",
        "model_info": {
            "model_type": "Naive Bayes",
            "vectorizer": "TF-IDF",
        }
    }

@app.post("/predict")
async def predict(data: dict):
    try:
        result = pipeline(data['data'])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)



def pipeline(document):
    # Split the document into paragraphs
    paragraphs = document.strip().split('\n\n')

    # Transform paragraphs into the same format used during training
    paragraphs_transformed = vectorizer.transform(paragraphs)

    # Predict categories
    predictions = model.predict(paragraphs_transformed)

    # Summarize by category
    summary_by_category = summarize_by_category(paragraphs, predictions)

    # return summary_by_category
    result = {}

    for category, summary in summary_by_category.items():
        result[category] = summary 

    return result






def lsa_summarize(text, num_sentences=1):


    sentences = text.split('.')
    sentences = [sent.strip() for sent in sentences if sent.strip()]
    
    if len(sentences) <= num_sentences:
        return '. '.join(sentences)
    
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    transformer = TfidfTransformer()
    X_tfidf = transformer.fit_transform(X)
    
    svd = TruncatedSVD(n_components=min(num_sentences, X_tfidf.shape[0]-1))
    X_svd = svd.fit_transform(X_tfidf)
    
    top_sentence_indices = np.argsort(X_svd[:, 0])[::-1][:num_sentences]
    top_sentence_indices = sorted(top_sentence_indices)
    
    summary = '. '.join([sentences[i] for i in top_sentence_indices])
    
    return summary

def spacy_summarize(text, num_sentences = 1):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = list(doc.sents)
    ranked_sentences = sorted(sentences, key=lambda s: len(s.text), reverse=True)
    summary = ' '.join([s.text for s in ranked_sentences[:num_sentences]])
    return summary

def summarize_by_category(paragraphs, predictions):
    # Define the categories (if needed for reference)
    categories = ['Data Collection', 'Data Usage', 'Data Sharing', 'Data Storage', 'Rights and Protection']

    # Ensure predictions are in list format
    predictions_labels = predictions.tolist()

    # Create a DataFrame to group paragraphs by category
    df_paragraphs = pd.DataFrame({'Paragraph': paragraphs, 'Category': predictions_labels})

    summary_by_category = {}
    for category in categories:
        # Filter paragraphs for the current category
        filtered_paragraphs = df_paragraphs[df_paragraphs['Category'] == category]['Paragraph']
        
        # Combine paragraphs for the category
        combined_text = ' '.join(filtered_paragraphs)
        
        # Generate summary using LSA summarization
        summary = lsa_summarize(combined_text, num_sentences=1)
        
        # Store summary
        summary_by_category[category] = summary

    return summary_by_category




