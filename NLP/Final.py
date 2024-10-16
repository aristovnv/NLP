import pandas as pd
import os
import ast
import spacy
import re
from spacy.tokens import Token
import json
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer, util
from itertools import chain

# Load English language model
dataPath = "/home/gridsan/naristov/NLP/Data"
#nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_md")
unwanted_terms = {'quot', 'tk', 'nan', 'gctba-', 'epdm'}  # Set of terms to remove
special_cases = [
    {"ORTH": "u-turn", "NORM": "u-turn", "POS": "NOUN"},  # Example: "U-turn"
    {"ORTH": "u-bolt", "NORM": "u-bolt", "POS": "NOUN"},   # Example: "U-bolt"
    {"ORTH": "o-ring", "NORM": "o-ring", "POS": "NOUN"},   # Example: "U-bolt"
    {"ORTH": "zinc-plated", "NORM": "zinc-plated", "POS": "NOUN"}   # Example: "U-bolt"
    
]
# Add special cases to the tokenizer
for case in special_cases:
    nlp.tokenizer.add_special_case(case["ORTH"], [{ "ORTH": case["ORTH"], "NORM": case["NORM"] }])
def clean_nouns(nouns_str):
    # Remove HTML entities and other unwanted characters
    cleaned = re.sub(r'&[a-z]+;', '', nouns_str)  # Remove HTML entities
    cleaned = re.sub(r'[0-9]', '', cleaned)  # Remove numbers

    # Preserve single-letter words that are part of hyphenated words
    cleaned = re.sub(r'\b(?<!-)\b[a-zA-Z]\b(?!-)\b', '', cleaned)  # Remove single-letter words not part of hyphenated words

    # Split the words, remove extra spaces, and join them back
    words = cleaned.split()
    cleaned_nouns = " ".join(words)  # Join back with a single space

    return cleaned_nouns

def replace_abbreviations(text):
    replacements = {
        'thrd': 'thread',
        'sz': 'size',
        'wd': 'width',
        'ss': 'stainless steel',
        'u-bolt': 'u bolt'
    }
    
    pattern = re.compile(r'\b(' + '|'.join(replacements.keys()) + r')\b')
    return pattern.sub(lambda x: replacements[x.group()], text) 

def remove_tk_words(text):
    doc = nlp(text)
    # Define a regex pattern to match words starting with 'tk' followed by letters and numbers
    pattern = re.compile(r'\btk[a-zA-Z0-9]+\b', re.IGNORECASE)
    
    # Filter out tokens that match the pattern
    filtered_tokens = [token.text for token in doc if not pattern.match(token.text)]
    
    # Join the filtered tokens back into a single string
    cleaned_text = ' '.join(filtered_tokens)
    
    return cleaned_text

    
def extract_valid_nouns(text):
    text = text.lower()
    text = remove_tk_words(text)
    # Create a spaCy Doc from the text
    doc = nlp(text)
    filtered_tokens = doc #[token for token in doc if not token.is_oov]
    # Extract nouns, proper nouns, and adjectives
    terms = [token.text for token in filtered_tokens if token.pos_ in ["NOUN", "PROPN", "ADJ", "ADP"]]
    

    # Clean the extracted terms
    cleaned_terms = [
        term for term in terms
        if re.match(r"^[a-zA-Z-]+$", term) and term.lower() not in unwanted_terms  # Exclude unwanted terms
    ]

    # Join the cleaned terms into a single string
    cleaned_terms_str = ' '.join(cleaned_terms)

    # Replace abbreviations with full words
    

    # Remove unwanted characters but preserve hyphens in hyphenated terms
    cleaned_terms_str = re.sub(r'(?<!\w)-|-(?!\w)', '-', cleaned_terms_str)  # Preserve hyphens
    cleaned_terms_str = replace_abbreviations(cleaned_terms_str)
    cleaned_terms_str = re.sub(r'[^a-zA-Z0-9- ]', '', cleaned_terms_str)  # Remove unwanted characters except hyphens
    #cleaned_terms_str = re.sub(r'(?<=\w)-\s*|-\s*(?=\w)', '-', cleaned_terms_str)
    cleaned_terms_str = ' '.join(cleaned_terms_str.split())

    return cleaned_terms_str

df = pd.read_csv(f"{dataPath}/MRO 2024.csv")
df_work = df.copy()

df = df_work.copy()
df = df[['Item Description', 'PO Description' ]]
print(df.columns)

#df = df.head(10)
df["nouns_cleaned"] = df['Item Description'].apply(clean_nouns)
#df["nouns_ID_cleaned"] = df["nouns_ID"].apply(clean_nouns)
df["nouns_cleaned_a"] = df["nouns_cleaned"].apply(extract_valid_nouns) 

#tar = pd.read_excel(f"{dataPath}/ds_our.xlsx") 
tar = pd.read_csv(f"{dataPath}/data-unspsc-codes.csv")

tar['low_case'] = tar['Commodity Name'].str.lower()

#from itertools import chain

modelPath = "/home/gridsan/naristov/NLP/model"

model = SentenceTransformer(modelPath)


# Generate n-grams from a list of words
def generate_ngrams(words, n):
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

# Create a function to get all n-grams (1, 2, and 3-grams)
def get_all_ngrams(text):
    words = text.split()
    ngrams = list(chain.from_iterable(generate_ngrams(words, n) for n in range(1, 4)))
    return ngrams

df['ngrams'] = df['nouns_cleaned_a'].apply(get_all_ngrams)

# Compute embeddings for commodities
tar['embedding'] = tar['low_case'].apply(lambda x: model.encode(x, convert_to_tensor=True))

# Convert tar embeddings to a tensor
tar_embeddings = torch.stack(tar['embedding'].tolist())

# Debugging: Print shapes of embeddings
print("Shape of tar_embeddings:", tar_embeddings.shape)
'''
# Set a threshold for matching
threshold = 0.7

# Function to get best matches for n-grams in a row
def get_best_matches(ngrams, tar_embeddings, tar_commodities):
    ngram_embeddings = model.encode(ngrams, convert_to_tensor=True)
    similarities = util.cos_sim(ngram_embeddings, tar_embeddings)
    
    best_matches = []
    for i, sim_scores in enumerate(similarities):
        matching_indices = torch.where(sim_scores > threshold)[0]
        for match_idx in matching_indices:
            match_idx = int(match_idx)  # Convert match_idx to integer
            match = tar_commodities.iloc[match_idx]
            score = sim_scores[match_idx].item()
            if score > threshold:
                best_matches.append((ngrams[i], match, score))
    return best_matches

# Apply the function to each row in df
df['matches'] = df['ngrams'].apply(lambda ngrams: get_best_matches(ngrams, tar_embeddings, tar['Commodity']))

# Print results
print(df[['Item Description', 'matches']])
'''

# Set thresholds for matching
threshold_all = 0.7
threshold_top_n = 0.9
top_n = 5
# Function to get best matches for n-grams in a row
# Function to get best matches for n-grams in a row
def get_best_matches_by_ngram_size(ngrams, tar_embeddings, tar_commodities, threshold_all, threshold_top_n, top_n):
    results = {1: [], 2: [], 3: []}
    
    for n in range(1, 4):
        ngram_list = [ngram for ngram in ngrams if len(ngram.split()) == n]
        if not ngram_list:
            continue
        
        ngram_embeddings = model.encode(ngram_list, convert_to_tensor=True)
        similarities = util.cos_sim(ngram_embeddings, tar_embeddings)
        
        all_matches = []
        top_n_matches = []
        
        for i, sim_scores in enumerate(similarities):
            # Ensure sim_scores is a 1-dimensional tensor
            sim_scores = sim_scores.flatten()
            
            # Get all matches with similarity > threshold_all
            matching_indices_all = torch.where(sim_scores > threshold_all)[0]
            for match_idx in matching_indices_all:
                match_idx = int(match_idx)  # Convert match_idx to integer
                match = tar_commodities.iloc[match_idx]
                score = sim_scores[match_idx].item()
                all_matches.append((ngram_list[i], match, score))
            
            # Get top N matches sorted by similarity score
            matching_indices_top_n = torch.where(sim_scores > threshold_top_n)[0]
            for match_idx in matching_indices_top_n:
                match_idx = int(match_idx)  # Convert match_idx to integer
                match = tar_commodities.iloc[match_idx]
                score = sim_scores[match_idx].item()
                top_n_matches.append((ngram_list[i], match, score))
        
        # Sort top N matches by similarity score
        top_n_matches = sorted(top_n_matches, key=lambda x: x[2], reverse=True)[:top_n]
        
        results[n] = {'all_matches': all_matches, 'top_n_matches': top_n_matches}
    
    return results

# Apply the function to each row in df
df['matches_by_ngram'] = df['ngrams'].apply(lambda ngrams: get_best_matches_by_ngram_size(ngrams, tar_embeddings, tar['Commodity'], threshold_all, threshold_top_n, top_n))
'''
# Extract top N matches into separate columns for each n-gram size
for n in range(1, 4):
    df[f'top_n_matches_{n}_gram'] = df['matches_by_ngram'].apply(lambda x: x[n]['top_n_matches'])
    df[f'high_confidence_matches_{n}_gram'] = df['matches_by_ngram'].apply(lambda x: [match for match in x[n]['all_matches'] if match[2] > threshold_top_n])

# Print results
print(df[['Item Description', 'top_n_matches_1_gram', 'high_confidence_matches_1_gram', 'top_n_matches_2_gram', 'high_confidence_matches_2_gram', 'top_n_matches_3_gram', 'high_confidence_matches_3_gram']])
'''

def extract_top_n_and_high_confidence(matches, n, threshold_top_n):
    if isinstance(matches, dict) and n in matches:
        ngram_matches = matches[n]
        if isinstance(ngram_matches, dict):
            top_n_matches = ngram_matches['top_n_matches']
            high_confidence_matches = [match for match in ngram_matches['all_matches'] if match[2] > threshold_top_n]
            return top_n_matches, high_confidence_matches
    return [], []

threshold_top_n = 0.9

# Initialize new columns in the DataFrame
for n in range(1, 4):
    df[f'top_n_matches_{n}_gram'] = None
    df[f'high_confidence_matches_{n}_gram'] = None

# Extract top N matches and high confidence matches for each n-gram size
for n in range(1, 4):
    df[[f'top_n_matches_{n}_gram', f'high_confidence_matches_{n}_gram']] = df['matches_by_ngram'].apply(
        lambda matches: pd.Series(extract_top_n_and_high_confidence(matches, n, threshold_top_n))
    )

print(df)
#6/25 8:50
df.to_csv(f'{dataPath}/res3.csv')
#df_check