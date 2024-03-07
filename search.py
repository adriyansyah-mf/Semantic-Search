import csv
import string
import re
from sentence_transformers import SentenceTransformer
from ftfy import fix_text
from sentence_transformers import util
import torch

# load the data from the csv file
def load_data():
    data = []
    with open('data.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            comment = row[1]
            data.append(comment)
    return data

def clean_data(data):
    #remove first row
    data = data[1:]
    
    data = [comment.lower() for comment in data]
    
    data = [comment.translate(str.maketrans('', '', string.punctuation)) for comment in data]
    #remove emojis
    data = [re.sub(r'[^\x00-\x7F]+','', comment) for comment in data]
    #remove \n characters
    data = [comment.replace('\n', ' ') for comment in data]
    
    
    return data

def main():
    
    data = load_data()
    data = clean_data(data)
    passages = [fix_text(chunk.strip()) for chunk in data if len(chunk.strip()) > 100]
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(passages, convert_to_tensor=True)
    query_text = [
    "02",
    ]
    query = model.encode(query_text, convert_to_tensor=True)
    cos_scores = util.cos_sim(query, embeddings)
    cos_scores.shape
    
    top_results = torch.topk(cos_scores, k=1, dim=-1)
    indices = top_results.indices.tolist()
    for i, result in enumerate(indices):
        print(f"Query: {query_text[i]}")
        for idx in result:
            print(f"Passage: {passages[idx]}")
        print()
    
    
print(main())
    
    