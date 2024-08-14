import re
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from transformers import BertTokenizer, BertModel


# Function to clean text
def text_cleaning(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text) # Removes symbols
    text = re.sub(r'\s+', ' ', text) # Removes whitespaces
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer() #changes words to their most basic forms
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Breaks text into words or subwords
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings for similarity matching
def get_bert_embeddings_similarity(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512) #Returns PyTorch tensors
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1) #computes the mean of all tokens
    return embeddings.detach().numpy()

# Function to get BERT embeddings for classification
def get_bert_embeddings_classification(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy() #extracts embeddings to the CLS tokken
    return embeddings
