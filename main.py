import os
import pandas as pd
import numpy as np
import openai
import logging
import requests
from pinecone import Pinecone, ServerlessSpec
from collections import deque
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template

PINECONE_API_KEY = "b2e45273-9712-42d0-bbfb-d989381add1d"
PINECONE_INDEX_NAME = "cyber-security"
PINECONE_ENVIRONMENT = "us-east-1"

pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # Dimension for Sentence Transformers (usually 768 for base models)
        metric='cosine',
        spec=ServerlessSpec(cloud='gcp', region=PINECONE_ENVIRONMENT)
    )

index = pc.Index(PINECONE_INDEX_NAME)

model = SentenceTransformer('all-MiniLM-L6-v2')

conversation_memory = deque(maxlen=3)

def upsert_vectors(vectors, namespace="default"):
    index.upsert(vectors=vectors, namespace=namespace)

def embed_text(text):
    return model.encode(text).tolist()

documents = {}

def insert_csv_vectors(csv_file_path, namespace="default"):
    df = pd.read_csv(csv_file_path)
    vectors = []
    for idx, row in df.iterrows():
        doc_id = f"vec_{idx}"
        vector = {
            "id": doc_id,
            "values": embed_text(row['Content']),
            "metadata": {"Title": row['Title']}
        }
        vectors.append(vector)
        documents[doc_id] = row['Content']
    upsert_vectors(vectors, namespace)

def query_vector_db(user_query, namespace="default"):
    query_vector = embed_text(user_query)
    response = index.query(
        namespace=namespace,
        vector=query_vector,
        top_k=2,
        include_values=True,
        include_metadata=True
    )
    relevant_doc_ids = [match['id'] for match in response['matches']]
    return relevant_doc_ids

logging.basicConfig(level=logging.INFO, filename="error_log.log", filemode="w")
API_KEY = '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'
API_URL="https://openrouter.ai/api/v1"

def basic_chatbot_conversation(user_query):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        "model": "Claude-v1",
        "messages": [{"role": "user", "content": user_query}]
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()

        logging.info(f"API Response: {response.json()}")

        response_json = response.json()

        if 'choices' in response_json and response_json['choices']:
            return response_json['choices'][0]['message']['content']
        else:
            return "Sorry, there was an issue with the response from the chatbot."

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.RequestException as err:
        logging.error(f"Request error occurred: {err}")
        return f"Request error occurred: {err}"
    except KeyError:
        logging.error("Error: Unexpected response structure from the API.")
        return "Error: Unexpected response structure from the API."

def rag_chatbot_conversation(user_query):
    relevant_doc_ids = query_vector_db(user_query)
    relevant_docs = [documents[doc_id] for doc_id in relevant_doc_ids]
    doc_context = "\n".join(relevant_docs)
    memory_context = "\n".join(conversation_memory)
    
    system_prompt = "You are a cybersecurity expert. Help users by providing cybersecurity best practices."
    conversation = f"{system_prompt}\nUser: {user_query}\nContext:\n{doc_context}\nMemory:\n{memory_context}\nAI:"
    
    response = basic_chatbot_conversation(conversation)
    conversation_memory.append(f"User: {user_query}")
    conversation_memory.append(f"AI: {response}")
    
    return response

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    bot_response = rag_chatbot_conversation(user_message)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    csv_file_path = "Cybersecurity list.csv"
    insert_csv_vectors(csv_file_path)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
