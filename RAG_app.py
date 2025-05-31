import logging
import warnings
from transformers import logging as hf_logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline

# Set log levels
logging.getLogger('langchain.text_splitter').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
hf_logging.set_verbosity_error()

# Filter out all Python warnings
warnings.filterwarnings('ignore')

chunk_size = 1800
chunk_overlap = 900
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 5

# Read the content of Selected_Document.txt into the variable 'text'
with open('Selected_Document.txt', 'r', encoding='utf-8') as file:
    text = file.read()
    
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', ' ', ''],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

chunks = text_splitter.split_text(text)

# Load the model
model = SentenceTransformer(model_name)

# Encode chunks with hidden progress bar
embeddings = model.encode(chunks, show_progress_bar=False)

# Convert to NumPy float32 array
embeddings = np.array(embeddings).astype('float32')

# Initialize FAISS index with correct dimension
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(embeddings)

generator = pipeline('text2text-generation', model='google/flan-t5-small', device=-1)

def retrieve_chunks(question, k=top_k):
    # Encode the question
    question_embedding = model.encode([question]).astype('float32')
    # Search FAISS index for top k closest chunks
    distances, indices = index.search(question_embedding, k)
    # Return the corresponding chunks
    return [chunks[i] for i in indices[0]]

def answer_question(question):
    # Retrieve relevant chunks
    relevant_chunks = retrieve_chunks(question)
    # Build context prompt by joining retrieved chunks (you can customize the prompt as needed)
    context = "\n\n".join(relevant_chunks)
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    
    # Generate answer using the text2text-generation pipeline
    generated = generator(prompt, max_length=200, do_sample=False)
    
    # Return the generated text
    return generated[0]['generated_text']

if __name__ == "__main__":
    print("Enter 'exit' or 'quit' to end.")
    while True:
        question = input("Your question: ")
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        answer = answer_question(question)
        print("Answer:", answer)
