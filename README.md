# Retrival Augmented Generation

## Document description
I chose the Wikipedia Page on Machine Learning to learn different concepts and get a general overview of the topic.

## Reflection on Key Aspects of the RAG System
***1. Embedding Dimensionality in FAISS Index***
- The embedding dimensionality corresponds to the vector size output by the sentence-transformers model. FAISS requires all vectors in the index to have the same dimension to accurately compute distances (like Euclidean L2). Matching dimensionality ensures correct and efficient similarity search, preventing errors and improving retrieval precision.

***2. FAISS IndexFlatL2 Search Behavior***
- IndexFlatL2 performs exact nearest neighbor searches using Euclidean distance, guaranteeing true closest matches. It is straightforward and accurate but may be slower for very large datasets since it doesn’t approximate or compress vectors. This makes it suitable for moderate-sized datasets where search accuracy is prioritized over speed.

***3. Role of Chunk Overlap in Text Splitting***
- Chunk overlap (e.g., 50 tokens) maintains context continuity across text splits, preventing important information from being fragmented. Overlapping chunks enhance semantic embedding quality and improve retrieval results by ensuring that chunks contain coherent, context-rich text.

***4. Importance of Prompt Design for Text Generation***
- Well-designed prompts that clearly separate context from the user’s question guide the language model to generate accurate and relevant answers. Including retrieved chunks as context helps reduce hallucinations. Prompt formatting and length directly influence model performance and output quality.

***5. Selecting the top_k Value for Retrieval***
- The top_k parameter controls how many top-matching chunks are fetched from the index. A higher top_k can provide richer context but increases the input size to the language model, potentially causing token limits or longer inference times. Finding the right balance is key to effective and efficient question answering.

## Analysis
While experimenting with different chunk sizes and overlaps in my retrieval-augmented generation (RAG) system, I found that the quality of the answers varied a lot depending on how the text was split. When I used a very small chunk size of 10 tokens with either 1 or 10 tokens of overlap, the answers were really short and unhelpful. For example, when I asked "What is ANN?" the system just replied with one word like "neuron" or repeated the word "ANN" from the question, without adding any useful information. This showed me that small chunks don’t provide enough context for the model to understand the question properly. On the other hand, when I used a large chunk size like 1800 tokens, the answers got much better. The model was able to pull useful information from the text and give complete explanations. However, the answers were sometimes too long and even got cut off before they finished, especially when the overlap was high (like 900 tokens), since it retrieved almost the whole paragraph or section. I also noticed that some of the text was poorly parsed—some words were stuck together without spaces, which probably confused the model. To improve this system, I think better text parsing would help a lot. It might also be useful to find a balance in chunk size—big enough to include useful context, but not so big that the response gets too long. Overall, I learned that how we split and prepare the text plays a big role in how well the system can understand and answer questions.