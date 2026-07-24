from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

# Test
vec = embedding.embed_query("test RAG")
print("Vector dimension:", len(vec))  # Output of 1024 means success!
