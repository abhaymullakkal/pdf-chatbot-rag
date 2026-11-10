from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv, find_dotenv
from langchain_pinecone import PineconeVectorStore
import streamlit as st

model_name ="sentence-transformers/all-MiniLM-L6-v2"
embedding=HuggingFaceEmbeddings(model=model_name)
file="machine_learning_for_pdf_chatbot (1).pdf"
data=PyPDFLoader(file)
document=data.load()
splitter=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)
chunks=splitter.split_documents(document)

load_dotenv(find_dotenv(),override=True)

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index_name = "pdf-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)


# Initialize Pinecone vector store
vector_Store = PineconeVectorStore(
    index=index,
    embedding=embedding
)

# Generate unique IDs for chunks
ids = [f"pdf-chunk-{i}" for i in range(len(chunks))]

# Check existing records in Pinecone
existing_records = index.fetch(ids=ids)

existing_ids = set(existing_records.vectors.keys())

# Identify chunks that are not stored
new_chunks = []
new_ids = []

for i, chunk in enumerate(chunks):
    if ids[i] not in existing_ids:
        new_chunks.append(chunk)
        new_ids.append(ids[i])

# Add only new chunks
if new_chunks:
    vector_Store.add_documents(
        documents=new_chunks,
        ids=new_ids
    )
    print(f"Added {len(new_chunks)} new chunks to Pinecone.")
else:
    print("All PDF chunks are already indexed.")


def ask_quesion(query):
    result=vector_Store.similarity_search(query,k=3)

    llm=ChatOllama(model="llama3.2:1b",temperature=0)

    context="\n\n".join([ doc.page_content for doc in result])
    prompt=f'''
    Answer the question using the context below.
    Context:
    {context}
    Question:
    {query}
    Answer:
    '''
    response=llm.invoke(prompt)
    return response.content


#                   ****Streamlit interface****
st.header("Chat with Your Pdf",divider="rainbow",text_alignment="center",icon="📖")
# 1. Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

# 2. Display previous conversations
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])

# 3. Accept a new question
question=st.chat_input("Enter your Question")
# 4. Generate a new answer
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking...."):
            answer=ask_quesion(question)
        st.write(answer)
    st.session_state.chat_history.append( {"question": question, "answer": answer})


