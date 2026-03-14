# 📄 Document Chat Application (RAG using LangChain & Chroma)

This project demonstrates how to build a **Document Chat / Retrieval-Augmented Generation (RAG) system** using **LangChain**, **Chroma Vector Database**, and **HuggingFace Embeddings**.

The application loads a research paper (**"Attention is All You Need"**), converts the content into embeddings, stores them in a vector database, and retrieves relevant document chunks based on user queries.

This allows users to **ask questions about a document and get accurate answers from the document itself.**

---

# 🚀 Features

* Load and process PDF documents
* Extract document text and metadata
* Split documents into smaller chunks
* Convert text chunks into embeddings
* Store embeddings in **Chroma vector database**
* Perform **semantic similarity search**
* Retrieve relevant document context for queries

---

# 🧠 How the System Works

The project uses a **Retrieval-Augmented Generation (RAG) pipeline**.

```
PDF Document
      ↓
Document Loader (PyPDFLoader)
      ↓
Text Splitting
      ↓
Embedding Generation
      ↓
Vector Database (Chroma)
      ↓
Similarity Search
      ↓
Retrieve Relevant Context
```

---

# 📂 Project Structure

```
Document-chat-application
│
├── document_chat_rag.ipynb      # Main notebook containing the RAG pipeline
├── Attention_is_all_you_need.pdf
├── langchain_chroma_db/         # Vector database storage
└── README.md
```

---

# ⚙️ Technologies Used

* Python
* LangChain
* Chroma Vector Database
* HuggingFace Sentence Transformers
* PyPDF
* Google Colab / Jupyter Notebook

---

# 📦 Installation

Install the required libraries before running the notebook.

```
pip install langchain
pip install langchain-community
pip install langchain-chroma
pip install chromadb
pip install sentence-transformers
pip install pypdf
```

---

# 📑 Step 1: Load the Document

The research paper is loaded using **PyPDFLoader**.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Attention is all you need.pdf")
docs = loader.load()
```

---

# ✂️ Step 2: Split the Document

Large documents are split into smaller chunks.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_splits = text_splitter.split_documents(docs)
```

---

# 🔢 Step 3: Create Embeddings

Embeddings convert text into numerical vector representations.

```python
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
```

---

# 🗄 Step 4: Store Data in Vector Database

The embeddings are stored in **Chroma Vector Store**.

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="Gen_AI_research_collection",
    embedding_function=embedding_model,
    persist_directory="./langchain_chroma_db"
)

vector_store.add_documents(documents=all_splits)
```

---

# 🔍 Step 5: Retrieve Relevant Context

The system retrieves the most relevant document chunks for a query.

```python
def retrieve_context(query: str, k: int = 2):
    retrieved_docs = vector_store.similarity_search(query, k=k)

    docs_content = ""
    for doc in retrieved_docs:
        docs_content += f"Source: {doc.metadata}\n"
        docs_content += f"Content: {doc.page_content}\n\n"

    return docs_content
```

---

# 💡 Example Query

```
What is the main idea behind the Transformer architecture?
```

The system retrieves relevant sections from the research paper and returns the most relevant content.

---

# 🎯 Applications

This system can be used for:

* AI research assistants
* Document question answering systems
* Knowledge retrieval systems
* Research paper analysis
* Legal and technical document search

---

# 📚 Reference

Research Paper Used:

**Attention is All You Need**
Ashish Vaswani et al., 2017

---

# 👩‍💻 Author

**Meghana K R**

B.E Computer Science & Design
Mysore University School of Engineering
