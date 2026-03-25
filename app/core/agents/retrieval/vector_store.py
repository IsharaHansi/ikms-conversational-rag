import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "ikms-rag")
        
        # Create index if it doesn't exist
        if self.index_name not in pc.list_indexes().names():
            print(f"Creating new Pinecone index: {self.index_name}")
            pc.create_index(
                name=self.index_name,
                dimension=384,                    # all-MiniLM-L6-v2 uses 384 dimensions
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        # Connect to vector store
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        print(f"✅ Connected to Pinecone index: {self.index_name}")

    def load_pdf(self, pdf_path: str):
        """Load PDF, split, and upload to Pinecone"""
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        # Add to Pinecone
        self.vectorstore.add_documents(chunks)
        
        print(f"✅ Successfully indexed {len(chunks)} chunks from {pdf_path}")
        return len(chunks)

    def retrieve(self, query: str, k: int = 5):
        """Retrieve relevant chunks from Pinecone"""
        if not self.vectorstore:
            return "No documents indexed yet. Please upload a PDF first.", []
        
        docs = self.vectorstore.similarity_search(query, k=k)
        
        context = "\n\n".join([
            f"Chunk from page {doc.metadata.get('page', 'unknown')}:\n{doc.page_content}" 
            for doc in docs
        ])
        
        return context, docs