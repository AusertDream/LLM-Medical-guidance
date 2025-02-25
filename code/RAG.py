import os
import sys

from chromadb import EmbeddingFunction, Documents, Embeddings
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import chromadb
import markdown
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import SentenceTransformer
import uuid

"""
the RAG data is from Medical Undergraduate Textbooks extracted from PDFs by PyPDFLoader

Haystack 
RAGflow
"""

data_dir = "./data/books"
model_name = './RAGEmbedding/m3e-base'
embeddings = SentenceTransformer(model_name)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_name,
    device="cuda",
    normalize_embeddings=True
)
db_path = "./chroma_db_1024"  # 设置持久化数据库的路径
client = chromadb.PersistentClient(path=db_path)
db = client.get_or_create_collection(name="AirDatabase", embedding_function=sentence_transformer_ef)


def load_md_file(file):
    loader = TextLoader(file, encoding="utf-8")
    documents = loader.load()
    return documents

def save_one_md_file(document, chunk_size=1024, chunk_overlap=64):
    # 创建拆分器
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # 拆分文档
    documents = text_splitter.split_documents(document)
    page_content = [markdown.markdown(doc.page_content) for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    ids = [str(uuid.uuid4()) for doc in documents]
    db.add(
        documents=page_content,
        ids=ids,
        metadatas=metadatas
    )



def build_chroma_db():
    files = os.listdir(data_dir)
    file_number = len(files)
    for i in tqdm(range(file_number), desc="processing"):
        file = os.path.join(data_dir, files[i])
        documents = load_md_file(file)
        save_one_md_file(documents, chunk_size=1024, chunk_overlap=64)




build_chroma_db()