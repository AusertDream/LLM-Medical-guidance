import os
import sys

from tqdm import tqdm
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceBgeEmbeddings
"""
the RAG data is from Medical Undergraduate Textbooks extracted from PDFs by PyPDFLoader
"""


loader = TextLoader("data/books/total.txt", encoding="utf-8")
documents = loader.load()

# 创建拆分器
text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=0)
# 拆分文档
documents = text_splitter.split_documents(documents)

model_name = './model/RAGEmbedding/m3e-base'
model_kwargs = {'device': 'cuda:0'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为文本生成向量表示用于文本检索"
)
# load data to Chroma db
db = Chroma.from_documents(documents, embedding)
res = db.similarity_search("浅层结构")
print(res)

