from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm

loader = PyPDFLoader("./data/pdfs/1.pdf")
doc = loader.load()

for i, item in tqdm(enumerate(doc), desc="Extracting content"):
    if i>=40 and i<=100 and item.page_content!="":
        print(f"Page {i}: {item.page_content}")