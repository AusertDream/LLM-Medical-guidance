import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm


# open the pdf directory
folder_path = "F:\\FastIn\\UniversityProduction\\GradutionProject\\RawData\\临床本科教材\\蓝色生死恋（字体优化版）"
docs = os.listdir(folder_path)

for doc in tqdm(docs, desc="Extracting content"):
    bookName = doc[:-3]
    loader = PyPDFLoader(os.path.join(folder_path, doc))
    loaded = loader.load()
    with open(f"./data/pdfs/{bookName}txt", "w", encoding="utf-8") as f:
        for i, item in enumerate(loaded):
            f.write(f"Page {i}: {item.page_content}\n")


print("Done")