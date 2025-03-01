import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer



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


# 形参输入prompt，返回数据库中相关的context内容。
# 返回为一个字典，context为文本内容（列表），metadata为文本的元数据（列表）
# 一共会返回10条相关的context内容，其中context和metadata中的数据一一对应。
def get_context(prompt, n_results=10):
    query_text = [prompt]
    query_answer = db.query(
        query_texts=query_text,
        n_results=n_results
    )
    res = {
        "context": query_answer["documents"],
        "metadata": query_answer["metadatas"]
    }
    print("RAG res get!")
    return res


