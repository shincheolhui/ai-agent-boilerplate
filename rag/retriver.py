from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def load_retriever(index_path: str):
    vs = FAISS.load_local(
        index_path,
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )
    return vs.as_retriever(search_kwargs={"k": 5})
