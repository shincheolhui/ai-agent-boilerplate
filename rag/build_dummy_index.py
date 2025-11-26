from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

texts = [
    "이 문서는 ai-agent-boilerplate 테스트용 더미 문서입니다.",
    "이 문서는 RAG와 요약 플러그인이 제대로 연결되는짖 확인하기 위한 예제입니다.",
]

if __name__ == "__main__":
    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_texts(texts, embeddings)
    vs.save_local("data/faiss")
    print("FAISS 인덱스 생성 완료: data/faiss")