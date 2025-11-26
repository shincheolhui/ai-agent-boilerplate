from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from plugins.base import Plugin
from plugins.registry import register
from graph.state import AgentState
from rag.retriver import load_retriever

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = load_retriever("data/faiss")

class SummaryPlugin(Plugin):
    name = "doc_summary"
    route_key = "DOC_SUMMARY"

    def node(self):
        def _node(state: AgentState) -> AgentState:
            docs = retriever.invoke(
                "이 문서의 전체 내용을 요약할 수 있는 핵심 부분을 찾아주세요."
            )
            full_text = "\n\n".join(d.page_content for d in docs)

            prompt = f"""아래 문서를 한국어로 구조적으로 요약하세요.
            - 5~10줄 핵심 요약
            - 목차/구조 추정
            - 중요한 용어 정의

            [문서]
            {full_text}
            """

            state.answer = llm.invoke([HumanMessage(content=prompt)]).content
            return state
        return _node

register(SummaryPlugin())