from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from plugins.base import Plugin
from plugins.registry import register
from graph.state import AgentState
from rag.retriver import load_retriever

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = load_retriever("data/faiss")

class RagPlugin(Plugin):
    name = "rag_qa"
    route_key = "RAG_QA"

    def node(self):
        def _node(state: AgentState) -> AgentState:
            docs = retriever.invoke(state.question)
            contexts = [d.page_content for d in docs]
            state.context_docs = contexts

            context_text = "\n\n---\n\n".join(contexts)
            prompt = f"""당신은 문서 근거로만 답해야 합니다.
            문서에서 찾을 수 없으면 '문서에서 근거를 찾지 못했습니다.' 라고 말해야 합니다.

            [근거]
            {context_text}

            [질문]
            {state.question}
            """

            state.answer = llm.invoke([HumanMessage(content=prompt)]).content
            return state
        return _node