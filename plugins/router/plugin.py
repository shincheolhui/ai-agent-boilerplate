from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from plugins.base import Plugin
from plugins.registry import register
from graph.state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

ROUTER_SYSTEM_PROMPT = """당신은 질문을 보고 다음 라우트 중 하나만 고르세요.
- META: 대화 히스토리 요약/회상/내가 전에 뭐라고 했나요?
- RAG_QA: 문서 근거 기반 질의응답
- DOC_SUMMARY: 문서 전체/구조/요약 요청
출력은 META 또는 RAG_QA 또는 DOC_SUMMARY 중 하나의 키워드만 반환하세요.
"""

class RouterPlugin(Plugin):
    name = "router"
    route_key = None

    def node(self):
        def _node(state: AgentState) -> AgentState:
            msg = [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=state.question),
            ]
            route = llm.invoke(msg).content.strip()
            state.route = route     # type: ignore
            return state
        return _node

register(RouterPlugin())