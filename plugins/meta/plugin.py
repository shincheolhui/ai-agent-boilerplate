from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from plugins.base import Plugin
from plugins.registry import register
from graph.state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class MetaPlugin(Plugin):
    name = "meta"
    route_key = "META"

    def node(self):
        def _node(state: AgentState) -> AgentState:
            history_text = "\n".join(
                f"{h['role']}: {h['content']}" for h in state.chat_history[-30:]
            )
            prompt = f"""아래는 대화 히스토리입니다. 사용자의 질문에 답하시오.

            [대화 히스토리]
            {history_text}

            [사용자 질문]
            {state.question}
            """

            state.answer = llm.invoke([HumanMessage(content=prompt)]).content
            return state
        return _node

register(MetaPlugin())