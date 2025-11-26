from dotenv import load_dotenv
load_dotenv()

# 플러그인 import가 "등록"을 일으키므로 반드시 먼저 import
import plugins.router.plugin
import plugins.meta.plugin
import plugins.rag.plugin
import plugins.summary.plugin

from graph.build_graph import build_app
from graph.state import AgentState

app = build_app()

if __name__ == "__main__":
    # 1) 초기 state 만들기 (pydantic 모델)
    init_state = AgentState(
        question="이 문서 전체 구조 요약해줘",
        chat_history=[]
    )

    # 2) LangGraph에는 dict를 넘겨주는게 안전
    result = app.invoke(init_state.model_dump())

    # 3) 결과도 dict로 돌아오기 때문에 key로 접근
    print(result.get("answer"))