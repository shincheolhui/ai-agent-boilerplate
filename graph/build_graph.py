from langgraph.graph import StateGraph, END
from graph.state import AgentState
from plugins.registry import all_plugins, route_plugins, get

def build_app():
    graph = StateGraph(AgentState)

    # 1) 모든 플러그인 노드 등록
    for p in all_plugins():
        graph.add_node(p.name, p.node())

    # 2) entry는 router로 고정
    graph.set_entry_point("router")

    # 3) router -> route_key별 노드로 조건부 분기
    def route_selector(state: AgentState) -> str:
        return state.route

    graph.add_conditional_edges(
        "router",
        route_selector,
        route_plugins(), # {META: "meta", RAG_QA: "rag_qa", DOC_SUMMARY: "doc_summary"} 자동 생성
    )

    # 4) 모든 노드는 END로 종료
    for route_key, node_name in route_plugins().items():
        graph.add_edge(node_name, END)

    return graph.compile()
