from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional
from graph.state import AgentState

class Plugin(ABC):
    """모든 플러그인의 최소 인터페이스"""

    name: str                   # 노드 이름
    route_key: Optional[str]    # router가 반환하는 값 (null이면 entry/utility 노드)

    @abstractmethod
    def node(self) -> Callable[[AgentState], AgentState]:
        """LangGraph에 붙는 노드 함수 반환"""
        raise NotImplementedError