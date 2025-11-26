from typing import Dict, List
from plugins.base import Plugin

# 플러그인 이름 -> 플러그인 객체
_REGISTRY: Dict[str, Plugin] = {}


def register(plugin: Plugin) -> None:
    """플러그인을 전역 레지스트리에 등록"""
    _REGISTRY[plugin.name] = plugin


def get(name: str) -> Plugin:
    return _REGISTRY[name]


def all_plugins() -> List[Plugin]:
    return list(_REGISTRY.values())


def route_plugins() -> Dict[str, str]:
    """
    router가 반환하는 route_key -> 실제 노드 이름(plugin.name) 매핑 생성

    예:
      META       -> "meta"
      RAG_QA     -> "rag_qa"
      DOC_SUMMARY-> "doc_summary"
    """
    mapping: Dict[str, str] = {}
    for p in _REGISTRY.values():
        if p.route_key:
            mapping[p.route_key] = p.name
    return mapping
