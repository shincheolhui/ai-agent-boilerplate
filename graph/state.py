from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel

Route = Literal["META", "RAG_QA", "DOC_SUMMARY"]

class AgentState(BaseModel):
    question: str
    route: Optional[Route] = None

    chat_history: List[Dict[str, str]] = []
    context_docs: List[str] = []
    doc_id: Optional[str] = None

    answer: Optional[str] = None
    extra: Dict[str, Any] = {}
    