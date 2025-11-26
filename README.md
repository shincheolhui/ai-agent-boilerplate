# 📌 ai-agent-boilerplate

**LangChain 1.x + LangGraph 1.x + Plugin Architecture 기반 AI Agent 템플릿**

---

## 🚀 Overview

**ai-agent-boilerplate**는 LangChain & LangGraph 1.x 기반으로
**확장 가능한 플러그인 구조**를 갖춘 실무형 AI Agent 템플릿입니다.

즉,

* “라우팅 → 여러 플러그인 → 그래프 기반 제어”
* “RAG, 문서요약, 메타(META) 대화 이해 등 독립 기능”
  을 각각 **모듈 형태로 조립**할 수 있는 구조입니다.

이 보일러플레이트는 아래 워크플로우를 **기본 제공**합니다:

1. **Router Plugin** → 사용자 질문을 3가지 라우트로 분류
2. **META Plugin** → 대화 히스토리 기반 응답
3. **RAG_QA Plugin** → FAISS 기반 문서 검색 + 근거 기반 QA
4. **DOC_SUMMARY Plugin** → 문서 요약/구조 분석
5. **LangGraph로 플러그인 자동 조립 → 상태(State) 기반 에이전트 실행**

또한 모든 기능은 **플러그인 단위로 분리**되어 있어
필요한 기능만 가져와 다른 프로젝트에 독립적으로 사용할 수 있습니다.

---

## 🧱 Features

### ✔ 플러그인 기반 아키텍처

각 기능은 `plugins/<기능명>/plugin.py` 에 완전히 독립적으로 존재합니다.

추가하고 싶은 기능이 있다면 폴더 하나만 만들고 플러그인 등록하면 됩니다.

### ✔ LangGraph 기반 상태(State) 제어

LangChain이 아닌 **LangGraph**가 전체 에이전트의 흐름을 제어합니다.
즉, LLM 호출과 RAG를 “그래프 노드”로 구성하여 **안정적인 파이프라인**을 구성합니다.

### ✔ RAG(검색 기반 질의응답) 기본 제공

* FAISS vector store 사용
* Embedding: `OpenAIEmbeddings`
* 검색 결과를 기반으로 한 근거 기반 질의응답

### ✔ 문서 요약 에이전트 기본 제공

Dummy 인덱스를 기반으로 문서 전체 구조/요약을 생성합니다.

### ✔ META/HISTORY 질문 처리

“내가 전에 뭐라 했지?”, “요약해줘” 같은
히스토리 기반 요청을 처리하는 META 플러그인 내장.

---

## 📁 Project Structure

```
ai-agent-boilerplate/
  app/
    __init__.py
    main.py                 # 실행 엔트리포인트
  graph/
    __init__.py
    state.py                # State 모델 정의
    build_graph.py          # 플러그인 기반 LangGraph 조립기
  plugins/
    __init__.py
    base.py                 # 플러그인 인터페이스
    registry.py             # 플러그인 등록/조회/라우트맵
    router/
      plugin.py             # Router 노드
    meta/
      plugin.py             # META 노드
    rag/
      plugin.py             # RAG QA 노드
    summary/
      plugin.py             # DOC_SUMMARY 노드
  rag/
    __init__.py
    build_dummy_index.py    # FAISS 더미 인덱스 생성(단일 실행 테스트용)
    retriever.py            # 공용 retriever 로딩
  memory/
    __init__.py
    chat_history.py         # 히스토리 저장소(확장 가능, 현재 미구현 상태)
  data/
    faiss/                  # FAISS 인덱스 저장 위치
  .env                      # OpenAI API KEY
  requirements.txt          # 버전 명시 없을 시 호환성 문제 발생, 반드시 호환성 체크해서 안정적인 최신 버전으로 명시
  README.md
```

---

## 🧭 Graph Flow (LangGraph 흐름도)

이 프로젝트의 그래프 흐름은 **하나의 Router 노드**에서
질문을 3개의 플러그인 노드로 분기하는 구조입니다.

```mermaid
flowchart TD
    U[사용자 입력 + 초기 State] --> R[router<br/>Router Plugin]

    R -->|META| M[META Plugin<br/>(대화 히스토리 응답)]
    R -->|RAG_QA| Q[RAG_QA Plugin<br/>(문서 근거 기반 QA)]
    R -->|DOC_SUMMARY| S[DOC_SUMMARY Plugin<br/>(문서 요약)]

    M --> E[END]
    Q --> E[END]
    S --> E[END]
```

* **Router Plugin**

  * LLM을 사용해 질문을 보고 `META / RAG_QA / DOC_SUMMARY` 중 하나를 결정
  * 결정된 값은 `state.route` 에 기록
* **각 Plugin**

  * `state`를 입력받아 처리 후 `state.answer` 에 최종 답변을 기록
* **LangGraph**

  * `state.route` 값에 따라 해당 플러그인 노드로 이동
  * 플러그인 노드 실행 후 `END` 로 종료

플러그인을 추가하면, 이 그래프에 **새로운 노드를 꽂아넣는 느낌**으로 확장할 수 있습니다.

---

## 🛠 Installation

### 1) 프로젝트 클론 & 가상환경 생성(Python 3.11 권장)

```powershell
git clone <repo-url>
cd ai-agent-boilerplate

python -m venv .venv
.\.venv\Scripts\activate
```

### 2) requirements 설치

```powershell
pip install -r requirements.txt
```

### 3) OpenAI API Key 설정

환경변수에 직접 설정:

```powershell
$env:OPENAI_API_KEY="sk-xxxx"
```

또는 `.env` 파일 사용:

```
OPENAI_API_KEY=sk-xxxx
```

---

## 🔍 Build Dummy FAISS Index (테스트용)

```powershell
python -m rag.build_dummy_index
```

성공 메시지:

```
✅ FAISS 인덱스 생성 완료: data/faiss
```

---

## ▶ Run AI Agent

```powershell
python -m app.main
```

예시 출력:

```
### 핵심 요약
이 문서는 ai-agent-boilerplate의 테스트용 더미 문서로...
```

---

## 🧩 Plugin Architecture (확장 방법)

### 플러그인 추가 방법

1. 새 폴더 생성

   ```
   plugins/my_feature/
   ```

2. 기본 구조 작성

   ```python
   from plugins.base import Plugin
   from plugins.registry import register

   class MyPlugin(Plugin):
       name = "my_feature"
       route_key = "MY_ROUTE"

       def node(self):
           def _node(state):
               # do something
               state.answer = "내 기능이 실행됨"
               return state
           return _node

   register(MyPlugin())
   ```

3. Router에 라우팅 규칙만 추가하면 끝
   LangGraph는 자동으로 이 노드를 그래프에 연결합니다.

이 방식은 다음과 같은 기능을 손쉽게 플러그인화할 수 있습니다:

* Document Upload / Indexing 플러그인
* Multi-Tool Agent 플러그인
* ReAct 기반 플러그인
* 검색 소스 변환 플러그인(예: Pinecone, Chroma로 변경)
* API 호출형 WorkFlow 플러그인 등등

---

## 🧪 Troubleshooting

### 1. `No module named 'plugins'`

루트에서 실행해야 합니다.

```powershell
python -m app.main
```

또한 모든 폴더에 `__init__.py`가 있어야 합니다.

### 2. `The api_key client option must be set`

`OPENAI_API_KEY` 를 설정해 주세요.

### 3. FAISS 인덱스가 없다

`rag/build_dummy_index.py`를 실행해야 합니다.

### 4. LangGraph 실행 후 return 값이 dict

LangGraph는 기본적으로 dict를 반환합니다.

```python
result = app.invoke(state_dict)
print(result["answer"])
```

---
