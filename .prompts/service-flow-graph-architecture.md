# Service Flow & Graph Architecture

---

## 1. 개요

### 1.1 정의

**Service Flow**는 사용자 인터페이스와 백엔드 로직 간의 데이터 흐름을 의미하며, **Graph Architecture**는 `LangGraph`를 활용하여 설계된 에이전트의 노드(Node)와 에지(Edge) 간의 상태 전이 구조를 정의한다.

### 1.2 목적

본 문서는 'Assist' 서비스의 세 가지 핵심 기능(AI Skill Lab, Gear Advisor, The Whistle)이 단일 백엔드 시스템 내에서 어떻게 유기적으로 연결되고 작동하는지를 명확히 기술하여, 개발자가 에이전트의 논리적 흐름을 이해하고 확장 가능한 구조를 구축할 수 있도록 지원한다.

### 1.3 핵심 원칙

- **상태 기반 설계**: 모든 에이전트는 공유 상태 객체(Agent State)를 통해 데이터를 전달하고 업데이트한다.
- **모듈화**: 각 기능은 독립적인 노드로 구현되어 있어 재사용성과 유지보수성이 높다.
- **동적 라우팅**: 사용자의 질문 의도를 실시간으로 분석하여 적절한 에이전트로 요청을 전달한다.

---

## 2. 그래프 구조

에이전트의 사고 과정은 다음과 같은 구성 요소를 통해 정교하게 제어된다.

### 2.1 상태 객체 (Agent State)

그래프 내의 모든 노드가 공유하며 업데이트하는 데이터 구조이다.

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]  # 사용자와 에이전트 간의 대화 이력 (대화 맥락 유지)
    intent: str                  # 사용자의 질문 의도 (추천/훈련/규칙)
    context: List[Document]      # Vector DB에서 검색된 문서 리스트 (RAG 데이터)
    user_info: dict              # 사용자의 신체 조건 및 포지션 정보
    routing_decision: str        # Router가 결정한 에이전트 경로
    final_response: str          # 최종 생성된 응답 텍스트
```

### 2.2 주요 노드 (Nodes)

각 노드는 특정 작업을 수행하며 상태 객체를 업데이트한다.

| 노드 명칭 | 역할 | 입력 | 출력 |
|-----------|------|------|------|
| **Router Node** | LLM이 질문을 분석하여 3가지 핵심 기능(농구화, 훈련, 규칙) 중 적절한 경로를 선택한다. | `messages`, `user_info` | `routing_decision`, `intent` |
| **Shoe Retrieval Node** | `ChromaDB`에서 농구화 스펙 데이터를 검색한다. | `intent`, `user_info` | `context` |
| **Coach Node** | 사용자의 프로필을 기반으로 맞춤형 훈련 루틴을 생성한다. | `intent`, `user_info` | `context` |
| **Rule Expert Node** | 농구 규칙 전문을 바탕으로 정확한 근거를 추출한다. | `intent`, `messages` | `context` |
| **Response Node** | 모든 정보를 취합하여 최종 답변을 API Response Convention 형식으로 가공한다. | `context`, `messages` | `final_response` |

### 2.3 에지 및 조건부 라우팅 (Edges & Conditional Routing)

노드 간의 전환은 상태 객체의 값을 기반으로 동적으로 결정된다.

```python
# 조건부 라우팅 예시
def should_continue(state: AgentState) -> str:
    """
    Router의 결정에 따라 다음 노드를 선택한다.
    """
    if state["routing_decision"] == "shoe_recommendation":
        return "shoe_retrieval"
    elif state["routing_decision"] == "training_plan":
        return "coach"
    elif state["routing_decision"] == "rule_query":
        return "rule_expert"
    else:
        return "response"
```

---

## 3. 워크플로우 상세

사용자의 질의부터 최종 응답까지의 전체 흐름을 단계별로 설명한다.

### 3.1 Input (사용자 입력)

사용자가 프론트엔드(React)를 통해 질의를 입력한다.

```json
{
  "user_id": "string",
  "message": "오프볼 무브가 좋은 선수에게 맞는 농구화는?",
  "user_profile": {
    "height_cm": 178,
    "weight_kg": 75,
    "position": "guard",
    "skill_level": "intermediate"
  }
}
```

### 3.2 Routing (경로 결정)

`FastAPI` 서버를 거쳐 `LangGraph`의 진입점(Entry Point)인 **Router Node**가 활성화된다.

**Router의 동작 원리**:
1. 사용자의 질문 텍스트를 LLM에 전달한다.
2. LLM이 질문의 의도를 분석하여 다음 중 하나로 분류한다:
   - `shoe_recommendation`: 농구화 추천
   - `training_plan`: 훈련 계획 생성
   - `rule_query`: 규칙 질의
3. 분류 결과를 `routing_decision`에 저장한다.

### 3.3 Task Execution (작업 수행)

Router의 결정에 따라 해당 전문 노드가 실행된다.

#### 3.3.1 농구화 추천 (Shoe Retrieval Node)

질문에 포함된 키워드와 사용자 신체 정보를 매칭하여 검색을 수행한다.

**처리 과정**:
1. 사용자 프로필에서 `position`, `skill_level` 추출
2. 질문에서 감각 키워드(예: "오프볼 무브") 추출
3. `ChromaDB`에서 벡터 유사도 검색 수행
4. 검색 결과를 `context`에 저장

#### 3.3.2 훈련 가이드 (Coach Node)

내부 지식과 사용자 데이터를 결합하여 루틴을 설계한다.

**처리 과정**:
1. 사용자의 `skill_level`과 `available_time` 분석
2. `drills.json`에서 적합한 훈련 항목 검색
3. 웜업 → 메인 → 쿨다운 순서로 루틴 구성
4. 결과를 `context`에 저장

#### 3.3.3 규칙 답변 (Rule Expert Node)

규칙 전문 데이터를 참조하여 사실 기반 답변을 준비한다.

**처리 과정**:
1. 질문에서 규칙 관련 키워드 추출(예: "트래블링")
2. `glossary.json`에서 용어 정의 검색
3. `rules.pdf`의 벡터 데이터에서 관련 조항 검색
4. 검색된 규정과 용어를 `context`에 저장

### 3.4 Verification (검증 루프)

답변의 정확도가 낮거나 추가 정보가 필요한 경우, 이전 단계로 회귀하여 로직을 재수행한다.

**검증 조건**:
- `context`가 비어있는 경우: 검색 조건 완화 후 재검색
- 사용자 정보가 불충분한 경우: 추가 정보 요청 메시지 생성
- 판정 근거가 불명확한 경우: 다른 규정 조항 탐색

### 3.5 Finalize (최종 응답)

**Response Node**가 모든 정보를 취합하여 최종 답변을 생성한다.

**처리 과정**:
1. `context`에 저장된 데이터를 LLM 프롬프트에 삽입
2. `GPT-4o`가 사용자 친화적인 답변 생성
3. API Response Convention에 맞춰 JSON 포맷팅
4. `final_response`에 저장 후 프론트엔드로 반환

**최종 응답 예시**:
```json
{
  "response_type": "shoe_recommendation",
  "data": {
    "shoes": [
      {
        "model_name": "Under Armour Curry 10",
        "reason": "오프볼 무브가 잦은 가드 포지션에 최적화된 접지력과 민첩성을 제공합니다."
      }
    ]
  },
  "metadata": {
    "search_duration_ms": 450,
    "sources": ["shoes.json", "players.json"]
  }
}
```

---

## 4. 시각적 그래프 표현

```
┌─────────────┐
│   START     │
│  (User Input)│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Router Node │
│ (Intent     │
│  Detection) │
└──────┬──────┘
       │
       ├─────────────┬─────────────┐
       │             │             │
       ▼             ▼             ▼
┌─────────┐   ┌──────────┐   ┌─────────┐
│  Shoe   │   │  Coach   │   │  Rule   │
│Retrieval│   │   Node   │   │ Expert  │
│  Node   │   │          │   │  Node   │
└────┬────┘   └────┬─────┘   └────┬────┘
     │             │              │
     └─────────────┼──────────────┘
                   │
                   ▼
            ┌─────────────┐
            │  Response   │
            │    Node     │
            └──────┬──────┘
                   │
                   ▼
            ┌─────────────┐
            │     END     │
            │(API Response)│
            └─────────────┘
```

---

## 5. 기술적 구성 요소

| 구성 요소 | 기술 스택 | 역할 |
|-----------|-----------|------|
| Graph Engine | `LangGraph` | 에이전트의 노드 및 상태 전이를 관리한다. |
| LLM | `GPT-4o` | 자연어 이해 및 응답 생성을 수행한다. |
| Vector DB | `ChromaDB` | RAG를 위한 임베딩 데이터를 저장하고 검색한다. |
| Backend Framework | `FastAPI` | 비동기 API 엔드포인트를 제공한다. |
| Frontend | `React` | 사용자 인터페이스를 구현한다. |

---

## 6. 확장 가능성

### 6.1 신규 기능 추가

새로운 기능을 추가할 때는 다음 절차를 따른다:
1. 새로운 노드 정의 (예: `NutritionNode`)
2. `Router Node`의 라우팅 로직에 분기 추가
3. 필요한 데이터 소스 연결 (예: `nutrition.json`)
4. 테스트 케이스 작성 및 검증

### 6.2 멀티모달 확장

향후 이미지 또는 동영상 분석 기능을 추가할 경우, 다음과 같이 확장할 수 있다:
- **Video Analysis Node**: 사용자가 업로드한 슛폼 영상을 분석하여 개선점 제시
- **Image Recognition Node**: 농구화 사진을 업로드하면 모델명과 스펙 자동 인식

---

## 7. 성능 최적화 전략

### 7.1 캐싱 (Caching)

자주 검색되는 데이터(예: 인기 농구화 모델, 기본 규칙)는 Redis를 활용하여 캐싱함으로써 응답 속도를 향상시킨다.

### 7.2 병렬 처리 (Parallel Execution)

여러 데이터 소스를 동시에 검색할 경우, `asyncio`를 활용하여 병렬 처리를 수행한다.

### 7.3 로드 밸런싱 (Load Balancing)

트래픽이 증가할 경우, 여러 FastAPI 인스턴스를 운영하고 Nginx를 통해 로드 밸런싱을 적용한다.

---
