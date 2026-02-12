# Tech Stack Specification

---

## 1. Technology Overview

| Category      | Technology       | Rationale (선택 근거)                                                              |
| ------------- | ---------------- | ---------------------------------------------------------------------------------- |
| Language      | Python 3.10+     | LLM 및 데이터 분석 라이브러리와의 생태계 호환성이 가장 뛰어남.                     |
| Backend       | FastAPI          | 비동기 처리를 통한 고성능 API 제공 및 Pydantic 기반의 자동 문서화 지원.            |
| Orchestration | LangGraph        | 순환(Cycle) 로직과 정교한 상태 관리를 통해 다중 기능 에이전트를 구현함.            |
| Framework     | LangChain        | LangGraph의 기반 프레임워크로서 각종 도구 및 LLM 인터페이스를 제공함.              |
| Model         | OpenAI GPT-4o    | 복잡한 농구 데이터 분석 및 에이전트의 논리적 추론 능력을 극대화함.                 |
| Vector DB     | ChromaDB         | 서버리스 기반의 가벼운 설치와 로컬 환경에서의 신속한 벡터 검색 지원.               |
| Package       | uv               | Rust 기반의 초고속 의존성 해결 및 안정적인 가상 환경 관리 체계 구축.               |
| Code Quality  | Ruff             | Linting 및 Formatting의 통합 수행을 통한 코드 품질 유지 및 생산성 향상.            |

---

## 2. Technical Details & Rationale

### AI Orchestration: LangGraph & Multi-Agent

- **상태 기반 제어:** 공유 상태(State) 객체를 활용하여 사용자의 이전 질의 맥락을 유지하고, 복잡한 추론 과정에서도 데이터 일관성을 보장한다.
- **다중 기능 라우팅:** 사용자의 의도를 분석하여 '농구화 추천', '훈련 루틴 생성', '규칙 질의' 등 적절한 작업 노드(Node)로 유연하게 흐름을 분기한다.
- **자가 수정 로직:** AI가 생성한 답변이 부적절할 경우, 스스로 판단하여 검색 단계로 다시 돌아가는 피드백 루프를 구현하여 답변 품질을 제고한다.

### Backend: FastAPI

- **비동기 최적화:** LangGraph의 비동기 실행 엔진과 완벽히 호환되어 LLM 호출 시 병목 현상을 최소화하고 고성능 API를 제공한다.
- **개발 효율성:** 자동 생성되는 Swagger 문서를 통해 프론트엔드 개발자와의 협업 효율을 극대화한다.

### Infrastructure: uv & Ruff

- **일관된 환경:** `uv.lock` 파일을 통해 모든 개발 환경에서 동일한 라이브러리 버전을 보장하여 버전 충돌을 원천 차단한다.
- **엄격한 품질 관리:** `Ruff`를 도입하여 PEP 8 표준을 준수하고 정적 분석을 통해 잠재적 런타임 오류를 사전에 식별한다.

---

## 3. System Architecture (Graph-based)

본 프로젝트는 다음과 같은 LangGraph 기반 워크플로우를 기반으로 동작한다.

```
[User Request]
      │
      ▼
┌─────────────┐
│   Router    │  ← 사용자 의도 분류
│   (Input)   │
└──────┬──────┘
       │
       ├──────────────────┬──────────────────┐
       ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Recommend    │  │  Coaching    │  │   Rules      │
│    Node      │  │    Node      │  │    Node      │
│ (ChromaDB)   │  │ (훈련 설계)  │  │ (규칙 질의)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          ▼
                  ┌──────────────┐
                  │  Refinement  │  ← 검증 및 피드백 루프
                  │    Node      │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │    Final     │  ← API 응답 포맷 반환
                  │   Response   │
                  └──────────────┘
```

### Flow Description

1. **Input & Routing:** 사용자의 요청이 들어오면 Router 노드가 의도를 분류한다.

2. **Task Execution:**
   - **Recommendation Node:** ChromaDB에서 농구화 데이터를 검색하여 추천 로직을 수행한다.
   - **Coaching Node:** 사용자의 신체 조건에 최적화된 훈련 루틴을 설계한다.

3. **Refinement:** 필요 시 생성된 결과물을 검증 노드에서 검토하고, 미흡할 경우 다시 검색 또는 생성 노드로 회귀(Loop)한다.

4. **Final Response:** 최종 완성된 답변을 사전에 정의된 API 응답 포맷에 맞춰 전송한다.
