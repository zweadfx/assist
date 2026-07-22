# Assist — 농구인을 위한 AI 어시스턴트

훈련 루틴 생성, 주간 플랜, 농구화 추천, 룰 판정 — 네 기능을 각각 전용 LangGraph 에이전트로 구현한 FastAPI 백엔드입니다.

**Live Demo: https://assist-frontend-plum.vercel.app**
(무료 서버라 **첫 요청은 서버 기동으로 약 1분** 걸릴 수 있습니다. 한 번 뜨면 이후엔 즉시 응답합니다.)

## 무엇을 왜 만들었나

혼자 운동하는 농구인은 전문 코칭에 접근하기 어렵습니다. "슛이 안 맞는데 뭘 연습해야 하지", "이 상황이 파울인가" 같은 질문에 근거 있는 답을 주는 도구를 목표로 했습니다.

| 기능 | 에이전트 파이프라인 | 하는 일 |
|---|---|---|
| **AI Skill Lab** | `diagnose → generate` | 한 가지 기술을 3~5개 마이크로 스텝으로 분해한 훈련 카드 생성. 요청한 연습 시간에 맞춰 스텝 시간을 배분 |
| **Weekly Routine** | `diagnose → plan_week → generate` | 1~7일 주간 훈련 플랜. LLM이 요일별 포커스를 배분(실패 시 라운드로빈 폴백) |
| **Gear Advisor** | `analyze → retrieve → generate` | 감각 선호(쿠션감·접지력 등)와 플레이 스타일 기반 농구화 추천 — 신발 59켤레·선수 프로필 20건 벡터 검색, 예산 필터 |
| **The Whistle** | `parse → extract_keywords → retrieve → generate` | 경기 상황을 룰 조문으로 판정 — FIBA·NBA 룰북을 조문 단위로 청킹해 검색, 용어집 22개 병행 |

모든 응답은 Pydantic 스키마로 강제되고, 파싱 실패 시 2단계 재시도를 거칩니다. JWT 인증(access+refresh)과 플랜 저장(SQLite/PostgreSQL)을 지원합니다.

## 스택과 구조

Python · FastAPI · LangGraph · ChromaDB(벡터) · SQLAlchemy(SQLite/PostgreSQL) · JWT(python-jose)+bcrypt · uv/Ruff — 프론트엔드는 별도 레포(Next.js 15, Vercel 배포), 백엔드는 Render.

```
[요청] → [FastAPI] → [기능별 LangGraph StateGraph]
                        ① 입력 파싱·정제 (프롬프트 인젝션 패턴 차단)
                        ② RAG 검색 (ChromaDB — 드릴·신발·선수·룰·용어집)
                        ③ 스키마 강제 생성 (Pydantic + 재시도 파싱)
```

The Whistle에는 `extract_keywords` 노드를 따로 뒀습니다 — 긴 상황 서술문과 짧은 룰 조문의 임베딩 공간이 어긋나 검색이 빗나가는 문제를, 상황문을 3~5개 위반 유형 키워드로 압축한 뒤 검색하는 방식으로 줄였습니다.

## 평가 (2026-04-22 측정)

LLM-as-Judge(gpt-4o) 파이프라인으로 기능별 25케이스를 채점했습니다. 아래는 당시 측정값이며, 이후 코드가 변경되어 현재 코드 기준으로 재측정된 값이 아닙니다.

| 기능 | Accuracy | Logical Consistency | 기타 |
|---|---|---|---|
| Gear Advisor (25케이스) | 3.56 / 5.0 | 4.20 / 5.0 | Data Fidelity 3.04 |
| The Whistle (25케이스) | 3.40 / 5.0 | 3.52 / 5.0 | Citation Appropriateness 2.84 |

평가 스크립트와 케이스 데이터셋은 `scripts/eval/`에 있습니다.

## 한계

- **평가 수치는 LLM이 LLM을 채점한 값입니다.** 사람 검증을 거치지 않았고, 측정 시점 이후 코드 변경분이 반영되지 않았습니다.
- **Whistle의 인용 적합성(2.84/5)이 가장 약합니다.** 판정은 그럴듯한데 근거 조문이 어긋나는 경우가 남아 있습니다.
- **일부 통합 테스트가 로컬 벡터 DB 적재 상태에 의존합니다.** fresh clone에서 전부 통과하지 않습니다.
- **무료 호스팅 콜드스타트** — 첫 요청 약 1분.

## 배운 것

- 기능마다 상태 머신(LangGraph)을 분리하면 프롬프트·검색·생성을 기능 단위로 독립 개선할 수 있습니다.
- LLM 출력은 스키마로 강제하고 재시도를 설계해야 서비스가 됩니다 — 프롬프트만으로는 형식이 지켜지지 않았습니다.
- 검색 품질은 임베딩 대상의 형태를 맞추는 것에서 갈렸습니다(상황문→키워드 압축).
- 평가를 LLM-as-Judge에만 맡기면 수치의 근거를 설명하기 어렵다는 한계를 남겼습니다.

> 이 반성이 다음 프로젝트의 출발점이 됐습니다 — **[jd-gap-analyzer](https://github.com/zweadfx/jd-gap-analyzer)**: 판정 기준 사전 커밋, 사람 판정, 원문 대조 검증 등 측정·검증을 중심에 둔 후속 프로젝트.

## 실행

```bash
git clone https://github.com/zweadfx/assist.git
cd assist
uv sync
cp .env.example .env   # OPENAI_API_KEY 등 입력
uv run uvicorn src.main:app --reload
```

Python 3.10+ · API 문서: 서버 기동 후 `/docs`

라이선스: MIT
