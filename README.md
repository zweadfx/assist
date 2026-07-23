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

## 평가 (2026-04-24 측정)

규칙 기반 지표 + LLM-as-Judge(gpt-4o)로 기능별 채점했습니다. 4/22 1차 측정 후 **평가 자체를 재설계**(데이터셋 기대 조문 수정, Judge 프롬프트 도메인 분리, retrieval/generation 지표 분리)했기 때문에, 아래는 재설계 후 측정값이며 1차 수치와의 직접 비교(델타)는 하지 않습니다.

| 기능 | 지표 | 값 | 근거 리포트 |
|---|---|---|---|
| **The Whistle** (25케이스) | Rule Hit@3 / MRR | 0.56 / 0.46 | [20260424_091815](docs/eval/eval_report_20260424_091815.md) |
| | Citation Hit Rate | 0.72 | 〃 |
| | Accuracy / Citation / Faithfulness (Judge) | 4.68 / 4.40 / 4.76 (5.0 만점) | 〃 |
| **Gear Advisor** (25케이스) | Hit@3 / MRR (RAG) | 0.92 / 0.87 | [20260424_093601](docs/eval/eval_report_20260424_093601.md) |
| | Accuracy / Data Fidelity (Judge) | 4.36 / 4.76 (5.0 만점) | 〃 |
| **Skill Lab** (5케이스, 2026-04-28) | 장비 준수율 / 시간 준수율 | 1.00 / 1.00 (RAG 베이스라인 0.80 / 1.00) | [20260428_230907](docs/eval/eval_report_20260428_230907.md) |

**평가 여정** — 실패런 포함 전 리포트를 [docs/eval/](docs/eval/)에 그대로 커밋했습니다: 3/18 규칙 기반 1차는 id 매칭 버그로 전 지표 0.00([리포트](docs/eval/eval_report_20260318_151043.md)) → 같은 날 수정 후 Gear Hit@3 0.87([리포트](docs/eval/eval_report_20260318_151753.md)). 4/22 LLM-as-Judge 도입 — 스코어 파싱 실패런 2회 뒤 성공([리포트](docs/eval/eval_report_20260422_164143.md)). 4/23~24 평가 재설계 후 재측정한 것이 위 표입니다. 평가 스크립트와 케이스 데이터셋은 `scripts/eval/`에 있습니다.

## 한계

- **Judge 수치는 LLM이 LLM을 채점한 값입니다.** 사람 검증을 거치지 않았고, 측정 시점 이후 코드 변경분이 반영되지 않았습니다.
- **Whistle의 검색 지표가 가장 약합니다(Rule Hit@3 0.56 · MRR 0.46).** 생성 단계 지표는 높지만, 정답 조문이 상위 검색에 못 든 케이스가 절반 가까이 됩니다.
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

**룰 데이터 출처**: 레포에는 FIBA·NBA 룰북의 **파싱 결과물**(조문 단위 청크, `data/parsed/rules_chunks.json`)만 포함됩니다. 원문 PDF는 저작권 문제로 포함하지 않습니다 — 원문은 공식 사이트(FIBA.basketball / official.nba.com)에서 구할 수 있습니다.

라이선스: MIT
