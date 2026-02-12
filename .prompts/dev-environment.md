# Project Setup Guide

## Runtime & Tools

### 1. Language Runtime

Python 3.10 이상의 버전을 사용한다. LLM 생태계와의 최적화된 호환성과 최신 타이핑 기능을 활용하기 위함이다.

### 2. Dependency & Package Management

Rust 기반의 초고속 패키지 관리자인 **uv**를 표준으로 사용한다. 의존성 해결 속도를 비약적으로 향상시키고 `uv.lock`을 통해 실행 환경의 일관성을 엄격히 통제한다.

### 3. Code Quality Control

린팅(Linting)과 포맷팅(Formatting)의 통합 수행을 위하여 **Ruff**를 도입한다. PEP 8 표준을 준수하며 정적 분석을 통해 코드의 잠재적 오류를 사전에 식별한다.

---

## Local Setup Guide

### 1. Repository Clone

원격 저장소의 소스 코드를 로컬 환경으로 복제한다.

```bash
git clone <repository_url>
cd assist
```

### 2. Environment Initialization

uv를 사용하여 가상 환경을 생성하고 필요한 라이브러리를 설치한다.

```bash
# 의존성 설치 및 가상 환경(.venv) 동기화
uv sync
```

### 3. Environment Variable Configuration

제공된 템플릿을 바탕으로 환경 변수 파일을 생성한다.

```bash
cp .env.example .env
# 이후 .env 파일을 열어 필수 API 키를 입력한다.
```

### 4. Execution

서버를 실행하여 정상 작동 여부를 확인한다.

```bash
# FastAPI 서버 실행 (Hot-reload 활성화)
uv run uvicorn src.main:app --reload
```

---

## Configuration (`.env`)

### 1. OpenAI Settings

| 변수 | 설명 |
|------|------|
| `OPENAI_API_KEY` | GPT-4o 모델 호출 및 임베딩 생성을 위한 인증 키 |

### 2. Database Settings

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `CHROMA_DB_PATH` | 벡터 데이터베이스 ChromaDB의 데이터가 저장될 로컬 경로 | `./data/chroma` |

### 3. LangChain / LangGraph Settings *(Optional)*

| 변수 | 설명 |
|------|------|
| `LANGCHAIN_TRACING_V2` | 에이전트의 추론 과정을 모니터링하기 위한 트레이싱 활성화 여부 |
| `LANGCHAIN_API_KEY` | LangSmith 연동을 위한 인증 키 |

### 4. Server Settings

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `HOST` | 서버가 수신 대기할 IP 주소 | `0.0.0.0` |
| `PORT` | 서버 포트 번호 | `8000` |
