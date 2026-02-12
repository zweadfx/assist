# Development Convention

---

## Git Convention

### Message Format

커밋 메시지는 기본적으로 다음과 같은 구조를 따르며,
해당 작업과 관련된 이슈 번호를 본문 또는 제목 끝에 반드시 포함한다.

```
[Type] Subject (#IssueNumber)
```

### Commit Types

| Type       | Description                                            |
| ---------- | ------------------------------------------------------ |
| `feat`     | 새로운 기능 추가 (A new feature)                       |
| `fix`      | 버그 수정 (A bug fix)                                  |
| `docs`     | 문서 수정 (Documentation only changes)                 |
| `style`    | 코드 포맷팅, 세미콜론 누락 등 (Formatting, no code change) |
| `refactor` | 코드 리팩토링 (Refactoring production code)            |
| `chore`    | 빌드 업무, 패키지 설정 등 (Updates to build tasks, etc) |

### Writing Rules

- `Subject`는 영문 소문자를 사용하며 50자 이내로 작성한다.
- 마침표(`.`)를 끝에 붙이지 않는다.
- 과거형이 아닌 **명령형** 동사로 시작한다.

### Usage Examples

**Good Examples**

```
feat: implement RAG-based shoe recommendation (#1)
fix: resolve api timeout error in shoe search (#12)
docs: update tech stack section in readme (#3)
refactor: optimize vector database search logic (#7)
chore: add ruff and uv to pyproject.toml (#2)
```

**Bad Examples**

```
fix: fixed a bug                → 과거형 사용 및 구체성 부족
FEAT: Add New Feature.          → 대문자 사용 및 마침표 포함
작업 완료                        → 한글 사용 및 타입 미지정
```

---

## Linter & Formatter Convention

코드 스타일의 자동화와 품질 관리를 위해 **Ruff**를 사용한다.

### Rules

- **Tool:** Ruff (Linting & Formatting 통합 도구)
- **Standard:** PEP 8 규칙을 준수하며, 한 줄 최대 길이는 **88자**(Black standard)로 설정한다.

### Naming Conventions

| Category              | Style              | Example              |
| --------------------- | ------------------ | -------------------- |
| Variables & Functions | `snake_case`       | `calculate_angle`    |
| Classes               | `PascalCase`       | `ShoeRecommender`    |
| Constants             | `UPPER_SNAKE_CASE` | `MAX_RETRY`          |

### Usage Examples

**Variables & Functions (`snake_case`):**

```python
user_height = 185

def get_shoe_recommendation(position, style):
    pass
```

**Classes (`PascalCase`):**

```python
class ShootingAnalyzer:
    pass
```

**Constants (`UPPER_SNAKE_CASE`):**

```python
MAX_RETRIES = 3
OPENAI_MODEL_NAME = "gpt-4o"
```

---

## Git Workflow

### Branch Strategy

모든 새로운 작업은 별도의 브랜치에서 진행하며, 완료 후 PR을 통해 `main`에 병합한다.

| Prefix      | Usage        | Example                     |
| ----------- | ------------ | --------------------------- |
| `feat/`     | 신규 기능 개발 | `feat/shoe-recommendation`  |
| `fix/`      | 오류 수정      | `fix/vector-db-query`       |
| `docs/`     | 문서 작성      | `docs/readme-update`        |
| `refactor/` | 코드 개선      | `refactor/api-structure`    |

### Issue & PR Templates

#### Feature Template

```markdown
## 어떤 기능인가요?

> 추가하려는 기능에 대해 간결하게 설명해주세요

## 작업 상세 내용

- [ ] TODO
- [ ] TODO
- [ ] TODO

## 참고할만한 자료(선택)
```

#### Bug Report Template

```markdown
## 어떤 버그인가요?

> 어떤 버그인지 간결하게 설명해주세요

## 어떤 상황에서 발생한 버그인가요?

> (가능하면) Given-When-Then 형식으로 서술해주세요

## 예상 결과

> 예상했던 정상적인 결과가 어떤 것이었는지 설명해주세요

## 참고할만한 자료(선택)
```

#### Pull Request Template

```markdown
## 어떤 변경사항인가요?

> 이번 PR을 통해 수정하거나 추가된 주요 내용을 간결하게 설명해주세요.

## 작업 상세 내용

- [ ] TODO
- [ ] TODO
- [ ] TODO

## 체크리스트

> 코드 리뷰를 요청하기 전, 아래 항목들을 확인해주세요.

- [ ] self-test를 수행하였는가?
- [ ] 관련 문서나 주석을 업데이트하였는가?
- [ ] 설정한 코딩 컨벤션을 준수하였는가?

## 관련 이슈

> 해당 PR과 관련된 이슈 번호를 기재해주세요. (예: #123)

- 관련 이슈: 

## 리뷰 포인트

> 리뷰어가 중점적으로 확인해주었으면 하는 부분이나, 논의가 필요한 내용을 작성해주세요.

## 참고사항 및 스크린샷(선택)

> 시각적인 확인이 필요하거나 참고할만한 자료가 있다면 첨부해주세요.
```

### Step-by-Step Scenario

새로운 기능을 개발할 때의 표준 흐름입니다.

1. **Issue 생성:** `#1: Implement shoe recommendation API`

2. **Branch 생성:**

   ```bash
   git checkout -b feat/shoe-recommendation
   ```

3. **코드 작업 및 Commit:**

   ```bash
   git add .
   git commit -m "feat: add initial shoe recommendation api"
   ```

4. **Remote Push:**

   ```bash
   git push origin feat/shoe-recommendation
   ```

5. **Pull Request 작성:** `main` 브랜치로 병합 요청 및 이슈 번호(`#1`) 연결.

---

## Package Management (uv)

프로젝트 의존성 관리 및 가상환경 구축에는 **uv**를 사용한다.

### Core Commands

| Command                       | Description       |
| ----------------------------- | ----------------- |
| `uv init`                     | Initialize        |
| `uv add <package_name>`       | Add Libs          |
| `uv add --dev <package_name>` | Add Dev Libs      |
| `uv sync`                     | Sync              |
| `uv run <script_name>`        | Run               |

### Dependency Rules

의존성 버전의 동일성을 보장하기 위해 `uv.lock` 파일은 반드시 커밋한다.

### Usage Examples

```bash
# 신규 라이브러리 설치 시
uv add langchain-community

# 개발 도구(테스트 등) 추가 시
uv add --dev pytest

# 프로젝트 실행 시 (가상환경 수동 활성화 불필요)
uv run uvicorn main:app --reload

# 최신 상태로 동기화 시
uv sync
```

---

## Directory Structure Convention

### 1. 프로젝트 트리 (Project Tree)

```
assist/
├── .venv/                       # uv를 통해 생성된 가상 환경 (Virtual Environment)
├── data/                        # RAG 엔진을 위한 데이터 저장소
│   ├── raw/                     # 원천 데이터 파일 (shoes.json, drills.json, glossary.json, rules.pdf)
│   └── vector_store/            # ChromaDB 임베딩 벡터가 영구 저장되는 경로
├── src/                         # 애플리케이션 메인 소스 코드
│   ├── main.py                  # FastAPI 앱 진입점 (Lifespan 설정 및 라우터 등록)
│   ├── api/                     # 클라이언트와의 통신 인터페이스 (Controller)
│   │   └── v1/                  # API 버전 1
│   │       ├── endpoints/       # 기능별 엔드포인트 (gear.py, skill.py, whistle.py)
│   │       └── router.py        # 전체 라우터 통합 관리
│   ├── core/                    # 프로젝트 전역 설정
│   │   ├── config.py            # 환경 변수(.env) 로드 및 Pydantic Settings 관리
│   │   └── constants.py         # 프롬프트 템플릿 및 상수 값 관리
│   ├── models/                  # 데이터 검증 및 스키마 정의 (DTO)
│   │   ├── gear_schema.py       # 농구화 추천 입출력 모델
│   │   ├── skill_schema.py      # 훈련 루틴 입출력 모델
│   │   └── rule_schema.py       # 심판/규정 입출력 모델
│   ├── services/                # 핵심 비즈니스 로직 및 AI 엔진
│   │   ├── agents/              # LangGraph 노드 및 에이전트 로직 (ShoeAgent, CoachAgent, JudgeAgent)
│   │   ├── rag/                 # Vector DB(Chroma) 검색 및 임베딩 처리 로직
│   │   └── workflow.py          # LangGraph 상태 그래프(StateGraph) 및 워크플로우 정의
│   └── utils/                   # 공통 유틸리티 (PDF 파싱, 로거 설정 등)
├── tests/                       # pytest 기반 테스트 코드
├── .env                         # API Key (OpenAI) 및 보안 설정 파일 (Git 제외)
├── .gitignore                   # Git 업로드 제외 설정
├── pyproject.toml               # uv 패키지 의존성 및 Ruff 린터/포매터 설정
└── README.md                    # 프로젝트 문서
```

### 2. 주요 디렉터리 역할 (Directory Rules)

- **`data/`**: AI 모델이 참조할 지식 베이스입니다. `raw` 폴더에는 우리가 작성한 JSON/PDF 파일이 위치하며, 앱 실행 시 이 데이터가 `vector_store`에 임베딩되어 저장됩니다.

- **`src/api`**: 외부 요청을 가장 먼저 받는 곳입니다. 복잡한 로직 없이 입력값을 받아 `models`로 검증한 뒤, `services`의 워크플로우를 실행하고 결과를 반환하는 역할만 수행합니다.

- **`src/models`**: 데이터의 '형태'를 정의합니다. Pydantic을 사용하여 프론트엔드에서 넘어온 데이터가 유효한지(예: 키/몸무게 숫자 확인), LLM이 내뱉은 결과가 우리 형식에 맞는지 엄격하게 검사합니다.

- **`src/services`**: 프로젝트의 '두뇌'입니다.
  - `agents/`: 각 기능(신발, 훈련, 심판)을 담당하는 AI의 페르소나와 처리 로직이 담깁니다.
  - `rag/`: 사용자의 질문을 벡터로 변환하고 DB에서 유사한 데이터를 찾아오는 검색 엔진 역할을 합니다.
  - `workflow.py`: 여러 에이전트와 검색 도구를 연결하여 하나의 흐름(Graph)으로 제어하는 LangGraph의 핵심 코드가 위치합니다.

---

## API Response Convention

### 1. 성공 응답 (Success Response)

요청이 정상적으로 처리된 경우 HTTP 상태 코드 `200(OK)` 또는 `201(Created)`과 함께 아래 형식을 반환한다.

```json
{
  "success": true,
  "data": { ... },
  "message": "요청이 성공적으로 처리되었습니다."
}
```

> **규칙:** 실제 반환 데이터는 반드시 `data` 객체 내에 포함시켜 구조적 일관성을 유지한다.

### 2. 에러 응답 (Error Response)

요청 처리 중 오류가 발생한 경우 적절한 HTTP 상태 코드(`4xx`, `5xx`)와 함께 에러 정보를 반환한다.

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE_STRING",
    "message": "사용자에게 전달할 구체적인 에러 메시지"
  }
}
```

### 주요 에러 코드

| Error Code              | Description                                     | HTTP Status |
| ----------------------- | ----------------------------------------------- | ----------- |
| `INVALID_PARAMETER`     | 클라이언트의 입력값이 유효하지 않은 경우          | 400         |
| `UNAUTHORIZED`          | 인증되지 않은 접근이 발생한 경우                  | 401         |
| `NOT_FOUND`             | 요청한 리소스를 찾을 수 없는 경우                 | 404         |
| `INTERNAL_SERVER_ERROR` | 서버 내부 로직 오류 또는 외부 API 에러            | 500         |

---

## Security Policy

### 1. 환경 변수 관리 (Environment Variable Management)

모든 민감 정보는 소스 코드 내에 하드코딩하지 않으며, 반드시 외부 설정 파일을 통해 관리한다.

- **`.env` 활용:** OpenAI API 키, 데이터베이스 경로 등 노출 시 위험한 설정값은 프로젝트 루트의 `.env` 파일에 저장한다.
- **보안 우선 원칙:** `.env` 파일은 어떠한 경우에도 원격 저장소(GitHub)에 업로드하지 않는다.
- **Git 관리 제외:** `.gitignore` 파일에 `.env`를 명시하여 실수로 인한 유출을 원천 차단한다.

### 2. API 및 데이터 보안 (API & Data Security)

- **입력값 검증:** FastAPI와 Pydantic 모델을 사용하여 클라이언트로부터 전달되는 모든 입력값을 엄격히 검증함으로써 인젝션 공격 및 부적절한 데이터 유입을 방지한다.
- **에러 메시지 제어:** 운영 환경에서는 상세한 시스템 에러 메시지를 외부로 노출하지 않으며, 정의된 API Response Convention에 따라 추상화된 에러 정보만을 제공한다.
- **최소 권한 원칙:** 외부 API(OpenAI 등) 이용 시 필요한 최소한의 권한만을 가진 키를 발급받아 사용한다.

### 3. 의존성 및 코드 보안 (Dependency & Code Security)

- **안전한 패키지 관리:** `uv` 패키지 매니저를 사용하여 의존성 라이브러리의 버전을 고정(`uv.lock`)함으로써 공급망 공격 및 예기치 않은 라이브러리 업데이트로 인한 취약점 발생을 방지한다.
- **정적 분석 수행:** Ruff를 상시 가동하여 코드 내의 잠재적인 보안 결함이나 안티 패턴을 사전에 식별하고 수정한다.

### 4. GitHub 보안 (GitHub Security)

- **Secret Management:** 만약 GitHub Actions 등 자동화 도구를 사용할 경우, 모든 인증 정보는 GitHub Secrets를 통해 관리한다.
- **민감 정보 스캔:** 저장소에 민감 정보가 포함되었는지 확인하기 위해 필요시 보안 스캔 도구를 활용한다.
