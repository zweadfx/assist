"""
A module to store project-wide constants.
"""

from pathlib import Path

# Base Directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Data Files
SHOES_FILE_PATH = RAW_DATA_DIR / "shoes.json"
PLAYERS_FILE_PATH = RAW_DATA_DIR / "players.json"
GLOSSARY_FILE_PATH = RAW_DATA_DIR / "glossary.json"
FIBA_RULES_PDF_PATH = RAW_DATA_DIR / "fiba_rules.pdf"
NBA_RULES_PDF_PATH = RAW_DATA_DIR / "nba_rules.pdf"

# Pre-calculated Embeddings
SHOES_EMBEDDINGS_FILE_PATH = RAW_DATA_DIR / "shoes_embeddings.json"
PLAYERS_EMBEDDINGS_FILE_PATH = RAW_DATA_DIR / "players_embeddings.json"
RULES_EMBEDDINGS_FILE_PATH = RAW_DATA_DIR / "rules_embeddings.json"
GLOSSARY_EMBEDDINGS_FILE_PATH = RAW_DATA_DIR / "glossary_embeddings.json"

# Sensory Tag Enum → Korean Label Mapping
SENSORY_TAG_MAP: dict[str, str] = {
    "cushioning": "쿠셔닝",
    "responsive": "반응성",
    "lightweight": "경량",
    "ankle_support": "발목 지지",
    "traction": "접지력",
    "wide_fit": "와이드 핏",
    "narrow_fit": "내로우 핏",
    "stability": "안정성",
    "comfort": "편안함",
    "court_feel": "코트 감각",
    "durability": "내구성",
    "flexibility": "유연성",
    "breathable": "통기성",
    "lockdown": "락다운",
    "impact_protection": "충격 보호",
    "value": "가성비",
}

# Korean Basketball Terminology Guide (injected into prompts when language=ko)
KO_BASKETBALL_TERMINOLOGY = """
**CRITICAL — 한국어 농구 용어 규칙 (반드시 준수):**
아래 용어 매핑표를 반드시 따를 것. 영어를 절대 직역하지 말 것.
(예: free throw → "프리드로우" X, "자유투" O / basket → "바구니" X, "림" O)

| English | 한국어 |
|---------|--------|
| free throw | 자유투 |
| layup / lay-in | 레이업 |
| reverse layup | 리버스 레이업 |
| finger roll | 핑거롤 |
| dunk | 덩크 |
| three-pointer / three-point shot | 3점슛 |
| mid-range shot | 미드레인지 슛 |
| bank shot | 뱅크 슛 |
| hook shot | 훅 슛 |
| tip-in | 팁인 |
| alley-oop | 앨리웁 |
| elbow (court area) | 엘보 |
| elbow jumper | 엘보 점퍼 |
| pull-up jumper | 풀업 점퍼 |
| step-back / retreat | 스텝백 |
| floater | 플로터 |
| euro step | 유로스텝 |
| crossover | 크로스오버 |
| behind the back | 비하인드 더 백 |
| between the legs | 비트윈 더 레그 |
| in-and-out | 인앤아웃 |
| hesitation | 헤지테이션 |
| figure-8 | 8자 드리블 |
| wrap-around / ball wrap | 볼 래핑 |
| pound dribble | 파운드 드리블 |
| power dribble | 파워 드리블 |
| speed dribble | 스피드 드리블 |
| retreat dribble | 리트릿 드리블 |
| weak hand / non-dominant hand | 약손 |
| ball handling | 볼 핸들링 |
| triple threat | 트리플 스렛 |
| pivot | 피벗 |
| jab step | 잽 스텝 |
| pump fake | 펌프 페이크 |
| face-up | 페이스업 |
| drop step | 드롭 스텝 |
| up-and-under | 업앤언더 |
| post up | 포스트업 |
| pick and roll | 픽앤롤 |
| screen | 스크린 |
| fast break | 속공 |
| transition | 전환 공격 |
| basket / rim | 림 |
| paint / lane | 페인트존 |
| baseline | 베이스라인 |
| outlet pass | 아울렛 패스 |
| chest pass | 체스트 패스 |
| bounce pass | 바운스 패스 |
| overhead pass | 오버헤드 패스 |
| no-look pass | 노룩 패스 |
| closeout | 클로즈아웃 |
| defensive slide | 디펜스 슬라이드 |
| defensive stance | 수비 자세 |
| help defense | 헬프 디펜스 |
| man-to-man | 대인 수비 |
| zone defense | 지역 수비 |
| recovery step | 리커버리 스텝 |
| steal | 스틸 |
| block / shot block | 블록 |
| box out | 박스 아웃 |
| rebound | 리바운드 |
| assist | 어시스트 |
| turnover | 턴오버 |
| charge | 차지 |
| follow-through | 팔로스루 |
| release | 릴리스 |
| arc | 슈팅 아크 |
| form shooting | 폼 슈팅 |
| spot-up | 스팟업 |
| catch and shoot | 캐치 앤 슛 |
| off-the-dribble | 드리블 슈팅 |
| suicide sprints / shuttle run | 셔틀런 |
| agility ladder | 사다리 스텝 |
| lateral bound | 래터럴 바운드 |
| box jump | 박스 점프 |
| wall sit | 월 싯 |
| plyometrics | 플라이오메트릭 |
| warm-up / warmup | 워밍업 |
| cool-down / cooldown | 쿨다운 |
| combination dribble / combo dribble | 컴비네이션 드리블 |
| dynamic stretching | 동적 스트레칭 |
| static stretching | 정적 스트레칭 |

- 위 표에 있는 용어는 반드시 한국어 컬럼의 표현만 사용할 것
- 표에 없는 용어도 한국 농구에서 실제로 사용하는 표현으로 번역하고, 직역은 절대 금지
- Reference Drills의 Drill Name이 이미 한국어로 제공된 경우 그대로 사용할 것
"""

# Endpoint Timeouts (seconds)
SKILL_TIMEOUT_SECONDS = 60
WEEKLY_TIMEOUT_SECONDS = 120
GEAR_TIMEOUT_SECONDS = 60
JUDGMENT_TIMEOUT_SECONDS = 60

# ChromaDB Collection Names
SHOES_COLLECTION_NAME = "basketball_shoes"
PLAYERS_COLLECTION_NAME = "basketball_players"
RULES_COLLECTION_NAME = "basketball_rules"
GLOSSARY_COLLECTION_NAME = "basketball_glossary"

# LLM-as-Judge Prompts
LLM_JUDGE_WHISTLE_SYSTEM_PROMPT = (
    "너는 농구 규정을 완벽히 숙지한 엄격한 RAG 평가자야. 반드시 순수 JSON으로만 답해."
)

LLM_JUDGE_WHISTLE_PROMPT = (
    "다음 농구 규칙 판정 답변을 평가하고, "
    "지정된 3가지 기준에 따라 1점부터 5점까지 채점해.\n"
    "반드시 마크다운 코드 블록 없이 순수한 JSON 형태({{...}})로만 출력해라.\n\n"
    "평가 기준:\n"
    "1. 정확성 (Accuracy): 생성된 답변이 예상 정답(판정)과 일치하며, "
    "농구 규정에 맞게 상황을 정확히 판단했는가?\n"
    "2. 논리 일관성 (Logical Consistency): 판정의 이유를 설명하는 논리 전개가 "
    "명확하고 모순이 없는가?\n"
    "3. 규칙 인용 적절성 (Citation Appropriateness): 근거로 제시한 조항(Article)이 "
    "주어진 상황에 적절하며 올바르게 인용되었는가?\n\n"
    "출력 형식 (오직 순수 JSON만 반환):\n"
    "{{\n"
    '  "accuracy_score": <int>,\n'
    '  "consistency_score": <int>,\n'
    '  "citation_score": <int>,\n'
    '  "reasoning": "<string explaining the scores in Korean>"\n'
    "}}\n\n"
    "---\n"
    "Question (Context): {context} {question}\n"
    "Expected Answer: {expected_answer}\n"
    "Generated Answer: {generated_answer}\n"
)

LLM_JUDGE_GEAR_SYSTEM_PROMPT = (
    "너는 농구화 추천 품질을 평가하는 엄격한 RAG 평가자야. 반드시 순수 JSON으로만 답해."
)

LLM_JUDGE_GEAR_PROMPT = (
    "다음 농구화 추천 답변을 평가하고, "
    "지정된 3가지 기준에 따라 1점부터 5점까지 채점해.\n"
    "반드시 마크다운 코드 블록 없이 순수한 JSON 형태({{...}})로만 출력해라.\n\n"
    "평가 기준:\n"
    "1. 정확성 (Accuracy): 추천된 신발이 사용자의 sensory preferences, player archetype, "
    "budget, position 조건에 잘 맞는가? 예상 정답(expected shoe IDs)과 비교해 판단.\n"
    "2. 논리 일관성 (Logical Consistency): 각 신발의 추천 이유가 사용자 조건과 모순 없이 "
    "논리적으로 설명되어 있는가?\n"
    "3. 데이터 충실도 (Data Fidelity): 응답의 신발 정보(brand, model, price, sensory tags)가 "
    "RAG로 검색된 실제 데이터를 정확히 반영했는가? 임의로 만들어낸 정보가 없는가?\n\n"
    "출력 형식 (오직 순수 JSON만 반환):\n"
    "{{\n"
    '  "accuracy_score": <int>,\n'
    '  "consistency_score": <int>,\n'
    '  "citation_score": <int>,\n'
    '  "reasoning": "<string explaining the scores in Korean>"\n'
    "}}\n\n"
    "---\n"
    "Question (Context): {context} {question}\n"
    "Expected Answer: {expected_answer}\n"
    "Generated Answer: {generated_answer}\n"
)

# Backward-compatible aliases
LLM_JUDGE_SYSTEM_PROMPT = LLM_JUDGE_WHISTLE_SYSTEM_PROMPT
LLM_JUDGE_EVAL_PROMPT = LLM_JUDGE_WHISTLE_PROMPT
