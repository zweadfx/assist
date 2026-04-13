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
| warm-up | 워밍업 |
| cool-down | 마무리 |
| dynamic stretching | 동적 스트레칭 |
| static stretching | 정적 스트레칭 |

- 위 표에 있는 용어는 반드시 한국어 컬럼의 표현만 사용할 것
- 표에 없는 용어도 한국 농구에서 실제로 사용하는 표현으로 번역하고, 직역은 절대 금지
- Reference Drills의 Drill Name이 이미 한국어로 제공된 경우 그대로 사용할 것
"""

# ChromaDB Collection Names
SHOES_COLLECTION_NAME = "basketball_shoes"
PLAYERS_COLLECTION_NAME = "basketball_players"
RULES_COLLECTION_NAME = "basketball_rules"
GLOSSARY_COLLECTION_NAME = "basketball_glossary"
