import logging
import os
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

# 데모 생존용 최소 레이트리밋 — LLM을 실제로 호출하는 엔드포인트에만 적용.
# 인메모리 카운터: 단일 인스턴스 전제, 재시작 시 리셋 허용.
IP_DAILY_LIMIT = int(os.getenv("RATE_LIMIT_IP_PER_DAY", "20"))
GLOBAL_DAILY_LIMIT = int(os.getenv("RATE_LIMIT_GLOBAL_PER_DAY", "200"))
WINDOW_S = 24 * 3600  # 롤링 24h — '자정 리셋'이 아니라 '약 24시간 후 해제'

RATE_LIMIT_MESSAGE = "오늘 요청 한도에 도달했어요. 내일 다시 시도해주세요."

_ip_hits: dict[str, deque[float]] = defaultdict(deque)
_global_hits: deque[float] = deque()


def client_ip(request: Request) -> str:
    # Render 프록시 뒤에서는 X-Forwarded-For의 첫 값이 실제 클라이언트 IP다.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _prune(hits: deque[float], now: float) -> None:
    while hits and now - hits[0] > WINDOW_S:
        hits.popleft()


async def enforce_rate_limit(request: Request) -> None:
    """IP당·전역 롤링 24h 한도를 확인하고 통과 시 카운트를 소비한다.

    거절된 요청은 카운트를 소비하지 않는다.
    """
    ip = client_ip(request)
    now = time.time()
    _prune(_global_hits, now)
    hits = _ip_hits[ip]
    _prune(hits, now)
    if len(_global_hits) >= GLOBAL_DAILY_LIMIT or len(hits) >= IP_DAILY_LIMIT:
        logger.warning(
            "Rate limit exceeded: ip=%s ip_count=%d/%d global_count=%d/%d",
            ip,
            len(hits),
            IP_DAILY_LIMIT,
            len(_global_hits),
            GLOBAL_DAILY_LIMIT,
        )
        raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
    _global_hits.append(now)
    hits.append(now)
    logger.info(
        "LLM request accepted: ip=%s ip_count=%d/%d global_count=%d/%d",
        ip,
        len(hits),
        IP_DAILY_LIMIT,
        len(_global_hits),
        GLOBAL_DAILY_LIMIT,
    )
