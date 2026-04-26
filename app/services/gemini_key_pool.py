"""Gemini API 키 로테이션 풀.

설계:
  * 키는 .env 의 ``GOOGLE_API_KEYS`` (콤마 구분) 또는 단일 ``GOOGLE_API_KEY``.
  * 분당/일간 사용량은 ``gemini_key_usage`` 테이블에 atomic 하게 기록.
  * ``acquire()`` 가 사용 가능한 키 한 개를 선택해 카운터 증가 + 커밋.
  * 호출 중 quota error 발생하면 ``mark_quota_exceeded()`` 로 그 키 강제 소진.
  * 모든 키가 한도 초과면 ``QuotaExhaustedError``.

동시성:
  * `SELECT ... FOR UPDATE` 로 row 락. 같은 key_id 에 대한 두 acquire 는
    DB 에서 직렬화됨. 키가 N개면 동시 처리량은 N 까지.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Iterator, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import GeminiKeyUsage

logger = logging.getLogger(__name__)


# === 한도 상수 (요구사항: 키별 1분 5회, 1일 20회) ============================
QUOTA_PER_MINUTE = int(os.getenv("GEMINI_QUOTA_PER_MINUTE", "5"))
QUOTA_PER_DAY = int(os.getenv("GEMINI_QUOTA_PER_DAY", "20"))


class QuotaExhaustedError(RuntimeError):
    """모든 Gemini 키가 분당/일간 한도를 넘겨서 사용 불가."""


def _key_id(api_key: str) -> str:
    """API 키를 SHA256 해시 앞 16자로 anonymize. DB 저장용."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


_NUMBERED_KEY_RE = re.compile(r"^GOOGLE_API_KEY(\d+)$")


def _parse_keys_from_env() -> List[Tuple[str, str]]:
    """환경변수에서 (label, api_key) 리스트 추출. 세 가지 형식 모두 지원.

    수집 순서 (중복 키는 한 번만):
      1. ``GOOGLE_API_KEYS`` (콤마 구분 단일 변수)
      2. ``GOOGLE_API_KEY1``, ``GOOGLE_API_KEY2``, ... (번호 변수, 번호 오름차순)
      3. ``GOOGLE_API_KEY`` (legacy 단일)
    """
    keys: List[Tuple[str, str]] = []
    seen: set[str] = set()

    # 1) 콤마 구분 GOOGLE_API_KEYS
    bulk = os.environ.get("GOOGLE_API_KEYS", "").strip()
    if bulk:
        for i, k in enumerate([k.strip() for k in bulk.split(",") if k.strip()]):
            if k not in seen:
                keys.append((f"key_{i+1}", k))
                seen.add(k)

    # 2) 번호 변수 GOOGLE_API_KEY{N} (모든 N 스캔, 번호 오름차순)
    numbered: List[Tuple[int, str]] = []
    for env_var, val in os.environ.items():
        m = _NUMBERED_KEY_RE.match(env_var)
        if not m:
            continue
        v = (val or "").strip()
        if v:
            numbered.append((int(m.group(1)), v))
    numbered.sort(key=lambda x: x[0])
    for n, k in numbered:
        if k not in seen:
            keys.append((f"key_{n}", k))
            seen.add(k)

    # 3) legacy GOOGLE_API_KEY (마지막)
    single = os.environ.get("GOOGLE_API_KEY", "").strip()
    if single and single not in seen:
        keys.append(("key_legacy", single))

    return keys


def _ensure_row(db: Session, kid: str, label: str) -> GeminiKeyUsage:
    """row 가 없으면 초기 row 생성. 호출 전 락은 별도 처리."""
    row = db.query(GeminiKeyUsage).filter_by(key_id=kid).first()
    if row is None:
        row = GeminiKeyUsage(
            key_id=kid,
            label=label,
            minute_count=0,
            day_count=0,
        )
        db.add(row)
        db.flush()
    return row


def _reset_windows_in_place(row: GeminiKeyUsage, now: datetime, today: date) -> None:
    """분/일 윈도우가 만료됐으면 그 자리에서 카운터 리셋."""
    if row.minute_window_start is None or (now - row.minute_window_start) >= timedelta(seconds=60):
        row.minute_window_start = now
        row.minute_count = 0
    if row.day_window_start != today:
        row.day_window_start = today
        row.day_count = 0


@contextmanager
def _session_scope() -> Iterator[Session]:
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def acquire() -> Tuple[str, str, str]:
    """사용 가능한 키 1개를 선택, 카운터 증가하고 (label, api_key, key_id) 반환.

    선택 우선순위는 .env 에 적힌 순서. 모든 키가 분당/일간 한도를 넘으면
    ``QuotaExhaustedError`` 발생.

    동시성: 각 키 row 에 ``SELECT ... FOR UPDATE`` 를 걸어 카운터 증가 직렬화.
    """
    keys = _parse_keys_from_env()
    if not keys:
        raise RuntimeError(
            "Gemini 키 미설정: .env 에 GOOGLE_API_KEYS (콤마 구분) 또는 GOOGLE_API_KEY 추가"
        )

    now = datetime.utcnow()
    today = now.date()

    with _session_scope() as db:
        for label, api_key in keys:
            kid = _key_id(api_key)
            # row 없으면 먼저 생성 (락 거는 SELECT 가 빈 결과 받지 않게).
            _ensure_row(db, kid, label)

            row = (
                db.query(GeminiKeyUsage)
                .filter_by(key_id=kid)
                .with_for_update()
                .one()
            )

            _reset_windows_in_place(row, now, today)

            if row.is_disabled:
                logger.debug("gemini key %s skipped: disabled", label)
                continue

            if row.minute_count >= QUOTA_PER_MINUTE:
                logger.debug(
                    "gemini key %s skipped: minute=%d >= %d",
                    label, row.minute_count, QUOTA_PER_MINUTE,
                )
                continue
            if row.day_count >= QUOTA_PER_DAY:
                logger.debug(
                    "gemini key %s skipped: day=%d >= %d",
                    label, row.day_count, QUOTA_PER_DAY,
                )
                continue

            row.minute_count += 1
            row.day_count += 1
            row.last_used_at = now
            row.updated_at = now
            # commit 은 _session_scope 가 처리 → 락 해제
            return label, api_key, kid

    raise QuotaExhaustedError(
        f"모든 Gemini 키 한도 초과 (분당 {QUOTA_PER_MINUTE} / 일간 {QUOTA_PER_DAY})"
    )


def mark_quota_exceeded(key_id: str) -> None:
    """API 호출 중 quota 에러가 나면 그 키를 즉시 분당 한도 강제 충족 처리.

    ``last_quota_at`` 도 기록해 모니터링 / 디버깅에 도움.
    """
    now = datetime.utcnow()
    today = now.date()
    with _session_scope() as db:
        row = (
            db.query(GeminiKeyUsage)
            .filter_by(key_id=key_id)
            .with_for_update()
            .one_or_none()
        )
        if row is None:
            return
        # 분당 한도까지 점프시켜 다음 acquire 에서 skip 되게.
        _reset_windows_in_place(row, now, today)
        row.minute_count = max(row.minute_count, QUOTA_PER_MINUTE)
        # 일간 한도까지 점프시킬지는 보수적으로 NO (분당만 막음). Gemini 가
        # 일일 한도까지 넘었다고 명시적으로 알려줄 때만 추가 처리.
        row.last_quota_at = now
        row.updated_at = now
        logger.warning("gemini key %s marked quota_exceeded", row.label)


def is_gemini_quota_error(exc: BaseException) -> bool:
    """Gemini SDK 예외에서 quota 초과 시그널 검출.

    google-genai 는 ``google.genai.errors.ClientError`` 를 던지고,
    HTTP 상태 429 / "RESOURCE_EXHAUSTED" 를 메시지에 포함한다. SDK 버전마다
    노출 형태가 달라 status code 와 메시지 양쪽 다 검사.
    """
    msg = str(exc).lower()
    if "resource_exhausted" in msg or "429" in msg or "quota" in msg or "rate limit" in msg:
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status == 429:
        return True
    return False


def snapshot() -> List[dict]:
    """현재 키 풀 상태 스냅샷. 디버깅/대시보드용. label/사용량/디스에이블 여부 반환."""
    keys = _parse_keys_from_env()
    if not keys:
        return []

    now = datetime.utcnow()
    today = now.date()
    out: List[dict] = []
    with _session_scope() as db:
        for label, api_key in keys:
            kid = _key_id(api_key)
            row = db.query(GeminiKeyUsage).filter_by(key_id=kid).one_or_none()
            if row is None:
                out.append({
                    "label": label, "key_id": kid,
                    "minute_count": 0, "minute_remaining": QUOTA_PER_MINUTE,
                    "day_count": 0, "day_remaining": QUOTA_PER_DAY,
                    "is_disabled": False,
                })
                continue
            # 보고용 — DB 에 쓰진 않음. 만료 윈도우면 0 으로 표시.
            min_count = row.minute_count
            if row.minute_window_start is None or (now - row.minute_window_start) >= timedelta(seconds=60):
                min_count = 0
            day_count = row.day_count
            if row.day_window_start != today:
                day_count = 0
            out.append({
                "label": label,
                "key_id": kid,
                "minute_count": min_count,
                "minute_remaining": max(QUOTA_PER_MINUTE - min_count, 0),
                "day_count": day_count,
                "day_remaining": max(QUOTA_PER_DAY - day_count, 0),
                "is_disabled": row.is_disabled,
                "last_used_at": row.last_used_at.isoformat() if row.last_used_at else None,
                "last_quota_at": row.last_quota_at.isoformat() if row.last_quota_at else None,
            })
    return out
