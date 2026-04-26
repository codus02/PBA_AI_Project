"""Cocktail Mixer — 라즈베리파이 5 모터 제어 코드.

하드웨어:
  Pi 5 → GPIO → 8채널 MOSFET 보드 → 12V 페리스탈틱 펌프 × 8

동작:
  1. 메인 서버에서 레시피 JSON 받기 (output_agent.generate_motor_recipe 결과)
  2. pump_commands 순회하며 GPIO HIGH/LOW 로 펌프 ON/OFF
  3. manual_steps 는 화면/음성으로 사용자에게 안내

Fail-safe:
  - RPi.GPIO 없으면 (메인 PC 등) MockGPIO 자동 fallback → 코드 동일하게 작동
  - MOCK_GPIO=1 환경변수면 Pi 에서도 강제 mock (배선 전 dry-run)
  - 각 펌프 명령 try/except → 한 채널 실패해도 다음 진행
  - emergency_stop 함수 → 비상 시 모든 GPIO LOW
  - context manager 로 시작/종료 시 cleanup 보장

사용:
  # CLI: JSON 파일 받아 실행
  python3 cocktail_mixer.py recipe.json

  # CLI: stdin 으로 받기
  curl ... | python3 cocktail_mixer.py -

  # Python 모듈로 사용
  from cocktail_mixer import gpio_session, execute_recipe
  with gpio_session():
      result = execute_recipe(recipe_json)

테스트 (mock 모드):
  MOCK_GPIO=1 python3 cocktail_mixer.py recipe.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# 하드웨어 추상화 — Pi 가 아니거나 mock 강제 시 Mock GPIO 사용
# ============================================================

class MockGPIO:
    """RPi.GPIO 없는 환경 (메인 PC, Docker, Pi 의 mock 모드) 용 stub."""
    BCM = "BCM"
    BOARD = "BOARD"
    OUT = "OUT"
    IN = "IN"
    HIGH = 1
    LOW = 0
    PUD_UP = "PUD_UP"
    PUD_DOWN = "PUD_DOWN"

    @staticmethod
    def setmode(mode):
        logger.debug(f"[mock GPIO] setmode({mode})")

    @staticmethod
    def setwarnings(flag):
        logger.debug(f"[mock GPIO] setwarnings({flag})")

    @staticmethod
    def setup(pin, direction, initial=None, pull_up_down=None):
        logger.debug(f"[mock GPIO] setup(pin={pin}, dir={direction}, initial={initial})")

    @staticmethod
    def output(pin, state):
        sym = "HIGH" if state else "LOW"
        logger.info(f"[mock GPIO] pin {pin} → {sym}")

    @staticmethod
    def input(pin):
        return 0

    @staticmethod
    def cleanup(pins=None):
        logger.debug(f"[mock GPIO] cleanup({pins})")


def _resolve_gpio():
    """Pi 환경이면 RPi.GPIO 로드, 아니면 MockGPIO. MOCK_GPIO 환경변수로 강제 가능."""
    if os.getenv("MOCK_GPIO", "0").lower() in ("1", "true", "yes"):
        logger.warning("MOCK_GPIO=1 set — using mock GPIO (no real hardware)")
        return MockGPIO(), False

    try:
        import RPi.GPIO as GPIO
        return GPIO, True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"RPi.GPIO unavailable ({type(e).__name__}: {e}) — using mock GPIO")
        return MockGPIO(), False


GPIO, HARDWARE_AVAILABLE = _resolve_gpio()


# ============================================================
# 채널 ↔ Pi GPIO 핀 매핑
# ============================================================
#
# DB.Ingredient.pump_no (0~7) → BCM GPIO 핀 → MOSFET 보드 IN1~IN8
#
# 변경 시: 이 dict 만 수정. 코드 다른 곳 변경 불필요.
# 매핑 검증: scripts/pi_motor/wiring_test.py 참조 (각 채널 1초 ON 테스트).

CHANNEL_TO_GPIO = {
    0: 17,  # MOSFET IN1
    1: 27,  # MOSFET IN2
    2: 22,  # MOSFET IN3
    3: 23,  # MOSFET IN4
    4: 24,  # MOSFET IN5
    5: 25,  # MOSFET IN6
    6: 5,   # MOSFET IN7
    7: 6,   # MOSFET IN8
}

# 비상정지 버튼 (옵션). DB 안 쓰고 GPIO 직접. None 이면 비활성.
EMERGENCY_STOP_PIN: Optional[int] = None  # 예: 16 (16mm 버튼을 GPIO 16 에 연결 시)


# ============================================================
# GPIO setup/teardown
# ============================================================

def setup_gpio() -> None:
    """전체 GPIO 초기화. 시작 시 1회 호출."""
    try:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        for ch, pin in CHANNEL_TO_GPIO.items():
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        if EMERGENCY_STOP_PIN is not None:
            GPIO.setup(EMERGENCY_STOP_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        logger.info(
            f"GPIO setup OK ({len(CHANNEL_TO_GPIO)} channels, "
            f"hardware={'real' if HARDWARE_AVAILABLE else 'mock'})"
        )
    except Exception as e:
        logger.error(f"GPIO setup failed: {e}")
        raise


def teardown_gpio() -> None:
    """모든 GPIO LOW + cleanup. 종료 시 호출."""
    for pin in CHANNEL_TO_GPIO.values():
        try:
            GPIO.output(pin, GPIO.LOW)
        except Exception as e:
            logger.warning(f"failed to LOW pin {pin}: {e}")
    try:
        GPIO.cleanup()
    except Exception as e:
        logger.warning(f"GPIO.cleanup() error: {e}")
    logger.info("GPIO teardown OK")


@contextmanager
def gpio_session():
    """안전한 GPIO 라이프사이클 컨텍스트 매니저.

    with gpio_session():
        execute_recipe(...)
    """
    setup_gpio()
    try:
        yield
    finally:
        teardown_gpio()


# ============================================================
# 단일 펌프 제어
# ============================================================

def run_pump(channel: int, duration_ms: int) -> bool:
    """채널 channel 의 펌프를 duration_ms 동안 ON. 성공 True / 실패 False.

    실패 시:
      - 알 수 없는 채널 → False (다음 명령 진행)
      - duration <= 0 → True (skip)
      - GPIO 에러 → False, pin 강제 LOW 시도, 다음 명령 진행
    """
    if channel not in CHANNEL_TO_GPIO:
        logger.error(f"unknown channel {channel}, skipping")
        return False
    if duration_ms <= 0:
        logger.debug(f"channel {channel} duration={duration_ms}ms, skipping")
        return True

    pin = CHANNEL_TO_GPIO[channel]
    try:
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(duration_ms / 1000.0)
        GPIO.output(pin, GPIO.LOW)
        return True
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt during pump run — stopping")
        try:
            GPIO.output(pin, GPIO.LOW)
        except Exception:
            pass
        raise
    except Exception as e:
        logger.error(f"pump channel {channel} (pin {pin}) failed: {e}")
        try:
            GPIO.output(pin, GPIO.LOW)
        except Exception:
            pass
        return False


def emergency_stop() -> None:
    """모든 펌프 즉시 OFF. 비상 시 호출."""
    logger.warning("EMERGENCY STOP — all pumps OFF")
    for ch, pin in CHANNEL_TO_GPIO.items():
        try:
            GPIO.output(pin, GPIO.LOW)
        except Exception as e:
            logger.critical(f"emergency stop ch {ch} (pin {pin}) failed: {e}")


def check_emergency_button() -> bool:
    """비상정지 버튼 눌렸으면 True. EMERGENCY_STOP_PIN 미설정 시 항상 False."""
    if EMERGENCY_STOP_PIN is None:
        return False
    try:
        # PUD_UP → 누르지 않으면 HIGH(1), 누르면 LOW(0)
        return GPIO.input(EMERGENCY_STOP_PIN) == GPIO.LOW
    except Exception as e:
        logger.warning(f"emergency button read failed: {e}")
        return False


# ============================================================
# 레시피 실행
# ============================================================

def execute_recipe(
    recipe: dict,
    sequential: bool = True,
    max_concurrent: int = 4,
    inter_command_delay_sec: float = 0.3,
) -> dict:
    """output_agent.generate_motor_recipe 의 출력 받아 펌프 제어.

    Args:
        recipe: {"cocktail_name", "pump_commands": [...], "manual_steps": [...]}
        sequential: True = 펌프 한 번에 하나만 (안전, 느림)
                    False = 병렬 펌프 (빠름, 12V 10A 한도 주의)
        max_concurrent: parallel 모드에서 동시 ON 채널 수 상한
        inter_command_delay_sec: sequential 모드에서 펌프 간 마진 (튜브 잔여 안정화)

    Returns:
        {
          "cocktail_name": ...,
          "success": bool,
          "completed_channels": [...],
          "failed_channels": [...],
          "manual_steps": [...],   # 사람이 처리할 항목 (가니쉬 등)
          "duration_total_sec": float,
        }

    비상정지 버튼 눌리면 진행 중단하고 emergency_stop 호출.
    """
    name = recipe.get("cocktail_name", "?")
    commands = recipe.get("pump_commands", [])
    manual = recipe.get("manual_steps", [])

    logger.info(f"=== Mix '{name}' === ({len(commands)} pumps, {len(manual)} manual)")
    t0 = time.time()

    completed: list[int] = []
    failed: list[int] = []
    aborted = False

    if sequential:
        for cmd in commands:
            if check_emergency_button():
                logger.warning("emergency button pressed — aborting")
                emergency_stop()
                aborted = True
                break

            ch = cmd.get("channel")
            dur = cmd.get("duration_ms", 0)
            ing = cmd.get("ingredient_name", "?")
            amt = cmd.get("amount_ml", 0.0)
            logger.info(f"  → ch{ch} '{ing}' {amt}ml ({dur}ms)")

            ok = run_pump(ch, dur)
            (completed if ok else failed).append(ch)
            if inter_command_delay_sec > 0:
                time.sleep(inter_command_delay_sec)
    else:
        # 병렬 모드 — max_concurrent 단위로 묶어서 동시 실행.
        import threading
        for i in range(0, len(commands), max_concurrent):
            if check_emergency_button():
                emergency_stop()
                aborted = True
                break
            batch = commands[i: i + max_concurrent]
            results: dict[int, bool] = {}
            threads = []
            for cmd in batch:
                ch = cmd.get("channel")
                dur = cmd.get("duration_ms", 0)

                def _runner(c=ch, d=dur):
                    results[c] = run_pump(c, d)

                t = threading.Thread(target=_runner, daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            for ch, ok in results.items():
                (completed if ok else failed).append(ch)

    duration_total = round(time.time() - t0, 2)

    if manual:
        logger.info("수동 토핑 안내:")
        for step in manual:
            logger.info(
                f"  ☐ {step.get('ingredient_name','?')} "
                f"{step.get('amount_ml',0)}ml "
                f"({step.get('reason','')})"
            )

    success = (not aborted) and (len(failed) == 0)
    logger.info(
        f"=== Done: success={success}, completed={len(completed)}, "
        f"failed={len(failed)}, duration={duration_total}s ==="
    )

    return {
        "cocktail_id": recipe.get("cocktail_id"),
        "cocktail_name": name,
        "success": success,
        "aborted": aborted,
        "completed_channels": completed,
        "failed_channels": failed,
        "manual_steps": manual,
        "duration_total_sec": duration_total,
    }


# ============================================================
# CLI entry point
# ============================================================

def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _read_recipe(arg: str) -> dict:
    if arg == "-":
        return json.load(sys.stdin)
    with open(arg, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    _setup_logging()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 cocktail_mixer.py <recipe.json>")
        print("  python3 cocktail_mixer.py -        # stdin")
        print()
        print("Env vars:")
        print("  MOCK_GPIO=1   # 강제 mock (배선 없이 dry-run)")
        sys.exit(2)

    recipe = _read_recipe(sys.argv[1])

    with gpio_session():
        result = execute_recipe(recipe)

    # 결과 stdout 으로 (메인 서버 가 받게)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
