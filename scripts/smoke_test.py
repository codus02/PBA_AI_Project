# scripts/smoke_test.py
"""
전체 플로우 스모크 테스트 (관대 모드)
- 초기 태그 → 공간 이미지 → 대화(슬롯 수집) → 1차 추천
- 피드백 루프 최대 3회 → 최종 확정 → 만족도 입력 → 세션 종료
- 80% 미달이어도 force=true 허용
"""
import json
import os
from pathlib import Path

import requests
from PIL import Image

BASE = os.getenv("BASE_URL", "http://141.223.140.32:8000/api/v1")
MAX_FEEDBACK_ROUNDS = 3


def expect_ok(resp, step_name):
    print(f"[{step_name}] status={resp.status_code}")
    try:
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        print(resp.text)
        raise RuntimeError(f"{step_name} returned non-JSON response")

    if not resp.ok:
        raise RuntimeError(f"{step_name} failed: {resp.status_code}")
    return data


def run_dialogue_until_proceed(guest_id: str, messages: list[str]) -> dict:
    """슬롯이 찰 때까지 또는 사용자가 중단 의사 표현할 때까지 대화."""
    last = None
    for idx, msg in enumerate(messages, start=1):
        print(f"\n[USER-{idx}] {msg}")
        r = requests.post(
            f"{BASE}/sessions/{guest_id}/dialogue",
            json={"message": msg},
            timeout=20,
        )
        last = expect_ok(r, f"dialogue-{idx}")

        if last.get("status") == "proceed_to_recommendation":
            break
        if last.get("should_proceed"):
            break
    return last


def run_feedback_loop(guest_id: str, sample_rec_id: str) -> dict:
    """피드백 루프 최대 3회. 3회 이후엔 ACCEPT로 마지막 추천 확정."""
    feedback_msgs = [
        "조금 더 달게 해줘",
        "조금 더 시게 해줘",
        "좋아 이걸로 할게요",  # 3회차는 ACCEPT 유도
    ]

    current_sample_id = sample_rec_id
    final_result = None

    for round_idx in range(1, MAX_FEEDBACK_ROUNDS + 1):
        msg = feedback_msgs[round_idx - 1]
        print(f"\n[FEEDBACK-{round_idx}] {msg}")
        r = requests.post(
            f"{BASE}/sessions/{guest_id}/feedback",
            json={
                "sample_recommendation_id": current_sample_id,
                "feedback_text": msg,
            },
            timeout=20,
        )
        result = expect_ok(r, f"feedback-{round_idx}")

        if result.get("status") in ("accepted", "force_finalized"):
            final_result = result
            break

        next_sample_id = result.get("sample_recommendation_id")
        if next_sample_id:
            current_sample_id = next_sample_id
        else:
            print("재추천 실패. 현재 샘플 유지")

    if final_result is None:
        # 3회 이후에도 확정 안 됐을 경우: 강제 ACCEPT
        print("\n[FEEDBACK-FORCE] 3회 소진. 현재 추천을 강제 확정")
        r = requests.post(
            f"{BASE}/sessions/{guest_id}/feedback",
            json={
                "sample_recommendation_id": current_sample_id,
                "feedback_text": "좋아요 이걸로 할게요",
            },
            timeout=20,
        )
        final_result = expect_ok(r, "feedback-force-accept")

    return final_result


def main():
    # 1. party
    r = requests.post(
        f"{BASE}/sessions/party",
        json={"session_name": "smoke test party", "demo_mode": True},
        timeout=20,
    )
    party_id = expect_ok(r, "party")["party_session_id"]

    # 2. guest
    r = requests.post(
        f"{BASE}/sessions/guest",
        json={"party_session_id": party_id, "guest_label": "게스트1"},
        timeout=20,
    )
    guest_id = expect_ok(r, "guest")["guest_session_id"]

    # 3. initial tags (내부 분석 1단계)
    r = requests.post(
        f"{BASE}/sessions/{guest_id}/tags",
        json={
            "familiarity": "가끔 마셔요",
            "tastes": ["단맛", "청량감"],
            "strength": "적당히",
            "aromas": ["과일향"],
        },
        timeout=20,
    )
    expect_ok(r, "tags")

    # 4. space image (내부 분석 2단계)
    image_path = Path("data/test.jpg")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        Image.new("RGB", (256, 256), color=(255, 180, 120)).save(image_path)

    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE}/sessions/{guest_id}/space-image",
            files={"file": ("test.jpg", f, "image/jpeg")},
            timeout=20,
        )
    expect_ok(r, "space-image")

    # 5. LLM 첫 질문
    r = requests.post(f"{BASE}/sessions/{guest_id}/start-dialogue", timeout=20)
    start_data = expect_ok(r, "start-dialogue")
    print("\n[LLM-FIRST] 첫 질문:", start_data.get("question"))

    # 6. 사용자 답변 + 꼬리 질문 반복
    messages = [
        "생일파티인데 달콤한 게 좋아요",
        "기분은 엄청 좋고 과일향 나는 칵테일이면 좋겠어요",
        "도수는 적당한 편이 좋고 민트향은 싫어요. 끝맛은 깔끔했으면 좋겠어요",
        "이제 추천해줘",  # 사용자 중단 의사
    ]
    last_dialogue = run_dialogue_until_proceed(guest_id, messages)

    print("\n=== dialogue summary ===")
    print("status:", last_dialogue.get("status"))
    print("completion:", last_dialogue.get("completion"))
    print("reason:", last_dialogue.get("reason"))

    # 7. 1차 시음용 추천 (관대 모드: 항상 force=true — user_skip/정보부족에도 진행)
    r = requests.post(
        f"{BASE}/sessions/{guest_id}/recommend-sample",
        params={"force": "true"},
        timeout=20,
    )
    rec_data = expect_ok(r, "recommend-sample")
    if rec_data.get("status") != "ok":
        raise RuntimeError(f"recommendation failed: {rec_data}")

    sample_rec_id = rec_data["sample_recommendation_id"]

    # 8. 피드백 루프 (최대 3회, 이후 강제 확정)
    final_result = run_feedback_loop(guest_id, sample_rec_id)
    if final_result.get("status") not in ("accepted", "force_finalized"):
        raise RuntimeError(f"final confirmation failed: {final_result}")

    final_rec_id = final_result["final_recommendation_id"]

    # 9. 최종 출력 조회
    r = requests.get(f"{BASE}/final-output/{final_rec_id}", timeout=20)
    expect_ok(r, "final-output")

    # 10. 만족도 입력
    r = requests.post(
        f"{BASE}/sessions/{guest_id}/evaluation",
        json={
            "final_recommendation_id": final_rec_id,
            "satisfaction_score": 4.5,
            "would_reorder": True,
            "review_text": "맛있었어요",
        },
        timeout=20,
    )
    expect_ok(r, "evaluation")

    print("\n[PASS] smoke test 전체 플로우 완료")


if __name__ == "__main__":
    main()
