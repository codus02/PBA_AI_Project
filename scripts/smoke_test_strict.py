"""
전체 플로우 스모크 테스트 (엄격 모드)
- 80% 완성 / 7턴 / 사용자 중단 중 하나 충족해야 추천 진입 (force 금지)
- 피드백 루프 최대 3회 → 3회차에 ACCEPT → 최종 확정 → 만족도 입력 → 종료
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


def main():
    # 1. party
    r = requests.post(
        f"{BASE}/sessions/party",
        json={"session_name": "strict smoke test party", "demo_mode": True},
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

    # 3. initial tags
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

    # 4. space image
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

    # 6. 슬롯을 충분히 채우기 위한 엄격 메시지
    messages = [
        "생일파티예요",
        "기분이 신나요",
        "단맛이 좋아요",
        "위스키는 싫어요",
        "적당히 마시고 싶어요",
        "모히토를 좋아해요",
        "끝맛은 깔끔하게 좋겠어요",
        "과일향 좋아요",
    ]

    last_dialogue = None
    for idx, msg in enumerate(messages, start=1):
        print(f"\n[USER-{idx}] {msg}")
        r = requests.post(
            f"{BASE}/sessions/{guest_id}/dialogue",
            json={"message": msg},
            timeout=20,
        )
        last_dialogue = expect_ok(r, f"dialogue-{idx}")

        if last_dialogue.get("status") == "proceed_to_recommendation":
            break
        if last_dialogue.get("should_proceed"):
            break

    if last_dialogue is None:
        raise RuntimeError("dialogue result is empty")

    print("\n=== dialogue summary ===")
    print("status:", last_dialogue.get("status"))
    print("completion:", last_dialogue.get("completion"))
    print("reason:", last_dialogue.get("reason"))

    proceed = last_dialogue.get("should_proceed") or \
              last_dialogue.get("status") == "proceed_to_recommendation"
    if not proceed:
        raise RuntimeError(
            "strict smoke test failed: 추천 진입 조건(80%/7턴/user_skip) 미충족"
        )

    # 7. 1차 시음용 추천 (엄격: force=false)
    r = requests.post(
        f"{BASE}/sessions/{guest_id}/recommend-sample",
        timeout=20,
    )
    rec_data = expect_ok(r, "recommend-sample")
    if rec_data.get("status") != "ok":
        raise RuntimeError(f"recommendation failed in strict mode: {rec_data}")

    sample_rec_id = rec_data["sample_recommendation_id"]

    # 8. 피드백 루프 (최대 3회, 마지막엔 ACCEPT)
    feedback_msgs = [
        "조금 더 달게 해줘",      # ADJUST
        "조금 더 시게 해줘",      # ADJUST
        "좋아 이걸로 할게요",     # ACCEPT
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

        if result.get("status") == "accepted":
            final_result = result
            break

        next_sample_id = result.get("sample_recommendation_id")
        if not next_sample_id:
            raise RuntimeError(f"재추천 실패: {result}")
        current_sample_id = next_sample_id

    if final_result is None:
        raise RuntimeError("3회 내 ACCEPT 도달 실패")

    final_rec_id = final_result["final_recommendation_id"]

    # 9. 최종 출력 조회
    r = requests.get(f"{BASE}/final-output/{final_rec_id}", timeout=20)
    final_output = expect_ok(r, "final-output")
    if not final_output:
        raise RuntimeError("final output empty")

    # 10. 만족도 입력
    r = requests.post(
        f"{BASE}/sessions/{guest_id}/evaluation",
        json={
            "final_recommendation_id": final_rec_id,
            "satisfaction_score": 4.5,
            "would_reorder": True,
            "review_text": "엄격 테스트 만족",
        },
        timeout=20,
    )
    eval_data = expect_ok(r, "evaluation")
    if eval_data.get("status") != "ok":
        raise RuntimeError(f"evaluation failed: {eval_data}")

    print("\n[PASS] strict smoke test 전체 플로우 완료")


if __name__ == "__main__":
    main()
