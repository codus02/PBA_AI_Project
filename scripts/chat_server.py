"""Simple Flask server wrapping chat_qwen_slots logic."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, request, jsonify, send_from_directory
from app.agents.preference_agent import (
    analyze_user_turn,
    generate_opening_question,
    merge_slots,
    should_move_to_recommendation,
    _calc_effective_completion,
)

MAX_USER_TURNS = 7

app = Flask(__name__)

# 세션 상태 (단일 사용자)
state: dict = {"history": [], "slots": {}, "turn": 0, "done": False}


@app.get("/")
def index():
    return send_from_directory(Path(__file__).parent, "chat.html")


@app.post("/api/start")
def start():
    state["history"] = []
    state["slots"] = {}
    state["turn"] = 0
    state["done"] = False
    first_q = generate_opening_question()
    state["history"].append({"speaker_role": "LLM", "utterance_text": first_q})
    return jsonify({"message": first_q})


@app.post("/api/chat")
def chat():
    if state["done"]:
        return jsonify({"error": "대화가 종료됐어요."})

    user_msg = request.json.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "메시지가 없어요."})

    state["turn"] += 1
    result = analyze_user_turn(
        history=state["history"], slots=state["slots"], user_msg=user_msg
    )
    state["slots"] = merge_slots(state["slots"], result["extracted_slots"])
    state["history"].append({"speaker_role": "USER", "utterance_text": user_msg})

    completion = _calc_effective_completion(state["slots"])
    debug = (
        f"source: {result['source']} | completion: {completion}% | turn: {state['turn']}/{MAX_USER_TURNS}\n"
        f"extracted: {json.dumps(result['extracted_slots'], ensure_ascii=False)}\n"
        f"should_stop={result['should_stop']} reason={result['stop_reason']!r}"
    )

    proceed, reason = should_move_to_recommendation(
        merged_slots=state["slots"],
        user_turn_count=state["turn"],
        user_msg=user_msg,
        llm_should_stop=result["should_stop"],
        llm_stop_reason=result["stop_reason"],
    )

    if proceed:
        state["done"] = True
        final_slots = json.dumps(state["slots"], ensure_ascii=False, indent=2)
        return jsonify({
            "message": f"[TERMINATE] reason={reason} → 추천 단계로 이동",
            "debug": debug,
            "done": True,
            "slots": final_slots,
            "completion": completion,
        })

    next_q = result["next_question"]
    state["history"].append({"speaker_role": "LLM", "utterance_text": next_q})
    return jsonify({"message": next_q, "debug": debug, "done": False})


if __name__ == "__main__":
    print("http://localhost:5000 에서 실행 중")
    app.run(debug=True, port=5000)
