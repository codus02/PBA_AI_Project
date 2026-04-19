# scripts/eval_feedback.py
import sys, json
sys.path.insert(0, ".")

import pandas as pd
from app.agents.preference_agent import classify_intent

def eval_feedback():
    df = pd.read_csv("data/eval/feedback_eval_v2_500.csv")

    correct = 0
    errors  = []

    for _, row in df.iterrows():
        pred = classify_intent(row["feedback_text"])
        gold = row["gold_intent"]

        if pred == gold:
            correct += 1
        else:
            errors.append({
                "text":     row["feedback_text"],
                "expected": gold,
                "got":      pred,
            })

    total = len(df)
    acc   = correct / total * 100

    # 클래스별 정확도
    for intent in ["ACCEPT", "ADJUST", "REJECT"]:
        sub = df[df["gold_intent"] == intent]
        sub_correct = sum(
            classify_intent(r["feedback_text"]) == intent
            for _, r in sub.iterrows()
        )
        print(f"  {intent}: {sub_correct}/{len(sub)} = {sub_correct/len(sub)*100:.1f}%")

    print(f"\n[피드백 분류] 전체 정확도: {correct}/{total} = {acc:.1f}%")

    # 오답 샘플 5개 출력
    print("\n오답 샘플 (최대 5개):")
    for e in errors[:5]:
        print(f"  '{e['text']}' → 예측:{e['got']} / 정답:{e['expected']}")

    return acc

if __name__ == "__main__":
    eval_feedback()