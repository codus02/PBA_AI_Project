# scripts/eval_all.py
import sys
sys.path.insert(0, ".")

from scripts.eval_feedback       import eval_feedback
from scripts.eval_slots          import eval_slots
from scripts.eval_recommendation import eval_recommendation
from datetime import datetime

if __name__ == "__main__":
    print("=" * 50)
    print(f" 룰베이스 Baseline 평가  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)

    feedback_acc = eval_feedback()
    slot_acc     = eval_slots()
    rec_result   = eval_recommendation()

    print("\n" + "=" * 50)
    print(" 최종 요약 (baseline 기록용)")
    print("=" * 50)
    print(f"  피드백 분류 정확도 : {feedback_acc:.1f}%")
    print(f"  슬롯 추출 평균     : {slot_acc:.1f}%")
    print(f"  추천 Hit@1         : {rec_result['hit@1']*100:.1f}%")
    print(f"  추천 Hit@3         : {rec_result['hit@3']*100:.1f}%")
    print(f"  카테고리 Hit@3     : {rec_result['cat_hit@3']*100:.1f}%")
    print("\n→ 이 수치를 baseline으로 저장해두고 Qwen 붙인 후 비교")