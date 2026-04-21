"""개별 eval 스크립트의 결과 자동 저장 헬퍼.

저장 경로: eval_results/quantitative/{kind}_{tag}_{stamp}.{json,txt}
  kind 예: feedback / slots / rec
  tag  예: exaone / qwen / qwen_current

eval_all_llm.py 의 통합 저장 포맷({tag}_{stamp})과 구분되도록 kind prefix 사용.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


OUT_DIR = Path("eval_results/quantitative")


def save_eval_result(
    kind: str,
    tag: str,
    limit: Optional[int],
    payload: dict,
    summary_lines: list[str],
    model: Optional[str] = None,
) -> tuple[Path, Path]:
    """eval 결과를 json + txt 로 저장.

    Args:
        kind: 'feedback' | 'slots' | 'rec' 등
        tag: 사용자 지정 라벨 (e.g. exaone, qwen_current)
        limit: 평가 case 수
        payload: json 본문 (eval 함수 반환 dict)
        summary_lines: 사람이 읽을 txt 본문 (print 출력과 동일 포맷이면 재활용)
        model: 명시 안 하면 env LLM_MODEL 로 자동 기록
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    base = f"{kind}_{tag}_{stamp}"

    model_name = model or os.getenv("LLM_MODEL", "")

    full_payload = {
        "kind": kind,
        "tag": tag,
        "model": model_name,
        "timestamp": stamp,
        "limit": limit,
        **payload,
    }

    json_path = OUT_DIR / f"{base}.json"
    txt_path = OUT_DIR / f"{base}.txt"

    json_path.write_text(json.dumps(full_payload, ensure_ascii=False, indent=2, default=str))

    header = [
        "=" * 60,
        f" {kind.upper()} EVAL — tag={tag}  model={model_name}  {stamp}",
        f" limit={limit if limit is not None else 'full'}",
        "=" * 60,
    ]
    txt_path.write_text("\n".join(header + summary_lines) + "\n")

    return json_path, txt_path
