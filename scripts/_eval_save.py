"""개별 eval 스크립트의 결과 자동 저장 헬퍼.

저장 경로: eval_results/summary/{kind}/{model}_{kind}_{tag}_{stamp}.{json,txt}
  model 예: gemma / qwen / exaone (env LLM_MODEL 에서 자동 추출)
  kind  예: feedback / slots / rec / e2e / all
  tag   예: n500 / smoke20 / oracle_smoke20 / rerank_aroma_v1

eval_all_llm.py 의 통합 저장도 같은 규칙(kind=all).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


QUANT_ROOT = Path("eval_results/summary")


def short_model_name(model: str) -> str:
    """LLM_MODEL 전체 경로에서 짧은 식별자 추출.

    예: 'google/gemma-2-9b-it' → 'gemma'
        'Qwen/Qwen3-8B' → 'qwen'
        'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct' → 'exaone'
    """
    m = (model or "").lower()
    for key in ("gemma", "qwen", "exaone"):
        if key in m:
            return key
    # fallback: 경로 끝 토큰의 첫 단어
    tail = m.split("/")[-1]
    return tail.split("-")[0] or "model"


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
        kind: 'feedback' | 'slots' | 'rec' | 'e2e' | 'all'
        tag: 실험 라벨 (e.g. n500, smoke20, oracle_smoke20)
        limit: 평가 case 수
        payload: json 본문 (eval 함수 반환 dict)
        summary_lines: 사람이 읽을 txt 본문 (print 출력과 동일 포맷이면 재활용)
        model: 명시 안 하면 env LLM_MODEL 로 자동 기록
    """
    out_dir = QUANT_ROOT / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    model_name = model or os.getenv("LLM_MODEL", "")
    model_short = short_model_name(model_name)
    base = f"{model_short}_{kind}_{tag}_{stamp}"

    full_payload = {
        "kind": kind,
        "tag": tag,
        "model": model_name,
        "timestamp": stamp,
        "limit": limit,
        **payload,
    }

    json_path = out_dir / f"{base}.json"
    txt_path = out_dir / f"{base}.txt"

    json_path.write_text(json.dumps(full_payload, ensure_ascii=False, indent=2, default=str))

    header = [
        "=" * 60,
        f" {kind.upper()} EVAL — model={model_name}  tag={tag}  {stamp}",
        f" limit={limit if limit is not None else 'full'}",
        "=" * 60,
    ]
    txt_path.write_text("\n".join(header + summary_lines) + "\n")

    return json_path, txt_path
