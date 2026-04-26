import os
import shutil
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.crud import (
    create_party_session,
    create_guest_session,
    get_party_session,
    get_guest_session,
    upsert_initial_tags,
    save_space_analysis,
    create_dialogue_turn,
    list_dialogue_turns,
    get_preference_slot,
    update_preference_slots,
    get_final_recommendation_by_id,
    get_initial_tag_response,
    create_evaluation_log,
    update_guest_stage,
    update_feedback_round,
    count_user_turns,
)
from app.utils.config import UPLOAD_DIR
from app.agents.preference_agent import (
    analyze_user_turn,
    generate_opening_question,
    merge_slots,
    should_move_to_recommendation,
    transition_opener,
    proceed_requires_force,
    _calc_effective_completion,
    _seed_slots_from_initial_tags,
    _sanitize_user_text,
)
from app.agents.orchestration_agent import run_recommendation, process_feedback, finalize_sample
from app.agents.output_agent import (
    generate_output_json,
    generate_motor_recipe,
    DEFAULT_SAMPLE_VOLUME_ML,
    DEFAULT_FINAL_VOLUME_ML,
)
from app.agents.mood_agent import analyze_space_image, img2tag


router = APIRouter()


# ============================================================
# Pydantic Schemas
# ============================================================

class PartySessionCreateRequest(BaseModel):
    session_name: Optional[str] = None
    demo_mode: bool = False


class GuestSessionCreateRequest(BaseModel):
    party_session_id: str
    guest_label: str = "게스트1"


class InitialTagRequest(BaseModel):
    familiarity: Optional[str] = Field(default=None, description="칵테일 친숙도")
    tastes: List[str] = Field(default_factory=list, description="초기 선호 맛 태그")
    strength: str = Field(..., description="초기 선호 도수/강도감 태그")
    aromas: List[str] = Field(default_factory=list, description="초기 선호 향 태그")


class DialogueRequest(BaseModel):
    message: str


class FeedbackRequest(BaseModel):
    sample_recommendation_id: str
    feedback_text: str

class EvaluationRequest(BaseModel):
    final_recommendation_id: str
    satisfaction_score: float = Field(..., ge=1.0, le=5.0, description="1~5점")
    would_reorder: bool
    review_text: Optional[str] = None


# ============================================================
# Helpers
# ============================================================

def _serialize_dialogue_turn(turn) -> dict:
    return {
        "turn_id": str(turn.turn_id),
        "guest_session_id": str(turn.guest_session_id),
        "turn_index": turn.turn_index,
        "speaker_role": turn.speaker_role,
        "utterance_text": turn.utterance_text,
        "extracted_slots_json": turn.extracted_slots_json,
        "created_at": turn.created_at.isoformat() if turn.created_at else None,
    }


def _slot_row_to_internal_dict(slot_row) -> dict:
    if slot_row is None:
        return {}

    return {
        "current_mood": slot_row.current_mood,
        "party_purpose": slot_row.party_purpose,
        "taste_profile": slot_row.taste_profile_json or {},
        "aroma_profile": slot_row.aroma_profile_json or {},
        "strength_preference": slot_row.strength_preference,
        "disliked_bases": slot_row.disliked_bases_json or [],
        "favorite_drinks": slot_row.favorite_drinks_json or [],
    }


def _serialize_preference_slot(slot_row) -> dict:
    return {
        "slot_profile_id": str(slot_row.slot_profile_id),
        "guest_session_id": str(slot_row.guest_session_id),
        "current_mood": slot_row.current_mood,
        "party_purpose": slot_row.party_purpose,
        "taste_profile_json": slot_row.taste_profile_json,
        "aroma_profile_json": slot_row.aroma_profile_json,
        "strength_preference": slot_row.strength_preference,
        "disliked_bases_json": slot_row.disliked_bases_json,
        "favorite_drinks_json": slot_row.favorite_drinks_json,
        "slot_completion_score": float(slot_row.slot_completion_score),
        "updated_at": slot_row.updated_at.isoformat() if slot_row.updated_at else None,
    }


# ============================================================
# Basic
# ============================================================

@router.get("/health")
def healthcheck():
    return {"status": "ok"}


# ============================================================
# 추천
# ============================================================

@router.post("/sessions/{gid}/recommend-sample")
def recommend_sample_endpoint(
    gid: str,
    force: bool = False,
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    # 대화 단계에서 사용자 STOP / 턴 상한으로 넘어온 경우 stage 가 READY_TO_RECOMMEND 로 박혀 있음.
    # completion<80 이어도 이 경로는 이미 통과 의사가 확인된 것이니 force 로 강행한다.
    effective_force = force or (guest.conversation_stage == "READY_TO_RECOMMEND")
    result = run_recommendation(db, gid, k=3, force=effective_force)

    if result.get("status") == "ok":
        update_guest_stage(db, gid, "FEEDBACK_LOOP")
        top = (result.get("top_k") or [{}])[0]
        name = top.get("name_kr", "이 칵테일")
        question = f"추천드린 '{name}' 어떠세요? 드셔보시고 느낌 알려주세요."
        llm_turn = create_dialogue_turn(
            db=db,
            guest_session_id=gid,
            speaker_role="LLM",
            utterance_text=question,
            extracted_slots_json=None,
        )
        result["llm_question"] = question
        result["llm_turn"] = _serialize_dialogue_turn(llm_turn)

        # 시음용 샘플 제조 명령 (Pi 펌프). 칵테일 원본 비율 그대로, 30ml.
        cocktail_id = top.get("cocktail_id")
        if cocktail_id is not None:
            try:
                result["sample_motor_recipe"] = generate_motor_recipe(
                    db=db,
                    cocktail_id=int(cocktail_id),
                    volume_ml=DEFAULT_SAMPLE_VOLUME_ML,
                    feedback_deltas=None,
                )
            except Exception as exc:
                result["sample_motor_recipe"] = None
                result.setdefault("warnings", []).append(
                    f"sample_motor_recipe 생성 실패: {exc}"
                )

    return result

  


@router.post("/sessions/{gid}/feedback")
def feedback_endpoint(
    gid: str,
    req: FeedbackRequest,
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    sanitized_feedback = _sanitize_user_text(req.feedback_text)
    current_round = (guest.feedback_round or 0) + 1

    # 4회차부터는 추가 피드백을 받지 않고 현재 sample을 강제 확정한다.
    if current_round > 3:
        try:
            result = finalize_sample(
                db=db,
                guest_session_id=gid,
                sample_recommendation_id=req.sample_recommendation_id,
                forced=True,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return result

    try:
        result = process_feedback(
            db=db,
            guest_session_id=gid,
            sample_recommendation_id=req.sample_recommendation_id,
            feedback_text=sanitized_feedback,
            feedback_round=current_round,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result.get("status") != "already_finalized":
        user_turn = create_dialogue_turn(
            db=db,
            guest_session_id=gid,
            speaker_role="USER",
            utterance_text=sanitized_feedback,
            extracted_slots_json=None,
        )
        update_feedback_round(db, gid, current_round)
        result["user_turn"] = _serialize_dialogue_turn(user_turn)

    # ADJUST → 같은 칵테일 + 레시피 조정 (sample_motor_recipe 는 이미 result 에 포함됨)
    if result.get("status") == "adjusted":
        name = result.get("sample_cocktail_name") or "이 칵테일"
        question = f"피드백 반영해서 '{name}' 다시 따라드릴게요. 어떠세요?"
        llm_turn = create_dialogue_turn(
            db=db,
            guest_session_id=gid,
            speaker_role="LLM",
            utterance_text=question,
            extracted_slots_json=None,
        )
        result["llm_question"] = question
        result["llm_turn"] = _serialize_dialogue_turn(llm_turn)

    # REJECT → 새 칵테일 추천 (시음한 칵테일 제외)
    if result.get("status") == "re_recommended":
        top = (result.get("top_k") or [{}])[0]
        name = top.get("name_kr", "이 칵테일")
        question = f"그럼 이번엔 '{name}' 어떠세요?"
        llm_turn = create_dialogue_turn(
            db=db,
            guest_session_id=gid,
            speaker_role="LLM",
            utterance_text=question,
            extracted_slots_json=None,
        )
        result["llm_question"] = question
        result["llm_turn"] = _serialize_dialogue_turn(llm_turn)

        cocktail_id = top.get("cocktail_id")
        if cocktail_id is not None:
            try:
                result["sample_motor_recipe"] = generate_motor_recipe(
                    db=db,
                    cocktail_id=int(cocktail_id),
                    volume_ml=DEFAULT_SAMPLE_VOLUME_ML,
                    feedback_deltas=None,
                )
            except Exception as exc:
                result["sample_motor_recipe"] = None
                result.setdefault("warnings", []).append(
                    f"sample_motor_recipe 생성 실패: {exc}"
                )

    return result

# ============================================================
# 1. Party / Guest Session
# ============================================================

@router.post("/sessions/party")
def create_party(req: PartySessionCreateRequest, db: Session = Depends(get_db)):
    row = create_party_session(
        db=db,
        session_name=req.session_name,
        demo_mode=req.demo_mode,
    )

    return {
        "party_session_id": str(row.party_session_id),
        "session_name": row.session_name,
        "session_status": row.session_status,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "demo_mode": row.demo_mode,
    }


@router.post("/sessions/guest")
def create_guest(req: GuestSessionCreateRequest, db: Session = Depends(get_db)):
    party = get_party_session(db, req.party_session_id)
    if not party:
        raise HTTPException(status_code=404, detail="party_session_id not found")

    row = create_guest_session(
        db=db,
        party_session_id=req.party_session_id,
        guest_label=req.guest_label,
    )

    return {
        "guest_session_id": str(row.guest_session_id),
        "party_session_id": str(row.party_session_id),
        "guest_label": row.guest_label,
        "session_status": row.session_status,
        "started_at": row.started_at.isoformat() if row.started_at else None,
    }


# ============================================================
# 2. Initial Tags
# ============================================================

@router.post("/sessions/{gid}/tags")
def save_initial_tags_endpoint(
    gid: str,
    req: InitialTagRequest,
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    row = upsert_initial_tags(
        db=db,
        guest_session_id=gid,
        familiarity_tag=req.familiarity,
        taste_tags_json=req.tastes,
        strength_tag=req.strength,
        aroma_tags_json=req.aromas,
    )

    return {
        "status": "ok",
        "tag_response_id": str(row.tag_response_id),
        "guest_session_id": str(row.guest_session_id),
        "familiarity_tag": row.familiarity_tag,
        "taste_tags_json": row.taste_tags_json,
        "strength_tag": row.strength_tag,
        "aroma_tags_json": row.aroma_tags_json,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


# ============================================================
# 3. Space Image Upload
# ============================================================

@router.post("/sessions/{gid}/space-image")
async def upload_space_image(
    gid: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    filename = f"{gid}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # 업로드 폴더 없으면 생성
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    mood_result = analyze_space_image(file_path)

    analysis = save_space_analysis(
        db=db,
        party_session_id=guest.party_session_id,
        image_path=file_path,
        mood_tags_json=mood_result["mood_tags_json"],
    )

    return {
        "status": "ok",
        "space_analysis": {
            "space_analysis_id": str(analysis.space_analysis_id),
            "party_session_id": str(analysis.party_session_id),
            "guest_session_id": gid,
            "image_path": analysis.image_path,
            "caption_en": mood_result["caption_en"],
            "best_mood_tag": mood_result["best_mood_tag"],
            "mood_tags_json": analysis.mood_tags_json,
            "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
        },
    }


@router.post("/space/img2tag")
async def img2tag_endpoint(file: UploadFile = File(...)):
    """공간 이미지 → mood_tag 3-tuple. 세션 없이 동작하는 경량 엔드포인트.

    내부적으로 Gemini API 키 풀 (gemini_key_pool) 에서 분당 5회/일간 20회 한도
    안에 있는 키를 자동 선택. 모든 키 소진 시 429 반환.

    응답: ``{"mood_tag": ["lively", "bright", "spacious"]}``
    """
    from app.services.gemini_key_pool import QuotaExhaustedError

    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"이미지 읽기 실패: {exc}")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="빈 이미지 파일입니다.")

    try:
        tags = img2tag(image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"참조 파일 누락: {exc}")
    except QuotaExhaustedError as exc:
        # 모든 Gemini 키가 분당/일간 한도 초과 → 잠시 후 재시도 가능.
        raise HTTPException(
            status_code=429,
            detail=f"Gemini quota 소진 — 잠시 후 다시 시도해주세요. ({exc})",
        )
    except RuntimeError as exc:
        # 키 미설정 등 server config 오류.
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini 호출 실패: {exc}")

    return {"mood_tag": list(tags)}


# ============================================================
# 4. Dialogue
# ============================================================

@router.post("/sessions/{gid}/start-dialogue")
def start_dialogue_endpoint(
    gid: str,
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    # 오프닝은 LLM 호출 없이 고정 문구 사용 (첫 턴은 정보가 0이라 생성 의미가 적음)
    nickname = (guest.guest_label or "").strip() or None
    question = generate_opening_question(nickname=nickname)

    llm_turn = create_dialogue_turn(
        db=db,
        guest_session_id=gid,
        speaker_role="LLM",
        utterance_text=question,
        extracted_slots_json=None,
    )

    update_guest_stage(db, gid, "COLLECTING")

    return {
        "status": "ok",
        "stage": "COLLECTING",
        "question": question,
        "llm_turn": _serialize_dialogue_turn(llm_turn),
    }

@router.post("/sessions/{gid}/dialogue")
def dialogue_endpoint(
    gid: str,
    req: DialogueRequest,
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    # 1) 사용자 턴 저장
    sanitized_message = _sanitize_user_text(req.message)

    user_turn = create_dialogue_turn(
        db=db,
        guest_session_id=gid,
        speaker_role="USER",
        utterance_text=sanitized_message,
        extracted_slots_json=None,
    )

    # 2) 현재 슬롯 로드
    slot_row = get_preference_slot(db, gid)
    current_slots = _slot_row_to_internal_dict(slot_row)

    tag_row = get_initial_tag_response(db, gid)
    seeded_slots = _seed_slots_from_initial_tags(tag_row)
    current_slots = merge_slots(seeded_slots, current_slots)

    # 3) 대화 히스토리 구성
    history_rows = list_dialogue_turns(db, gid)
    history = [
        {
            "speaker_role": row.speaker_role,
            "utterance_text": row.utterance_text,
            "extracted_slots_json": row.extracted_slots_json,
        }
        for row in history_rows
    ]

    # 4) 단일 LLM 호출 — 추출 + 종료 + 다음 질문
    user_turn_count = count_user_turns(db, gid)
    agent_result = analyze_user_turn(
        history=history,
        slots=current_slots,
        user_msg=sanitized_message,
        familiarity=getattr(tag_row, "familiarity_tag", None),
        user_turn_count=user_turn_count,
    )
    extracted_slots = agent_result["extracted_slots"]

    # 5) 추출된 슬롯 DB 반영 (merge_slots로 병합 후 저장)
    merged_slots = merge_slots(current_slots, extracted_slots)
    updated_slot_row = update_preference_slots(
        db=db,
        guest_session_id=gid,
        extracted_slots=merged_slots,
    )

    # 6) 추천 이동 판단
    user_turn_count = count_user_turns(db, gid)
    should_proceed, proceed_reason = should_move_to_recommendation(
        merged_slots=merged_slots,
        user_turn_count=user_turn_count,
        user_msg=sanitized_message,
        llm_should_stop=agent_result["should_stop"],
        llm_stop_reason=agent_result["stop_reason"],
    )

    if should_proceed:
        update_guest_stage(db, gid, "READY_TO_RECOMMEND")
        transition_msg = transition_opener(proceed_reason)
        transition_turn = create_dialogue_turn(
            db=db,
            guest_session_id=gid,
            speaker_role="LLM",
            utterance_text=transition_msg,
            extracted_slots_json=None,
        )
        return {
            "status": "proceed_to_recommendation",
            "guest_session_id": gid,
            "reason": proceed_reason,
            "transition_message": transition_msg,
            "transition_turn": _serialize_dialogue_turn(transition_turn),
            "force_required": proceed_requires_force(proceed_reason),
            "slot_state": _serialize_preference_slot(updated_slot_row),
            "completion": float(updated_slot_row.slot_completion_score),
            "should_proceed": True,
        }

    # 7) LLM이 생성한 다음 질문 저장
    question = (agent_result["next_question"] or "").strip() or "좋아요. 흐름을 이어가게 한 가지만 더 여쭤볼게요."
    llm_turn = create_dialogue_turn(
        db=db,
        guest_session_id=gid,
        speaker_role="LLM",
        utterance_text=question,
        extracted_slots_json=extracted_slots,
    )
    update_guest_stage(db, gid, "COLLECTING")

    return {
        "status": "ok",
        "guest_session_id": gid,
        "user_turn": _serialize_dialogue_turn(user_turn),
        "llm_turn": _serialize_dialogue_turn(llm_turn),
        "question": question,
        "extracted_slots": extracted_slots,
        "slot_state": _serialize_preference_slot(updated_slot_row) if updated_slot_row else {},
        "completion": _calc_effective_completion(merged_slots),
        "should_proceed": False,
    }

@router.get("/final-output/{final_recommendation_id}")
def final_output_endpoint(
    final_recommendation_id: str,
    db: Session = Depends(get_db),
):
    final_row = get_final_recommendation_by_id(db, final_recommendation_id)
    if not final_row:
        raise HTTPException(status_code=404, detail="final_recommendation_id not found")

    return generate_output_json(
        db=db,
        cocktail_id=final_row.final_cocktail_id,
        volume_ml=90,
    )

@router.post("/sessions/{gid}/evaluation")
def save_evaluation(
    gid: str,
    req: EvaluationRequest,
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    # final_recommendation 존재 확인
    final = get_final_recommendation_by_id(db, req.final_recommendation_id)
    if not final:
        raise HTTPException(status_code=404, detail="final_recommendation not found")

    if str(final.guest_session_id) != str(gid):
        raise HTTPException(status_code=403, detail="이 세션의 추천이 아닙니다")

    row = create_evaluation_log(
        db=db,
        guest_session_id=gid,
        final_satisfaction_score=req.satisfaction_score,
        would_reorder=req.would_reorder,
        review_text=req.review_text,
    )

    update_guest_stage(db, gid, "FINISHED")

    return {
        "status": "ok",
        "evaluation_id": str(row.evaluation_id),
        "satisfaction_score": float(row.final_satisfaction_score),
        "would_reorder": row.would_reorder,
        "stage": "FINISHED",
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
