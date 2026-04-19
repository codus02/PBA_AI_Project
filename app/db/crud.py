from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import (
    PartySession,
    PartySpaceAnalysis,
    GuestSession,
    InitialTagResponse,
    PreferenceSlot,
    PreferenceVector,
    DialogueTurn,
    SampleRecommendation,
    SampleFeedback,
    FinalRecommendation,
    BrewOrder,
    EvaluationLog,
)


# ============================================================
# 공통 유틸
# ============================================================

def _to_decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _clamp(value: float, min_value: float, max_value: float) -> Decimal:
    value = max(min_value, min(max_value, float(value)))
    return Decimal(str(value))


# ============================================================
# 공통 조회
# ============================================================

def get_party_session(db: Session, party_session_id: str | Any) -> Optional[PartySession]:
    return (
        db.query(PartySession)
        .filter(PartySession.party_session_id == party_session_id)
        .first()
    )


def get_guest_session(db: Session, guest_session_id: str | Any) -> Optional[GuestSession]:
    return (
        db.query(GuestSession)
        .filter(GuestSession.guest_session_id == guest_session_id)
        .first()
    )


def get_initial_tag_response(
    db: Session, guest_session_id: str | Any
) -> Optional[InitialTagResponse]:
    return (
        db.query(InitialTagResponse)
        .filter(InitialTagResponse.guest_session_id == guest_session_id)
        .first()
    )


def get_preference_slot(
    db: Session, guest_session_id: str | Any
) -> Optional[PreferenceSlot]:
    return (
        db.query(PreferenceSlot)
        .filter(PreferenceSlot.guest_session_id == guest_session_id)
        .first()
    )


def get_preference_vector(
    db: Session, guest_session_id: str | Any
) -> Optional[PreferenceVector]:
    return (
        db.query(PreferenceVector)
        .filter(PreferenceVector.guest_session_id == guest_session_id)
        .first()
    )


def get_latest_space_analysis_by_party(
    db: Session,
    party_session_id: str | Any,
) -> Optional[PartySpaceAnalysis]:
    return (
        db.query(PartySpaceAnalysis)
        .filter(PartySpaceAnalysis.party_session_id == party_session_id)
        .order_by(PartySpaceAnalysis.created_at.desc())
        .first()
    )


# ============================================================
# 1. 세션 생성 / 종료
# ============================================================

def create_party_session(
    db: Session,
    session_name: Optional[str] = None,
    demo_mode: bool = False,
) -> PartySession:
    row = PartySession(
        session_name=session_name,
        session_status="ACTIVE",
        demo_mode=demo_mode,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def end_party_session(
    db: Session,
    party_session_id: str | Any,
) -> Optional[PartySession]:
    row = get_party_session(db, party_session_id)
    if not row:
        return None

    row.session_status = "ENDED"
    row.ended_at = datetime.utcnow()

    db.commit()
    db.refresh(row)
    return row

def update_guest_stage(
    db: Session,
    guest_session_id: str | Any,
    conversation_stage: str,
) -> Optional[GuestSession]:
    row = get_guest_session(db, guest_session_id)
    if not row:
        return None

    row.conversation_stage = conversation_stage
    db.commit()
    db.refresh(row)
    return row


def update_feedback_round(
    db: Session,
    guest_session_id: str | Any,
    feedback_round: int,
) -> Optional[GuestSession]:
    row = get_guest_session(db, guest_session_id)
    if not row:
        return None

    row.feedback_round = feedback_round
    db.commit()
    db.refresh(row)
    return row


def create_guest_session(
    db: Session,
    party_session_id: str | Any,
    guest_label: str,
) -> GuestSession:
    row = GuestSession(
        party_session_id=party_session_id,
        guest_label=guest_label,
        session_status="IN_PROGRESS",
    )
    db.add(row)
    db.flush()  # guest_session_id 확보

    slot_row = PreferenceSlot(
        guest_session_id=row.guest_session_id,
        slot_completion_score=Decimal("0.00"),
    )
    db.add(slot_row)

    vector_row = PreferenceVector(
        guest_session_id=row.guest_session_id,
        sweetness_score=Decimal("3.0"),
        bitterness_score=Decimal("3.0"),
        sourness_score=Decimal("3.0"),
        freshness_score=Decimal("3.0"),
        body_score=Decimal("3.0"),
        herbal_score=Decimal("2.0"),
        citrus_score=Decimal("2.0"),
        alcohol_score=Decimal("3.0"),
        version=1,
    )
    db.add(vector_row)

    db.commit()
    db.refresh(row)
    return row

def count_user_turns(
    db: Session,
    guest_session_id: str | Any,
) -> int:
    return (
        db.query(DialogueTurn)
        .filter(
            DialogueTurn.guest_session_id == guest_session_id,
            DialogueTurn.speaker_role == "USER",
        )
        .count()
    )

def complete_guest_session(
    db: Session,
    guest_session_id: str | Any,
) -> Optional[GuestSession]:
    row = get_guest_session(db, guest_session_id)
    if not row:
        return None

    row.session_status = "COMPLETED"
    row.ended_at = datetime.utcnow()

    db.commit()
    db.refresh(row)
    return row


# ============================================================
# 2. 공간 이미지 분석 결과 저장
# ============================================================

def save_space_analysis(
    db: Session,
    party_session_id: str | Any,
    image_path: str,
    mood_tags_json: dict,
    mood_weight: float = 0.30,
) -> PartySpaceAnalysis:
    row = PartySpaceAnalysis(
        party_session_id=party_session_id,
        image_path=image_path,
        mood_tags_json=mood_tags_json,
        mood_weight=_to_decimal(mood_weight),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ============================================================
# 3. 초기 태그 저장
# ============================================================

def upsert_initial_tags(
    db: Session,
    guest_session_id: str | Any,
    familiarity_tag: Optional[str],
    taste_tags_json: list[str],
    strength_tag: str,
    aroma_tags_json: list[str],
) -> InitialTagResponse:
    row = get_initial_tag_response(db, guest_session_id)

    if row is None:
        row = InitialTagResponse(
            guest_session_id=guest_session_id,
            familiarity_tag=familiarity_tag,
            taste_tags_json=taste_tags_json,
            strength_tag=strength_tag,
            aroma_tags_json=aroma_tags_json,
        )
        db.add(row)
    else:
        row.familiarity_tag = familiarity_tag
        row.taste_tags_json = taste_tags_json
        row.strength_tag = strength_tag
        row.aroma_tags_json = aroma_tags_json

    db.commit()
    db.refresh(row)
    return row


# ============================================================
# 4. preference_slots / preference_vectors 보장 생성
# ============================================================

def _ensure_preference_slot(
    db: Session,
    guest_session_id: str | Any,
) -> PreferenceSlot:
    row = get_preference_slot(db, guest_session_id)
    if row is None:
        row = PreferenceSlot(
            guest_session_id=guest_session_id,
            slot_completion_score=Decimal("0.00"),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def _ensure_preference_vector(
    db: Session,
    guest_session_id: str | Any,
) -> PreferenceVector:
    row = get_preference_vector(db, guest_session_id)
    if row is None:
        row = PreferenceVector(
            guest_session_id=guest_session_id,
            sweetness_score=Decimal("3.0"),
            bitterness_score=Decimal("3.0"),
            sourness_score=Decimal("3.0"),
            freshness_score=Decimal("3.0"),
            body_score=Decimal("3.0"),
            herbal_score=Decimal("2.0"),
            citrus_score=Decimal("2.0"),
            alcohol_score=Decimal("3.0"),
            version=1,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


# ============================================================
# 5. 슬롯 업데이트
# ============================================================

def calculate_slot_completion(slots: PreferenceSlot) -> Decimal:
    """
    슬롯 완료도 계산:
    아래 7개 중 채워진 개수 / 7 * 100
    """
    fields = [
        slots.party_purpose,
        slots.current_mood,
        slots.taste_profile_json,
        slots.aroma_profile_json,
        slots.strength_preference,
        slots.disliked_bases_json,
        slots.favorite_drinks_json,
    ]

    filled_count = 0
    for value in fields:
        if value is None:
            continue
        if isinstance(value, list) and len(value) == 0:
            continue
        if isinstance(value, dict) and len(value) == 0:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        filled_count += 1

    score = round((filled_count / 7) * 100, 2)
    return Decimal(str(score))


def update_preference_slots(
    db: Session,
    guest_session_id: str | Any,
    extracted_slots: dict[str, Any],
) -> PreferenceSlot:
    row = _ensure_preference_slot(db, guest_session_id)

    slot_to_column = {
        "party_purpose": "party_purpose",
        "current_mood": "current_mood",
        "taste_profile": "taste_profile_json",
        "aroma_profile": "aroma_profile_json",
        "strength_preference": "strength_preference",
        "disliked_bases": "disliked_bases_json",
        "favorite_drinks": "favorite_drinks_json",
    }

    for slot_key, value in extracted_slots.items():
        column_name = slot_to_column.get(slot_key)
        if not column_name:
            continue
        setattr(row, column_name, value)

    row.slot_completion_score = calculate_slot_completion(row)

    db.commit()
    db.refresh(row)
    return row


# ============================================================
# 6. 대화 로그 저장
# ============================================================

def get_next_turn_index(
    db: Session,
    guest_session_id: str | Any,
) -> int:
    current_max = (
        db.query(func.max(DialogueTurn.turn_index))
        .filter(DialogueTurn.guest_session_id == guest_session_id)
        .scalar()
    )
    return 1 if current_max is None else current_max + 1


def create_dialogue_turn(
    db: Session,
    guest_session_id: str | Any,
    speaker_role: str,
    utterance_text: str,
    extracted_slots_json: Optional[dict] = None,
) -> DialogueTurn:
    turn_index = get_next_turn_index(db, guest_session_id)

    row = DialogueTurn(
        guest_session_id=guest_session_id,
        turn_index=turn_index,
        speaker_role=speaker_role,
        utterance_text=utterance_text,
        extracted_slots_json=extracted_slots_json,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_dialogue_turns(
    db: Session,
    guest_session_id: str | Any,
) -> list[DialogueTurn]:
    return (
        db.query(DialogueTurn)
        .filter(DialogueTurn.guest_session_id == guest_session_id)
        .order_by(DialogueTurn.turn_index.asc())
        .all()
    )


# ============================================================
# 7. 벡터 조회 / 갱신
# ============================================================

def update_preference_vector(
    db: Session,
    guest_session_id: str | Any,
    updates: dict[str, Any],
    increment_version: bool = True,
) -> PreferenceVector:
    row = _ensure_preference_vector(db, guest_session_id)

    ranges = {
        "sweetness_score": (1.0, 5.0),
        "bitterness_score": (1.0, 5.0),
        "sourness_score": (1.0, 5.0),
        "freshness_score": (1.0, 5.0),
        "body_score": (1.0, 5.0),
        "herbal_score": (0.0, 3.0),
        "citrus_score": (0.0, 3.0),
        "alcohol_score": (0.0, 5.0),
    }

    for key, value in updates.items():
        if key not in ranges:
            continue
        min_v, max_v = ranges[key]
        setattr(row, key, _clamp(value, min_v, max_v))

    if increment_version:
        row.version += 1

    db.commit()
    db.refresh(row)
    return row


# ============================================================
# 8. 1차 추천 저장
# ============================================================

def create_sample_recommendation(
    db: Session,
    guest_session_id: str | Any,
    space_analysis_id: str | Any,
    recommended_cocktail_id: int,
    recommendation_reason: str,
    rag_retrieved_ids_json: list[int],
    recipe_snapshot_json: dict,
) -> SampleRecommendation:
    row = SampleRecommendation(
        guest_session_id=guest_session_id,
        space_analysis_id=space_analysis_id,
        recommended_cocktail_id=recommended_cocktail_id,
        recommendation_reason=recommendation_reason,
        rag_retrieved_ids_json=rag_retrieved_ids_json,
        recipe_snapshot_json=recipe_snapshot_json,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_sample_recommendation(
    db: Session,
    sample_recommendation_id: str | Any,
) -> Optional[SampleRecommendation]:
    return (
        db.query(SampleRecommendation)
        .filter(SampleRecommendation.sample_recommendation_id == sample_recommendation_id)
        .first()
    )


def get_latest_sample_recommendation_by_guest(
    db: Session,
    guest_session_id: str | Any,
) -> Optional[SampleRecommendation]:
    return (
        db.query(SampleRecommendation)
        .filter(SampleRecommendation.guest_session_id == guest_session_id)
        .order_by(SampleRecommendation.created_at.desc())
        .first()
    )


# ============================================================
# 9. 피드백 저장
# ============================================================

def create_sample_feedback(
    db: Session,
    sample_recommendation_id: str | Any,
    feedback_text: str,
    feedback_intent: str,
    sweetness_delta: Optional[float] = None,
    sourness_delta: Optional[float] = None,
    bitterness_delta: Optional[float] = None,
    body_delta: Optional[float] = None,
    freshness_delta: Optional[float] = None,
    aroma_delta_json: Optional[dict] = None,
    parsed_summary: Optional[str] = None,
) -> SampleFeedback:
    row = SampleFeedback(
        sample_recommendation_id=sample_recommendation_id,
        feedback_text=feedback_text,
        feedback_intent=feedback_intent,
        sweetness_delta=_to_decimal(sweetness_delta) if sweetness_delta is not None else None,
        sourness_delta=_to_decimal(sourness_delta) if sourness_delta is not None else None,
        bitterness_delta=_to_decimal(bitterness_delta) if bitterness_delta is not None else None,
        body_delta=_to_decimal(body_delta) if body_delta is not None else None,
        freshness_delta=_to_decimal(freshness_delta) if freshness_delta is not None else None,
        aroma_delta_json=aroma_delta_json,
        parsed_summary=parsed_summary,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_feedbacks_by_sample_recommendation(
    db: Session,
    sample_recommendation_id: str | Any,
) -> list[SampleFeedback]:
    return (
        db.query(SampleFeedback)
        .filter(SampleFeedback.sample_recommendation_id == sample_recommendation_id)
        .order_by(SampleFeedback.created_at.asc())
        .all()
    )


# ============================================================
# 10. 최종 추천 저장
# ============================================================

def create_final_recommendation(
    db: Session,
    guest_session_id: str | Any,
    sample_recommendation_id: str | Any,
    final_cocktail_id: int,
    used_feedback_ids_json: Optional[list[str]],
    is_adjusted_recipe: bool,
    final_recipe_snapshot_json: dict,
    final_reason_text: str,
) -> FinalRecommendation:
    row = FinalRecommendation(
        guest_session_id=guest_session_id,
        sample_recommendation_id=sample_recommendation_id,
        final_cocktail_id=final_cocktail_id,
        used_feedback_ids_json=used_feedback_ids_json,
        is_adjusted_recipe=is_adjusted_recipe,
        final_recipe_snapshot_json=final_recipe_snapshot_json,
        final_reason_text=final_reason_text,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row

def get_final_recommendation_by_id(
    db: Session,
    final_recommendation_id: str | Any,
) -> Optional[FinalRecommendation]:
    return (
        db.query(FinalRecommendation)
        .filter(FinalRecommendation.final_recommendation_id == final_recommendation_id)
        .first()
    )


# ============================================================
# 11. 제조 주문 저장
# ============================================================

def create_brew_order(
    db: Session,
    guest_session_id: str | Any,
    order_type: str,
    total_volume_ml: int,
    device_command_json: dict,
    final_recommendation_id: Optional[str | Any] = None,
    order_status: str = "PENDING",
) -> BrewOrder:
    row = BrewOrder(
        guest_session_id=guest_session_id,
        final_recommendation_id=final_recommendation_id,
        order_type=order_type,
        order_status=order_status,
        total_volume_ml=total_volume_ml,
        device_command_json=device_command_json,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ============================================================
# 12. 평가 저장
# ============================================================

# ============================================================
# 13. 재료 전체 로드 (추천 엔진용 N+1 방지)
# ============================================================

def list_recommended_cocktail_ids_by_guest(
    db: Session,
    guest_session_id: str | Any,
) -> list[int]:
    """해당 게스트의 모든 sample_recommendation에서 추천된 칵테일 ID 목록 반환."""
    rows = (
        db.query(SampleRecommendation.recommended_cocktail_id)
        .filter(SampleRecommendation.guest_session_id == guest_session_id)
        .all()
    )
    return [r.recommended_cocktail_id for r in rows]


def get_available_ingredient_ids(db: Session) -> Optional[set[int]]:
    """현재 재고가 충분한 ingredient_id 집합.

    - InventoryItem 테이블이 비어있으면 None 반환 (필터 비활성화 = 데이터 미세팅 상태)
    - 데이터가 있으면 is_available=True AND 현재량 >= 임계값인 ingredient_id 집합 반환
    """
    from app.db.models import InventoryItem
    total = db.query(InventoryItem).count()
    if total == 0:
        return None
    rows = (
        db.query(InventoryItem.ingredient_id)
        .filter(InventoryItem.is_available.is_(True))
        .filter(InventoryItem.current_volume_ml >= InventoryItem.low_threshold_ml)
        .all()
    )
    return {r[0] for r in rows}


def get_all_recipes_with_ingredients(db: Session) -> dict[int, list[tuple]]:
    """cocktail_id → [(Recipe, Ingredient)] 딕셔너리로 한 번에 로드"""
    from app.db.models import Recipe, Ingredient
    rows = (
        db.query(Recipe, Ingredient)
        .join(Ingredient, Recipe.ingredient_id == Ingredient.ingredient_id)
        .all()
    )
    result: dict[int, list[tuple]] = {}
    for recipe, ingredient in rows:
        result.setdefault(recipe.cocktail_id, []).append((recipe, ingredient))
    return result


def create_evaluation_log(
    db: Session,
    guest_session_id: str | Any,
    final_satisfaction_score: Optional[float] = None,
    would_reorder: Optional[bool] = None,
    review_text: Optional[str] = None,
    rag_precision_at_k: Optional[float] = None,
    context_utilization_rate: Optional[float] = None,
    llm_comparison_json: Optional[dict] = None,
) -> EvaluationLog:
    row = EvaluationLog(
        guest_session_id=guest_session_id,
        final_satisfaction_score=_to_decimal(final_satisfaction_score)
        if final_satisfaction_score is not None
        else None,
        would_reorder=would_reorder,
        review_text=review_text,
        rag_precision_at_k=_to_decimal(rag_precision_at_k)
        if rag_precision_at_k is not None
        else None,
        context_utilization_rate=_to_decimal(context_utilization_rate)
        if context_utilization_rate is not None
        else None,
        llm_comparison_json=llm_comparison_json,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row