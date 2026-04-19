from sqlalchemy import (
    Column,
    String,
    Text,
    Boolean,
    Integer,
    DECIMAL,
    TIMESTAMP,
    ForeignKey,
    CheckConstraint,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import uuid

from app.db.database import Base


# ============================================================
#  1. 세션 관리
# ============================================================

class PartySession(Base):
    __tablename__ = "party_sessions"

    party_session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_name = Column(String(100), nullable=True)
    session_status = Column(String(20), nullable=False, default="ACTIVE")
    started_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    ended_at = Column(TIMESTAMP, nullable=True)
    demo_mode = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        CheckConstraint(
            "session_status IN ('ACTIVE','ENDED')",
            name="chk_party_sessions_status",
        ),
    )


class PartySpaceAnalysis(Base):
    __tablename__ = "party_space_analysis"

    space_analysis_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    party_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("party_sessions.party_session_id", ondelete="CASCADE"),
        nullable=False,
    )
    image_path = Column(Text, nullable=False)
    mood_tags_json = Column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )  # {"신나는": 0.91, ...}
    mood_weight = Column(DECIMAL(3, 2), nullable=False, default=0.30)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())


class GuestSession(Base):
    __tablename__ = "guest_sessions"

    guest_session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    party_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("party_sessions.party_session_id", ondelete="CASCADE"),
        nullable=False,
    )
    guest_label = Column(String(50), nullable=False)
    session_status = Column(String(20), nullable=False, default="IN_PROGRESS")
    conversation_stage = Column(String(30), nullable=False, server_default=text("'INIT'"), default="INIT")
    feedback_round = Column(Integer, nullable=False, server_default=text("0"), default=0)
    started_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    ended_at = Column(TIMESTAMP, nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "session_status IN ('IN_PROGRESS','COMPLETED')",
            name="chk_guest_sessions_status",
        ),
    )


# ============================================================
#  2. 사용자 입력 및 취향
# ============================================================

class InitialTagResponse(Base):
    __tablename__ = "initial_tag_responses"

    tag_response_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    guest_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("guest_sessions.guest_session_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    familiarity_tag = Column(String(20), nullable=True)
    taste_tags_json = Column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    strength_tag = Column(String(20), nullable=False)
    aroma_tags_json = Column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())


class PreferenceSlot(Base):
    __tablename__ = "preference_slots"

    slot_profile_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    guest_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("guest_sessions.guest_session_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    # enum 값은 영문 소문자 (preference_agent.SLOT_SCHEMA 참조)
    party_purpose = Column(String(30), nullable=True)
    current_mood = Column(String(30), nullable=True)
    # {sweet|sour|bitter|body|creamy|freshness: low|medium|high}
    taste_profile_json = Column(JSONB, nullable=True)
    # {woody|minty|fruity|citrus|floral|coffee|herbal: low|medium|high}
    aroma_profile_json = Column(JSONB, nullable=True)
    strength_preference = Column(String(10), nullable=True)
    #finish_preference = Column(String(10), nullable=True)
    disliked_bases_json = Column(JSONB, nullable=True)
    favorite_drinks_json = Column(JSONB, nullable=True)
    slot_completion_score = Column(DECIMAL(5, 2), nullable=False, default=0.00)
    updated_at = Column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class PreferenceVector(Base):
    __tablename__ = "preference_vectors"

    vector_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    guest_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("guest_sessions.guest_session_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    sweetness_score = Column(DECIMAL(3, 1), nullable=False, default=3.0)
    bitterness_score = Column(DECIMAL(3, 1), nullable=False, default=3.0)
    sourness_score = Column(DECIMAL(3, 1), nullable=False, default=3.0)
    freshness_score = Column(DECIMAL(3, 1), nullable=False, default=3.0)
    body_score = Column(DECIMAL(3, 1), nullable=False, default=3.0)
    herbal_score = Column(DECIMAL(3, 1), nullable=False, default=2.0)
    citrus_score = Column(DECIMAL(3, 1), nullable=False, default=2.0)
    alcohol_score = Column(DECIMAL(3, 1), nullable=False, default=3.0)
    version = Column(Integer, nullable=False, default=1)
    updated_at = Column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "sweetness_score BETWEEN 1.0 AND 5.0",
            name="chk_preference_vectors_sweetness_score",
        ),
        CheckConstraint(
            "bitterness_score BETWEEN 1.0 AND 5.0",
            name="chk_preference_vectors_bitterness_score",
        ),
        CheckConstraint(
            "sourness_score BETWEEN 1.0 AND 5.0",
            name="chk_preference_vectors_sourness_score",
        ),
        CheckConstraint(
            "freshness_score BETWEEN 1.0 AND 5.0",
            name="chk_preference_vectors_freshness_score",
        ),
        CheckConstraint(
            "body_score BETWEEN 1.0 AND 5.0",
            name="chk_preference_vectors_body_score",
        ),
        CheckConstraint(
            "herbal_score BETWEEN 0.0 AND 3.0",
            name="chk_preference_vectors_herbal_score",
        ),
        CheckConstraint(
            "citrus_score BETWEEN 0.0 AND 3.0",
            name="chk_preference_vectors_citrus_score",
        ),
        CheckConstraint(
            "alcohol_score BETWEEN 0.0 AND 5.0",
            name="chk_preference_vectors_alcohol_score",
        ),
    )


class DialogueTurn(Base):
    __tablename__ = "dialogue_turns"

    turn_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    guest_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("guest_sessions.guest_session_id", ondelete="CASCADE"),
        nullable=False,
    )
    turn_index = Column(Integer, nullable=False)
    speaker_role = Column(String(20), nullable=False)
    utterance_text = Column(Text, nullable=False)
    extracted_slots_json = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "speaker_role IN ('SYSTEM','USER','LLM')",
            name="chk_dialogue_turns_speaker_role",
        ),
    )


# ============================================================
#  3. 음료 마스터
# ============================================================

class Cocktail(Base):
    __tablename__ = "cocktails"

    cocktail_id = Column(Integer, primary_key=True, autoincrement=True)
    name_kr = Column(String(100), nullable=False)
    name_en = Column(String(100), nullable=True)
    category = Column(String(50), nullable=False)
    is_non_alcoholic = Column(Boolean, nullable=False, default=False)

    sweet_level = Column(DECIMAL(3, 1), nullable=True)
    sour_level = Column(DECIMAL(3, 1), nullable=True)
    bitter_level = Column(DECIMAL(3, 1), nullable=True)
    body_level = Column(DECIMAL(3, 1), nullable=True)
    freshness_level = Column(DECIMAL(3, 1), nullable=True)
    creamy_level = Column(DECIMAL(3, 1), nullable=True)
    spicy_level = Column(DECIMAL(3, 1), nullable=True)
    nutty_level = Column(DECIMAL(3, 1), nullable=True)

    mood_tag = Column(String(50), nullable=True)
    description = Column(Text, nullable=False)
    embedding = Column(Vector(1024), nullable=True)  # Qwen3-Embedding (pgvector)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())

    __table_args__ = (
        CheckConstraint("sweet_level BETWEEN 0 AND 5", name="chk_cocktails_sweet_level"),
        CheckConstraint("sour_level BETWEEN 0 AND 5", name="chk_cocktails_sour_level"),
        CheckConstraint("bitter_level BETWEEN 0 AND 5", name="chk_cocktails_bitter_level"),
        CheckConstraint("body_level BETWEEN 0 AND 5", name="chk_cocktails_body_level"),
        CheckConstraint("freshness_level BETWEEN 0 AND 5", name="chk_cocktails_freshness_level"),
        CheckConstraint("creamy_level BETWEEN 0 AND 5", name="chk_cocktails_creamy_level"),
        CheckConstraint("spicy_level BETWEEN 0 AND 5", name="chk_cocktails_spicy_level"),
        CheckConstraint("nutty_level BETWEEN 0 AND 5", name="chk_cocktails_nutty_level"),
    )


# ============================================================
#  4. 재료 및 재고
# ============================================================

class Ingredient(Base):
    __tablename__ = "ingredients"

    ingredient_id = Column(Integer, primary_key=True, autoincrement=True)
    ingredients_name = Column(String(100), nullable=False)
    ingredient_type = Column(String(30), nullable=False)
    pump_no = Column(Integer, nullable=True)
    is_pumpable = Column(Boolean, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)

    # 감각 점수
    sweet_score = Column(DECIMAL(3, 1), default=0.0)
    sour_score = Column(DECIMAL(3, 1), default=0.0)
    bitter_score = Column(DECIMAL(3, 1), default=0.0)
    body_score = Column(DECIMAL(3, 1), default=0.0)
    freshness_score = Column(DECIMAL(3, 1), default=0.0)
    creamy_score = Column(DECIMAL(3, 1), default=0.0)
    spicy_score = Column(DECIMAL(3, 1), default=0.0)
    nutty_score = Column(DECIMAL(3, 1), default=0.0)
    fruity_score = Column(DECIMAL(3, 1), default=0.0)
    woody_score = Column(DECIMAL(3, 1), default=0.0)
    coffee_score = Column(DECIMAL(3, 1), default=0.0)
    herbal_score = Column(DECIMAL(3, 1), default=0.0)
    floral_score = Column(DECIMAL(3, 1), default=0.0)
    citrus_score = Column(DECIMAL(3, 1), default=0.0)
    minty_score = Column(DECIMAL(3, 1), default=0.0)
    sensory_note = Column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "ingredient_type IN ('BASE','MIXER','SYRUP','JUICE','TOPPING','GARNISH','OTHER')",
            name="chk_ingredients_ingredient_type",
        ),
    )


class Recipe(Base):
    __tablename__ = "recipes"

    recipe_id = Column(Integer, primary_key=True, autoincrement=True)
    cocktail_id = Column(
        Integer,
        ForeignKey("cocktails.cocktail_id", ondelete="CASCADE"),
        nullable=False,
    )
    ingredient_id = Column(
        Integer,
        ForeignKey("ingredients.ingredient_id"),
        nullable=False,
    )
    amount_ml = Column(DECIMAL(6, 2), nullable=False)
    step_order = Column(Integer, nullable=False)
    is_optional = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint(
            "cocktail_id", "ingredient_id", "step_order",
            name="uq_recipes_cocktail_ingredient_step",
        ),
    )


class InventoryItem(Base):
    __tablename__ = "inventory_items"

    inventory_id = Column(Integer, primary_key=True, autoincrement=True)
    ingredient_id = Column(
        Integer,
        ForeignKey("ingredients.ingredient_id"),
        nullable=False,
        unique=True,
    )
    current_volume_ml = Column(DECIMAL(8, 2), nullable=False, default=0.0)
    low_threshold_ml = Column(DECIMAL(8, 2), nullable=False, default=50.0)
    is_available = Column(Boolean, nullable=False, default=True)


# ============================================================
#  5. 추천 및 피드백
# ============================================================

class SampleRecommendation(Base):
    __tablename__ = "sample_recommendations"

    sample_recommendation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    guest_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("guest_sessions.guest_session_id", ondelete="CASCADE"),
        nullable=False,
    )
    space_analysis_id = Column(
        UUID(as_uuid=True),
        ForeignKey("party_space_analysis.space_analysis_id"),
        nullable=False,
    )
    recommended_cocktail_id = Column(
        Integer,
        ForeignKey("cocktails.cocktail_id"),
        nullable=False,
    )
    recommendation_reason = Column(Text, nullable=False)
    rag_retrieved_ids_json = Column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    recipe_snapshot_json = Column(JSONB, nullable=False)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())


class SampleFeedback(Base):
    __tablename__ = "sample_feedback"

    sample_feedback_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sample_recommendation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sample_recommendations.sample_recommendation_id", ondelete="CASCADE"),
        nullable=False,
    )
    feedback_text = Column(Text, nullable=False)
    feedback_intent = Column(String(20), nullable=False)
    sweetness_delta = Column(DECIMAL(3, 1), nullable=True)
    sourness_delta = Column(DECIMAL(3, 1), nullable=True)
    bitterness_delta = Column(DECIMAL(3, 1), nullable=True)
    body_delta = Column(DECIMAL(3, 1), nullable=True)
    freshness_delta = Column(DECIMAL(3, 1), nullable=True)
    aroma_delta_json = Column(JSONB, nullable=True)
    parsed_summary = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "feedback_intent IN ('ACCEPT','ADJUST','REJECT')",
            name="chk_sample_feedback_intent",
        ),
    )


class FinalRecommendation(Base):
    __tablename__ = "final_recommendations"

    final_recommendation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    guest_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("guest_sessions.guest_session_id", ondelete="CASCADE"),
        nullable=False,
    )
    sample_recommendation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sample_recommendations.sample_recommendation_id"),
        nullable=False,
    )
    final_cocktail_id = Column(
        Integer,
        ForeignKey("cocktails.cocktail_id"),
        nullable=False,
    )
    used_feedback_ids_json = Column(JSONB, nullable=True)
    is_adjusted_recipe = Column(Boolean, nullable=False, default=False)
    final_recipe_snapshot_json = Column(JSONB, nullable=False)
    final_reason_text = Column(Text, nullable=False)
    confirmed_at = Column(TIMESTAMP, nullable=False, server_default=func.now())


# ============================================================
#  6. 제조
# ============================================================

class BrewOrder(Base):
    __tablename__ = "brew_orders"

    order_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    guest_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("guest_sessions.guest_session_id", ondelete="CASCADE"),
        nullable=False,
    )
    final_recommendation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("final_recommendations.final_recommendation_id"),
        nullable=True,
    )  # SAMPLE 주문 시 NULL
    order_type = Column(String(20), nullable=False)
    order_status = Column(String(20), nullable=False, default="PENDING")
    total_volume_ml = Column(Integer, nullable=False)
    device_command_json = Column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    started_at = Column(TIMESTAMP, nullable=True)
    finished_at = Column(TIMESTAMP, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "order_type IN ('SAMPLE','FINAL')",
            name="chk_brew_orders_order_type",
        ),
        CheckConstraint(
            "order_status IN ('PENDING','BREWING','DONE','FAILED')",
            name="chk_brew_orders_order_status",
        ),
    )


# ============================================================
#  7. 평가
# ============================================================

class EvaluationLog(Base):
    __tablename__ = "evaluation_logs"

    evaluation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    guest_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("guest_sessions.guest_session_id", ondelete="CASCADE"),
        nullable=False,
    )
    final_satisfaction_score = Column(DECIMAL(3, 1), nullable=True)
    would_reorder = Column(Boolean, nullable=True)
    review_text = Column(Text, nullable=True)
    rag_precision_at_k = Column(DECIMAL(5, 2), nullable=True)
    context_utilization_rate = Column(DECIMAL(5, 2), nullable=True)
    llm_comparison_json = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "final_satisfaction_score BETWEEN 1.0 AND 5.0",
            name="chk_evaluation_logs_final_satisfaction_score",
        ),
    )