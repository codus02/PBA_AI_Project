const API_BASE = 'http://141.223.140.32:8000/api/v1';
//야 여기 로컬호스트잖아 당연히안되지;;;;

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return apiFetch<T>(path, {
    method: 'POST',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
}

function get<T>(path: string): Promise<T> {
  return apiFetch<T>(path);
}

// ─────────────────────────────────────────
// Party / Guest
// ─────────────────────────────────────────

export function createPartySession(name: string) {
  return post<{ party_session_id: string; session_name: string }>('/sessions/party', {
    session_name: name,
  });
}

export function createGuestSession(partyDbId: string, label: string) {
  return post<{ guest_session_id: string }>('/sessions/guest', {
    party_session_id: partyDbId,
    guest_label: label,
  });
}

// ─────────────────────────────────────────
// Tags (TasteForm 매핑)
// ─────────────────────────────────────────

const STRENGTH_MAP: Record<string, string> = {
  none: '무알콜',
  low: '약함',
  medium: '중간',
  high: '강함',
};

const TASTE_MAP: Record<string, string> = {
  sweet: '단맛',
  sour: '신맛',
  bitter: '쓴맛',
  refreshing: '청량함',
  body: '바디감',
  creamy: '크리미함',
};

const AROMA_MAP: Record<string, string> = {
  fruity: '과일향',
  herbal: '허브향',
  mint: '민트향',
  citrus: '시트러스향',
  woody: '우디향',
  coffee: '커피향',
  floral: '꽃향',
};

export function saveInitialTags(
  gid: string,
  prefs: { experience: string; alcoholTolerance: string; tasteTags: string[]; aromaTags: string[] }
) {
  return post<{ status: string }>(`/sessions/${gid}/tags`, {
    familiarity: prefs.experience,
    strength: STRENGTH_MAP[prefs.alcoholTolerance] ?? '중간',
    tastes: prefs.tasteTags.map((t) => TASTE_MAP[t] ?? t),
    aromas: prefs.aromaTags.map((a) => AROMA_MAP[a] ?? a),
  });
}

// ─────────────────────────────────────────
// Dialogue
// ─────────────────────────────────────────

export function startDialogue(gid: string) {
  return post<{ status: string; question: string }>(`/sessions/${gid}/start-dialogue`);
}

export interface DialogueResponse {
  status: string;
  question?: string;
  should_proceed: boolean;
  completion?: number;
}

export function sendMessage(gid: string, message: string) {
  return post<DialogueResponse>(`/sessions/${gid}/dialogue`, { message });
}

// ─────────────────────────────────────────
// Recommendation
// ─────────────────────────────────────────

export interface ApiCocktail {
  cocktail_id: number;
  name_kr: string;
  score: number;
  reason_parts: string[];
  source: string;
}

export interface RecommendationResponse {
  status: string;
  top_k: ApiCocktail[];
  sample_recommendation_id: string;
  llm_question?: string;
}

export function getRecommendation(gid: string) {
  return post<RecommendationResponse>(`/sessions/${gid}/recommend-sample`);
}

// ─────────────────────────────────────────
// Feedback
// ─────────────────────────────────────────

export interface FeedbackResponse {
  status: string;
  top_k?: ApiCocktail[];
  sample_recommendation_id?: string;
  final_recommendation_id?: string;
  final_cocktail_id?: number;
  llm_question?: string;
}

export function submitFeedback(gid: string, sampleId: string, text: string) {
  return post<FeedbackResponse>(`/sessions/${gid}/feedback`, {
    sample_recommendation_id: sampleId,
    feedback_text: text,
  });
}

// ─────────────────────────────────────────
// Final Output
// ─────────────────────────────────────────

export interface FinalOutputStep {
  ingredient_name: string;
  amount_ml: number;
  step_order: number;
  is_optional: boolean;
  adjusted: boolean;
}

export interface FinalOutputResponse {
  cocktail_id: number;
  cocktail_name: string;
  total_volume_ml: number;
  steps: FinalOutputStep[];
}

export function getFinalOutput(finalRecommendationId: string) {
  return get<FinalOutputResponse>(`/final-output/${finalRecommendationId}`);
}

// ─────────────────────────────────────────
// Evaluation
// ─────────────────────────────────────────

export function saveEvaluation(
  gid: string,
  finalRecommendationId: string,
  score: number,
  wouldReorder: boolean
) {
  return post(`/sessions/${gid}/evaluation`, {
    final_recommendation_id: finalRecommendationId,
    satisfaction_score: score,
    would_reorder: wouldReorder,
  });
}
