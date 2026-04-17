// ─────────────────────────────────────────
// Party
// ─────────────────────────────────────────
export interface Party {
  id: string;
  code: string;
  name: string;
  guests: Guest[];
  spaceImage?: string;
  spaceAnalysis?: SpaceAnalysis;
  createdAt: string;
}

// ─────────────────────────────────────────
// Guest
// ─────────────────────────────────────────
export type GuestStep =
  | 'onboarding'
  | 'tasting'
  | 'feedback'
  | 'final'
  | 'brew'
  | 'complete';

export interface Guest {
  id: string;
  partyId: string;
  name: string;
  step: GuestStep;
  preferences?: GuestPreferences;
  followUpAnswers?: FollowUpAnswer[];
  tastingRecommendation?: CocktailRecommendation;
  feedback?: string;
  feedbackAnalysis?: FeedbackAnalysis;
  finalRecommendation?: CocktailRecommendation;
  brewState?: BrewState;
  satisfaction?: number;
  logs: GuestLog;
}

// ─────────────────────────────────────────
// Preferences
// ─────────────────────────────────────────
export interface GuestPreferences {
  sweetness: number;         // 1–5
  sourness: number;
  bitterness: number;
  spiciness: number;
  aromas: AromaType[];
  alcoholTolerance: 'none' | 'low' | 'medium' | 'high';
  intensity: 'light' | 'medium' | 'strong';
  experience: 'beginner' | 'casual' | 'experienced';
}

export type AromaType =
  | 'citrus'
  | 'floral'
  | 'woody'
  | 'herbal'
  | 'fruity'
  | 'spicy'
  | 'smoky'
  | 'sweet';

// ─────────────────────────────────────────
// Space Analysis
// ─────────────────────────────────────────
export interface SpaceAnalysis {
  style: string;
  mood: string;
  colors: string[];
  atmosphere: string;
}

// ─────────────────────────────────────────
// Follow-up Questions
// ─────────────────────────────────────────
export interface FollowUpQuestion {
  id: string;
  question: string;
  options: string[];
}

export interface FollowUpAnswer {
  questionId: string;
  question: string;
  answer: string;
}

// ─────────────────────────────────────────
// Cocktail Recommendation
// ─────────────────────────────────────────
export interface CocktailRecommendation {
  id: string;
  name: string;
  description: string;
  reason: string;
  recipe: RecipeItem[];
  adjustments?: RecipeAdjustment[];
  imageEmoji: string;
  tags: string[];
}

export interface RecipeItem {
  ingredient: string;
  amount: number;
  unit: string;
}

export interface RecipeAdjustment {
  ingredient: string;
  change: string;
  reason: string;
}

// ─────────────────────────────────────────
// Feedback
// ─────────────────────────────────────────
export interface FeedbackAnalysis {
  originalText: string;
  adjustments: SensoryAdjustment[];
  summary: string;
}

export interface SensoryAdjustment {
  dimension: 'sweetness' | 'sourness' | 'bitterness' | 'strength' | 'aroma';
  direction: 'increase' | 'decrease';
  magnitude: number; // 0.0–1.0
  label: string;
}

// ─────────────────────────────────────────
// Brew
// ─────────────────────────────────────────
export type BrewStatus = 'idle' | 'preparing' | 'pouring' | 'mixing' | 'complete';

export interface BrewState {
  status: BrewStatus;
  progress: number; // 0–100
  startedAt?: string;
  completedAt?: string;
  devicePayload?: DevicePayload;
}

export interface DevicePayload {
  sessionId: string;
  guestId: string;
  cocktailId: string;
  cocktailName: string;
  recipe: RecipeItem[];
  adjustments: RecipeAdjustment[];
  timestamp: string;
  metadata: {
    temperature: string;
    mixingSpeed: string;
    glassType: string;
  };
}

// ─────────────────────────────────────────
// Logs
// ─────────────────────────────────────────
export interface GuestLog {
  conversationLog: ConversationEntry[];
  recommendationLog: RecommendationEntry[];
  feedbackLog: FeedbackEntry[];
  satisfaction?: number;
}

export interface ConversationEntry {
  id: string;
  timestamp: string;
  type: 'preference_input' | 'followup_q' | 'followup_a' | 'feedback' | 'system';
  content: string;
}

export interface RecommendationEntry {
  id: string;
  timestamp: string;
  stage: 'tasting' | 'final';
  cocktailId: string;
  cocktailName: string;
}

export interface FeedbackEntry {
  id: string;
  timestamp: string;
  rawFeedback: string;
  adjustments: SensoryAdjustment[];
}
