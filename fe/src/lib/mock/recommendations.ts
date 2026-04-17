import type {
  GuestPreferences,
  CocktailRecommendation,
  FeedbackAnalysis,
  SensoryAdjustment,
  RecipeAdjustment,
} from '@/lib/types';
import { MOCK_COCKTAILS, getRandomCocktail } from './cocktails';

// 취향 기반 1차 추천 (mock)
export function getInitialRecommendation(
  preferences: GuestPreferences
): CocktailRecommendation {
  // 알코올 없음 → 논알코올
  if (preferences.alcoholTolerance === 'none') {
    return MOCK_COCKTAILS.find((c) => c.id === 'virgin-mojito')!;
  }

  // 쓴맛 강하고 경험자 → Old Fashioned
  if (preferences.bitterness >= 4 && preferences.experience === 'experienced') {
    return MOCK_COCKTAILS.find((c) => c.id === 'old-fashioned')!;
  }

  // 단맛 강하고 꽃향기 선호 → Hugo Spritz
  if (preferences.sweetness >= 4 && preferences.aromas.includes('floral')) {
    return MOCK_COCKTAILS.find((c) => c.id === 'hugo-spritz')!;
  }

  // 신맛 강하고 강도 강함 → Margarita
  if (preferences.sourness >= 4 && preferences.intensity === 'strong') {
    return MOCK_COCKTAILS.find((c) => c.id === 'margarita')!;
  }

  // 커피향 선호 → Espresso Martini
  if (preferences.aromas.includes('smoky') || preferences.bitterness >= 3) {
    return MOCK_COCKTAILS.find((c) => c.id === 'espresso-martini')!;
  }

  // 초보자 가벼운 → Aperol Spritz
  if (preferences.experience === 'beginner' || preferences.intensity === 'light') {
    return MOCK_COCKTAILS.find((c) => c.id === 'aperol-spritz')!;
  }

  // 기본 → Moscow Mule
  return MOCK_COCKTAILS.find((c) => c.id === 'moscow-mule')!;
}

// 자연어 피드백 분석 (mock)
export function analyzeFeedback(rawFeedback: string): FeedbackAnalysis {
  const text = rawFeedback.toLowerCase();
  const adjustments: SensoryAdjustment[] = [];

  if (text.includes('달') || text.includes('달콤') || text.includes('달아')) {
    if (text.includes('너무') || text.includes('덜') || text.includes('줄')) {
      adjustments.push({ dimension: 'sweetness', direction: 'decrease', magnitude: 0.4, label: '단맛 줄이기' });
    } else {
      adjustments.push({ dimension: 'sweetness', direction: 'increase', magnitude: 0.3, label: '단맛 높이기' });
    }
  }

  if (text.includes('시') || text.includes('새콤') || text.includes('신') || text.includes('상큼')) {
    if (text.includes('더') || text.includes('좀 더') || text.includes('좋겠')) {
      adjustments.push({ dimension: 'sourness', direction: 'increase', magnitude: 0.35, label: '산미 높이기' });
    } else if (text.includes('덜') || text.includes('너무')) {
      adjustments.push({ dimension: 'sourness', direction: 'decrease', magnitude: 0.3, label: '산미 줄이기' });
    }
  }

  if (text.includes('쓴') || text.includes('쓴맛')) {
    if (text.includes('너무') || text.includes('덜')) {
      adjustments.push({ dimension: 'bitterness', direction: 'decrease', magnitude: 0.4, label: '쓴맛 줄이기' });
    } else {
      adjustments.push({ dimension: 'bitterness', direction: 'increase', magnitude: 0.25, label: '쓴맛 높이기' });
    }
  }

  if (text.includes('강') || text.includes('독') || text.includes('세') || text.includes('알코올')) {
    if (text.includes('덜') || text.includes('약') || text.includes('낮')) {
      adjustments.push({ dimension: 'strength', direction: 'decrease', magnitude: 0.5, label: '도수 낮추기' });
    } else {
      adjustments.push({ dimension: 'strength', direction: 'increase', magnitude: 0.4, label: '도수 높이기' });
    }
  }

  if (text.includes('향') || text.includes('냄새') || text.includes('향기')) {
    adjustments.push({ dimension: 'aroma', direction: 'increase', magnitude: 0.3, label: '아로마 강조' });
  }

  // 아무 키워드도 없으면 기본 조정
  if (adjustments.length === 0) {
    adjustments.push({ dimension: 'sweetness', direction: 'decrease', magnitude: 0.2, label: '전반적 밸런스 조정' });
  }

  const summaryParts = adjustments.map((a) =>
    `${a.label} (${Math.round(a.magnitude * 100)}%)`
  );

  return {
    originalText: rawFeedback,
    adjustments,
    summary: `피드백 반영: ${summaryParts.join(', ')}`,
  };
}

// 최종 추천 (피드백 반영 버전)
export function getFinalRecommendation(
  initial: CocktailRecommendation,
  feedbackAnalysis: FeedbackAnalysis
): CocktailRecommendation {
  const adjustments: RecipeAdjustment[] = feedbackAnalysis.adjustments.map((adj) => {
    const changeLabel = adj.direction === 'increase' ? '증가' : '감소';
    const pct = Math.round(adj.magnitude * 100);

    const ingredientMap: Record<string, string> = {
      sweetness: '시럽 / 감미료',
      sourness: '레몬주스 / 라임주스',
      bitterness: '비터스',
      strength: '알코올 베이스',
      aroma: '허브 / 과일 가니쉬',
    };

    return {
      ingredient: ingredientMap[adj.dimension] ?? adj.dimension,
      change: `${pct}% ${changeLabel}`,
      reason: adj.label,
    };
  });

  // 조정 내용이 많이 달라지면 다른 칵테일로 전환
  const strongAdjustments = feedbackAnalysis.adjustments.filter(
    (a) => a.magnitude >= 0.4
  );

  if (strongAdjustments.length >= 2) {
    const alternative = getRandomCocktail(initial.id);
    return {
      ...alternative,
      adjustments,
      reason: `${initial.name}의 피드백을 반영해 더 잘 맞는 ${alternative.name}으로 조정했습니다. ${alternative.reason}`,
    };
  }

  return {
    ...initial,
    adjustments,
    reason: `피드백을 반영해 ${initial.name}의 레시피를 조정했습니다. ${feedbackAnalysis.summary}`,
  };
}
