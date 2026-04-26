'use client';

import { use, useState } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { usePartyStore, useGuest } from '@/lib/store/partyStore';
import { submitFeedback, getFinalOutput, type RecommendationResponse } from '@/lib/api';
import type { CocktailRecommendation } from '@/lib/types';
import { now } from '@/lib/utils';
import CocktailCard from '@/components/recommendation/CocktailCard';
import FeedbackInput from '@/components/recommendation/FeedbackInput';

const MOCK_RECOMMENDATION: CocktailRecommendation = {
  id: 'preview-1',
  name: '아페롤 스프리츠',
  description: '이탈리아에서 온 상큼하고 가벼운 아페리티프 칵테일',
  reason:
    '달콤하고 상큼한 맛을 좋아하시고 알코올은 약하게 선호하신다고 하셨어요.\n아페롤의 쌉쌀한 오렌지 향과 프로세코의 가벼운 버블이 어우러져\n부담 없이 즐기실 수 있는 칵테일이에요.',
  imageEmoji: '🍊',
  tags: ['상큼한', '가벼운', '과일향'],
  recipe: [
    { ingredient: '아페롤', amount: 60, unit: 'ml' },
    { ingredient: '프로세코', amount: 90, unit: 'ml' },
    { ingredient: '탄산수', amount: 30, unit: 'ml' },
    { ingredient: '오렌지 슬라이스', amount: 1, unit: '개' },
  ],
};

function mapApiToRecommendation(data: RecommendationResponse): CocktailRecommendation {
  const top = data.top_k[0];
  return {
    id: String(top.cocktail_id),
    name: top.name_kr,
    reason: top.reason_parts.join('\n'),
    imageEmoji: '🍹',
    tags: [],
    recipe: [],
  };
}

export default function TastingPage({
  params,
}: {
  params: Promise<{ id: string; guestId: string }>;
}) {
  const { id, guestId } = use(params);
  const router = useRouter();
  const searchParams = useSearchParams();
  const isPreview = searchParams.get('preview') === 'true';
  const guest = useGuest(id, guestId);
  const store = usePartyStore();
  const [analyzing, setAnalyzing] = useState(false);
  const [feedbackError, setFeedbackError] = useState('');

  if (isPreview) {
    return (
      <main className="min-h-screen px-4 py-8 max-w-lg mx-auto">
        <div className="mb-6">
          <span className="text-xs text-zinc-500">← 파티로</span>
          <h1 className="text-xl font-bold text-zinc-100 mt-1">미리보기님의 시음 추천</h1>
          <p className="text-sm text-zinc-400 mt-0.5">먼저 시음해보고 피드백을 남겨주세요</p>
          <span className="inline-block mt-2 px-2 py-0.5 rounded text-xs bg-amber-400/10 text-amber-400 border border-amber-400/20">
            디자인 미리보기 모드
          </span>
        </div>
        <div className="flex flex-col gap-5">
          <CocktailCard cocktail={MOCK_RECOMMENDATION} stage="tasting" />
          <FeedbackInput onSubmit={() => {}} loading={false} />
        </div>
      </main>
    );
  }

  if (!guest) {
    return (
      <main className="min-h-screen flex items-center justify-center px-4">
        <p className="text-zinc-400">게스트를 찾을 수 없어요</p>
      </main>
    );
  }

  if (!guest.tastingRecommendation) {
    return (
      <main className="min-h-screen flex items-center justify-center px-4">
        <div className="text-center">
          <p className="text-zinc-400 mb-3">아직 추천이 생성되지 않았어요</p>
          <Link
            href={`/party/${id}/guest/${guestId}/onboarding`}
            className="text-amber-400 text-sm hover:underline"
          >
            취향 입력하러 가기
          </Link>
        </div>
      </main>
    );
  }

  const handleFeedback = async (feedback: string) => {
    if (!guest.dbId || !guest.sampleRecommendationId) {
      setFeedbackError(`세션 정보 누락 — dbId: ${guest.dbId ?? 'null'}, sampleId: ${guest.sampleRecommendationId ?? 'null'}`);
      return;
    }
    setAnalyzing(true);
    setFeedbackError('');

    store.setFeedback(id, guestId, feedback);
    store.addConversationEntry(id, guestId, { timestamp: now(), type: 'feedback', content: feedback });

    try {
      const result = await submitFeedback(guest.dbId, guest.sampleRecommendationId, feedback);

      if (result.status === 'accepted' || result.status === 'force_finalized') {
        // ACCEPT → 최종 확정, final output 가져오기
        const finalId = result.final_recommendation_id!;
        store.setFinalRecommendationId(id, guestId, finalId);

        const output = await getFinalOutput(finalId);
        const finalRec: CocktailRecommendation = {
          id: String(output.cocktail_id),
          name: output.cocktail_name,
          reason: guest.tastingRecommendation!.reason,
          imageEmoji: '🍹',
          tags: [],
          recipe: output.steps.map((s) => ({
            ingredient: s.ingredient_name,
            amount: s.amount_ml,
            unit: 'ml',
          })),
        };
        store.setFinalRecommendation(id, guestId, finalRec);
        store.addRecommendationEntry(id, guestId, {
          timestamp: now(),
          stage: 'final',
          cocktailId: String(output.cocktail_id),
          cocktailName: output.cocktail_name,
        });
        router.push(`/party/${id}/guest/${guestId}/final`);

      } else if (result.status === 're_recommended' && result.top_k && result.top_k.length > 0) {
        // ADJUST/REJECT → 새 추천 표시
        const newRec = mapApiToRecommendation(result as RecommendationResponse);
        store.setTastingRecommendation(id, guestId, newRec);
        store.setSampleRecommendationId(id, guestId, result.sample_recommendation_id!);
        store.addRecommendationEntry(id, guestId, {
          timestamp: now(),
          stage: 'tasting',
          cocktailId: newRec.id,
          cocktailName: newRec.name,
        });
        setAnalyzing(false);

      } else {
        setFeedbackError('응답을 처리할 수 없어요. 다시 시도해주세요.');
        setAnalyzing(false);
      }
    } catch {
      setFeedbackError('서버 오류가 발생했어요. 다시 시도해주세요.');
      setAnalyzing(false);
    }
  };

  return (
    <main className="min-h-screen px-4 py-8 max-w-lg mx-auto">
      <div className="mb-6">
        <Link href={`/party/${id}`} className="text-xs text-zinc-500 hover:text-zinc-300">
          ← 파티로
        </Link>
        <h1 className="text-xl font-bold text-zinc-100 mt-1">{guest.name}님의 시음 추천</h1>
        <p className="text-sm text-zinc-400 mt-0.5">먼저 시음해보고 피드백을 남겨주세요</p>
      </div>

      <div className="flex flex-col gap-5">
        <CocktailCard cocktail={guest.tastingRecommendation} stage="tasting" />
        <FeedbackInput onSubmit={handleFeedback} loading={analyzing} />
        {feedbackError && (
          <p className="text-red-400 text-sm text-center">{feedbackError}</p>
        )}
      </div>
    </main>
  );
}
