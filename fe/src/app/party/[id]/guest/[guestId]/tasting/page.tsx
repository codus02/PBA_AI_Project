'use client';

import { use, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { usePartyStore, useGuest } from '@/lib/store/partyStore';
import { analyzeFeedback, getFinalRecommendation } from '@/lib/mock/recommendations';
import { now } from '@/lib/utils';
import CocktailCard from '@/components/recommendation/CocktailCard';
import FeedbackInput from '@/components/recommendation/FeedbackInput';

export default function TastingPage({
  params,
}: {
  params: Promise<{ id: string; guestId: string }>;
}) {
  const { id, guestId } = use(params);
  const router = useRouter();
  const guest = useGuest(id, guestId);
  const store = usePartyStore();
  const [analyzing, setAnalyzing] = useState(false);

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
    setAnalyzing(true);

    // 피드백 분석
    const feedbackAnalysis = analyzeFeedback(feedback);
    store.setFeedback(id, guestId, feedback);
    store.setFeedbackAnalysis(id, guestId, feedbackAnalysis);
    store.addConversationEntry(id, guestId, {
      timestamp: now(),
      type: 'feedback',
      content: feedback,
    });
    store.addFeedbackEntry(id, guestId, {
      timestamp: now(),
      rawFeedback: feedback,
      adjustments: feedbackAnalysis.adjustments,
    });

    // 최종 추천 생성
    const finalRec = getFinalRecommendation(guest.tastingRecommendation!, feedbackAnalysis);
    store.setFinalRecommendation(id, guestId, finalRec);
    store.addRecommendationEntry(id, guestId, {
      timestamp: now(),
      stage: 'final',
      cocktailId: finalRec.id,
      cocktailName: finalRec.name,
    });

    await new Promise((r) => setTimeout(r, 600));
    setAnalyzing(false);
    router.push(`/party/${id}/guest/${guestId}/final`);
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
      </div>
    </main>
  );
}
