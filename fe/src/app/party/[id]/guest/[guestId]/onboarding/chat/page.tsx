'use client';

import { use, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { usePartyStore, useGuest } from '@/lib/store/partyStore';
import { getRecommendation, type RecommendationResponse } from '@/lib/api';
import type { CocktailRecommendation } from '@/lib/types';
import FollowUpQuestions from '@/components/onboarding/FollowUpQuestions';
import StepIndicator from '@/components/ui/StepIndicator';

const STEPS = [
  { label: '취향', icon: '🎯' },
  { label: '추가질문', icon: '💬' },
];

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

export default function OnboardingChatPage({
  params,
}: {
  params: Promise<{ id: string; guestId: string }>;
}) {
  const { id, guestId } = use(params);
  const router = useRouter();
  const guest = useGuest(id, guestId);
  const store = usePartyStore();
  const [recommending, setRecommending] = useState(false);
  const [recError, setRecError] = useState('');

  if (!guest) {
    return (
      <main className="min-h-screen flex items-center justify-center px-4">
        <div className="text-center">
          <p className="text-zinc-400">게스트를 찾을 수 없어요</p>
          <Link href={`/party/${id}`} className="text-amber-400 text-sm mt-2 inline-block">
            파티로 돌아가기
          </Link>
        </div>
      </main>
    );
  }

  if (!guest.dbId) {
    return (
      <main className="min-h-screen flex items-center justify-center px-4">
        <div className="text-center">
          <p className="text-zinc-400 mb-2">세션 정보가 없어요</p>
          <Link href={`/party/${id}/guest/${guestId}/onboarding`} className="text-amber-400 text-sm">
            처음부터 다시 시작하기
          </Link>
        </div>
      </main>
    );
  }

  const handleChatDone = async () => {
    setRecommending(true);
    setRecError('');
    try {
      const data = await getRecommendation(guest.dbId!);
      if (data.status === 'ok' && data.top_k.length > 0) {
        const rec = mapApiToRecommendation(data);
        store.setTastingRecommendation(id, guestId, rec);
        store.setSampleRecommendationId(id, guestId, data.sample_recommendation_id);
        router.push(`/party/${id}/guest/${guestId}/tasting`);
      } else {
        setRecError(`추천 실패 (${data.status})${(data as {message?:string}).message ? ': ' + (data as {message?:string}).message : ''}`);
        setRecommending(false);
      }
    } catch (e) {
      setRecError(`서버 오류: ${e instanceof Error ? e.message : String(e)}`);
      setRecommending(false);
    }
  };

  return (
    <main className="min-h-screen px-4 py-8 max-w-lg mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <Link
            href={`/party/${id}/guest/${guestId}/onboarding`}
            className="text-xs text-zinc-500 hover:text-zinc-300"
          >
            ← 취향 입력으로
          </Link>
          <h1 className="text-xl font-bold text-zinc-100 mt-1">
            {guest.name}님의 취향 입력
          </h1>
        </div>
      </div>

      <div className="flex justify-center mb-8">
        <StepIndicator steps={STEPS} current={1} />
      </div>

      {recommending ? (
        <div className="flex flex-col items-center justify-center gap-4 py-16">
          <div className="text-5xl animate-pulse">🍹</div>
          <p className="text-zinc-300 font-medium">취향에 맞는 칵테일을 찾고 있어요...</p>
        </div>
      ) : (
        <>
          <FollowUpQuestions gid={guest.dbId} onSubmit={handleChatDone} />
          {recError && (
            <p className="text-red-400 text-sm text-center mt-4">{recError}</p>
          )}
        </>
      )}
    </main>
  );
}
